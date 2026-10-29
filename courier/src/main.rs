use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use courier::assistant;
use courier::chat::Agent;
use courier::config::Config;
use courier::engine::{self, Engine};
use courier::tools::{HostOptions, Recorder, ToolHost};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "courier", about = "An offline voice assistant: say \"Hey Courier\".")]
struct Cli {
    /// The configuration file (paths in it are relative to its folder).
    #[arg(long, global = true, default_value = "courier.toml")]
    config: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Listen on the microphone and answer through the speakers.
    Run,
    /// Type instead of speaking (tools run for real).
    Chat {
        /// Also speak the answers.
        #[arg(long)]
        voice: bool,
    },
    /// Run the assistant on a recording, in real time, and print or save its timeline.
    Sim {
        wav: PathBuf,
        /// Write the timeline as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Seconds of silence after the recording, for the answer to finish.
        #[arg(long, default_value_t = 6.0)]
        tail: f64,
        #[arg(long)]
        quiet: bool,
        /// Wake at these seconds of the recording (after its 2.5 s of lead-in) instead of by the
        /// detector, to test the rest apart from the wake word.
        #[arg(long, value_delimiter = ',')]
        wake_at: Vec<f64>,
    },
    /// List the plugins' tools; with --test, run the plugins' own tests.
    Plugins {
        #[arg(long)]
        test: bool,
    },
    /// Say something with the assistant's voice (to the speakers, or into a WAV file).
    Say {
        text: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// The audio devices it would use.
    Devices,
    /// The evaluations behind the README's results.
    Eval {
        #[command(subcommand)]
        what: courier::eval::What,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load(&cli.config)?;
    match cli.cmd {
        Cmd::Run => {
            let eng = Engine::load(&cfg.root)?;
            assistant::live(&cfg, &eng)
        }
        Cmd::Chat { voice } => chat(&cfg, voice),
        Cmd::Sim { wav, json, tail, quiet, wake_at } => {
            let eng = Engine::load(&cfg.root)?;
            let (mut samples, rate) = courier::wav::read(&wav)?;
            // A microphone has been on for a while when you speak: the wake word detector scores
            // once it has 2 s of audio, so the recording gets 2.5 s of silence before it.
            samples.splice(0..0, std::iter::repeat_n(0.0, (2.5 * rate as f64) as usize));
            let wake_at = wake_at.iter().map(|s| (s + 2.5) * 1000.0).collect();
            let marks = assistant::simulate(&cfg, &eng, samples, rate, tail, quiet, wake_at)?;
            if let Some(p) = json {
                std::fs::write(&p, serde_json::to_string_pretty(&marks)?)?;
            }
            Ok(())
        }
        Cmd::Plugins { test } => plugins(&cfg, test),
        Cmd::Say { text, out } => say(&cfg, &text, out.as_deref()),
        Cmd::Devices => devices(),
        Cmd::Eval { what } => courier::eval::run(&cfg, what),
    }
}

fn chat(cfg: &Config, voice: bool) -> Result<()> {
    let eng = Engine::load(&cfg.root)?;
    let llm = engine::Llm::new(&eng, &cfg.path(&cfg.models.llm), cfg.models.context, cfg.models.gpu_layers)?;
    let speaker = if voice {
        let (tx, _rx) = crossbeam_channel::unbounded();
        Some(courier::audio::Speaker::open(tx)?)
    } else {
        None
    };
    let services = Arc::new(courier::services::System { out: speaker.as_ref().map(|s| s.out.clone()), network: cfg.assistant.network });
    let mut opt = HostOptions::new(cfg.path(&cfg.assistant.plugins_dir), services);
    opt.data_dir = Some(cfg.data_dir());
    opt.config = cfg.plugins.clone();
    opt.place = cfg.assistant.place.clone();
    let host = ToolHost::load(opt)?;
    for e in &host.errors {
        eprintln!("plugin error: {e}");
    }
    let mut tts = if voice { Some(engine::Tts::new(&eng, &cfg.path(&cfg.models.voice), &cfg.path(&cfg.models.cmudict), 2)?) } else { None };
    let mut agent = Agent::new(llm, host);
    agent.place = cfg.assistant.place.clone();
    println!("{} tools. Type a request (an empty line quits).", agent.tools.len());
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Ok(());
        }
        let log = agent.respond(line.trim(), &mut |p| {
            print!("{p}");
            let _ = std::io::stdout().flush();
        })?;
        println!();
        for s in &log.steps {
            if let (Some(c), Some(r)) = (&s.call, &s.result) {
                println!("  -> {} {}  ({:.0} ms)", c.name, c.arguments, s.ms);
                println!("  <- {r}");
            }
        }
        if let (Some(t), Some(sp)) = (tts.as_mut(), speaker.as_ref()) {
            let pcm = t.speak(&log.reply, cfg.voice.speaker, cfg.voice.speed, 1.0)?;
            sp.out.speak(&pcm, t.rate());
        }
    }
}

fn plugins(cfg: &Config, test: bool) -> Result<()> {
    let mut opt = HostOptions::new(cfg.path(&cfg.assistant.plugins_dir), Arc::new(Recorder::default()));
    opt.config = cfg.plugins.clone();
    opt.place = cfg.assistant.place.clone();
    // Tests write into a folder of their own, never into your data.
    let tmp = std::env::temp_dir().join(format!("courier-plugin-tests-{}", std::process::id()));
    if test {
        let _ = std::fs::remove_dir_all(&tmp);
        opt.data_dir = Some(tmp.clone());
    }
    let host = ToolHost::load(opt)?;
    for e in &host.errors {
        println!("error: {e}");
    }
    if !test {
        for t in host.tools() {
            println!("{:<16} {}", t.name, t.description);
        }
        return Ok(());
    }
    let results = host.run_tests();
    let failed = results.iter().filter(|r| r.2.is_some()).count();
    for (plugin, name, err) in &results {
        match err {
            None => println!("ok   {plugin}: {name}"),
            Some(e) => println!("FAIL {plugin}: {name}: {e}"),
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    println!("{} tests, {} failed, {} plugin errors", results.len(), failed, host.errors.len());
    if failed > 0 || !host.errors.is_empty() {
        bail!("plugin tests failed");
    }
    Ok(())
}

fn say(cfg: &Config, text: &str, out: Option<&Path>) -> Result<()> {
    let eng = Engine::load(&cfg.root)?;
    let mut tts = engine::Tts::new(&eng, &cfg.path(&cfg.models.voice), &cfg.path(&cfg.models.cmudict), 2)?;
    let e = tts.explain(text)?;
    println!("reads: {}\nphonemes: {}", e.text, e.ipa);
    let pcm = tts.speak(text, cfg.voice.speaker, cfg.voice.speed, 1.0)?;
    match out {
        Some(p) => courier::wav::write_i16(p, &pcm, tts.rate())?,
        None => {
            let (tx, rx) = crossbeam_channel::unbounded();
            let sp = courier::audio::Speaker::open(tx)?;
            sp.out.speak(&pcm, tts.rate());
            let _ = rx.recv_timeout(std::time::Duration::from_secs(60));
        }
    }
    Ok(())
}

fn devices() -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let name = |d: &cpal::Device| d.description().map(|x| x.name().to_string()).unwrap_or_default();
    if let Some(d) = host.default_input_device() {
        println!("microphone: {} {:?}", name(&d), d.default_input_config().map(|c| (c.sample_rate(), c.channels())).ok());
    }
    if let Some(d) = host.default_output_device() {
        println!("output: {} {:?}", name(&d), d.default_output_config().map(|c| (c.sample_rate(), c.channels())).ok());
    }
    Ok(())
}
