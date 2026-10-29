//! The running assistant: audio in, the dialog, the models on worker threads, audio out.
//!
//! ```text
//!  microphone ─► resample to 16 kHz ─► Silero (speech?) ─► dialog ◄─ transcripts ─ Whisper thread
//!                                  └─► wake word ────────►   │    ◄─ text pieces ─ model thread
//!                                                            │    ◄─ results ───── tools thread (Lua)
//!                                                            └──► speech pieces ─► voice thread ─► speaker
//! ```
//! One loop owns the dialog; everything slow happens on a worker that sends its result back as an
//! event. The same loop runs on a recording instead of the microphone (`courier sim`), in real time,
//! with a speaker that only keeps time: that is how the latency is measured.

use crate::audio::{Out, Played, Speaker};
use crate::config::Config;
use crate::dialog::{Chime, Dialog, DialogConfig, Effects, Note, State};
use crate::engine::{self, Engine, Generation};
use crate::prompt::{self, ToolCall, ToolDef};
use crate::resample::Resampler;
use crate::services::System;
use crate::tools::{Clock, HostOptions, Services, ToolHost};
use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender, select, unbounded};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

enum Ev {
    Transcript { job: u64, text: String, ms: f64 },
    Piece { turn: u64, text: String },
    Generated { turn: u64, g: Generation },
    ToolResult { turn: u64, value: Value },
    Alarm(String),
    ToolsChanged(Vec<ToolDef>, Vec<String>),
    Fatal(String),
}

/// One line of the run's timeline (ms since the audio started).
#[derive(Debug, Clone, Serialize)]
pub struct Mark {
    pub ms: f64,
    pub what: String,
    #[serde(skip_serializing_if = "Value::is_null")]
    pub data: Value,
}

/// Where speech goes: a device, or nowhere but a clock.
pub enum Output {
    Device(Out),
    /// Plays nothing; a piece "plays" for as long as it lasts, after the one before it.
    Virtual(Arc<Mutex<VirtualSpeaker>>),
}

#[derive(Default)]
pub struct VirtualSpeaker {
    busy_until: f64,
    ends: VecDeque<f64>,
}

impl Output {
    pub fn virtual_speaker() -> Output {
        Output::Virtual(Arc::new(Mutex::new(VirtualSpeaker::default())))
    }
}

struct Shared {
    /// Time zero: when the models were loaded and the audio started.
    start: Mutex<Instant>,
    marks: Mutex<Vec<Mark>>,
    /// Generations to stop (the model thread checks between tokens).
    cancelled: Mutex<HashSet<u64>>,
    /// Speech of turns up to this one is dropped before it is synthesized.
    silenced_upto: AtomicU64,
    /// The newest turn sent to the voice.
    last_spoken: AtomicU64,
    quiet: bool,
}

impl Shared {
    fn now(&self) -> f64 {
        self.start.lock().unwrap().elapsed().as_secs_f64() * 1000.0
    }

    fn mark(&self, what: &str, data: Value) {
        let ms = self.now();
        if !self.quiet {
            print_mark(ms, what, &data);
        }
        self.marks.lock().unwrap().push(Mark { ms, what: what.to_string(), data });
    }
}

fn print_mark(ms: f64, what: &str, data: &Value) {
    let text = |k: &str| data.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let line = match what {
        "ready" => format!("ready, say \"Hey Courier\" ({})", text("info")),
        "wake" => "(wake word)".to_string(),
        "listening" if data["follow_up"].as_bool() == Some(true) => "(listening, no wake word needed)".into(),
        "listening" => "(listening)".into(),
        "asleep" => "(asleep)".into(),
        "partial" => format!("  ... {}", text("text")),
        "heard" => format!("you: {}", text("text")),
        "tool" => format!("  -> {} {}", text("name"), data["arguments"]),
        "result" => format!("  <- {}", data["value"]),
        "say" => format!("courier: {}", text("text")),
        "interrupted" => "(interrupted)".into(),
        "alarm" => format!("alarm: {}", text("text")),
        "plugins" | "error" => format!("{what}: {}", text("text")),
        _ => return,
    };
    println!("[{:>7.2}] {line}", ms / 1000.0);
}

/// The last minute of 16 kHz audio, addressed by sample index since the start.
struct Ring {
    buf: VecDeque<f32>,
    first: usize,
}

impl Ring {
    const KEEP: usize = 16000 * 60;

    fn push(&mut self, x: &[f32]) {
        self.buf.extend(x);
        let over = self.buf.len().saturating_sub(Self::KEEP);
        if over > 0 {
            self.buf.drain(..over);
            self.first += over;
        }
    }

    fn range(&self, from: usize, to: usize) -> Vec<f32> {
        let a = from.max(self.first) - self.first;
        let z = (to.max(self.first) - self.first).min(self.buf.len());
        self.buf.range(a.min(z)..z).copied().collect()
    }
}

struct Workers {
    asr: Sender<(u64, Vec<f32>)>,
    llm: Sender<(u64, String, String)>,
    tts: Sender<(u64, String)>,
    tools: Sender<(u64, ToolCall)>,
}

/// The dialog's requests, carried out on the workers.
struct Fx<'a> {
    sh: &'a Shared,
    ring: &'a Ring,
    w: &'a Workers,
    out: &'a Output,
    schema: &'a str,
}

impl Effects for Fx<'_> {
    fn transcribe(&mut self, job: u64, from: u64, to: u64) {
        let _ = self.w.asr.send((job, self.ring.range(from as usize * 512, to as usize * 512)));
    }

    fn generate(&mut self, turn: u64, prompt: String) {
        self.sh.cancelled.lock().unwrap().remove(&turn);
        let _ = self.w.llm.send((turn, prompt, self.schema.to_string()));
    }

    fn cancel_generation(&mut self, turn: u64) {
        self.sh.cancelled.lock().unwrap().insert(turn);
    }

    fn call_tool(&mut self, turn: u64, call: ToolCall) {
        let _ = self.w.tools.send((turn, call));
    }

    fn speak(&mut self, turn: u64, text: String) {
        self.sh.last_spoken.fetch_max(turn, Ordering::SeqCst);
        let _ = self.w.tts.send((turn, text));
    }

    fn stop_speaking(&mut self) {
        // Pieces of the turns so far that still wait for the voice are dropped too.
        self.sh.silenced_upto.store(self.sh.last_spoken.load(Ordering::SeqCst), Ordering::SeqCst);
        match self.out {
            Output::Device(o) => {
                o.stop_speech();
            }
            Output::Virtual(v) => {
                let mut v = v.lock().unwrap();
                v.busy_until = self.sh.now();
                v.ends.clear();
            }
        }
    }

    fn chime(&mut self, kind: Chime) {
        if let Output::Device(o) = self.out {
            o.chime(kind);
        }
    }

    fn note(&mut self, n: Note) {
        let (what, data) = match n {
            Note::Woke => ("wake", Value::Null),
            Note::Asleep => ("asleep", Value::Null),
            Note::Listening { follow_up } => ("listening", json!({"follow_up": follow_up})),
            Note::Heard(t) => ("heard", json!({"text": t})),
            Note::Partial(t) => ("partial", json!({"text": t})),
            Note::EndOfSpeech { speech_end_ms, confirmed_ms } => {
                ("end_of_speech", json!({"speech_end_ms": speech_end_ms, "confirmed_ms": confirmed_ms}))
            }
            Note::Speculating(t) => ("speculating", json!({"text": t})),
            Note::SpeculationKept => ("speculation_kept", Value::Null),
            Note::SpeculationDropped => ("speculation_dropped", Value::Null),
            Note::Calling(c) => ("tool", json!({"name": c.name, "arguments": c.arguments})),
            Note::Grounded(n) => ("grounded", json!({"changes": n})),
            Note::Result(v) => ("result", json!({"value": v})),
            Note::Said(t) => ("said", json!({"text": t})),
            Note::Interrupted => ("interrupted", Value::Null),
        };
        self.sh.mark(what, data);
    }
}

pub struct Options {
    pub quiet: bool,
    /// For `sim`: return once the input has ended and the assistant is idle.
    pub stop_when_idle: bool,
    /// Told when the models are loaded; the clock starts then (`sim` starts its audio on it).
    pub ready: Option<Sender<()>>,
    /// Wake at these moments (ms of audio) instead of when the detector fires: measures the rest of
    /// the assistant apart from the wake word. The detector still runs; what it finds is marked.
    pub wake_at: Vec<f64>,
}

/// Each worker says so when its model is loaded.
struct Loaded;

fn spawn_asr(cfg: &Config, eng: &Engine, tx: Sender<Ev>, loaded: Sender<Loaded>) -> Sender<(u64, Vec<f32>)> {
    let (jobs, rx) = unbounded::<(u64, Vec<f32>)>();
    let (e, path, gpu, fast) = (eng.clone(), cfg.path(&cfg.models.whisper), cfg.models.whisper_gpu, cfg.models.whisper_fast);
    std::thread::spawn(move || {
        let mut asr = match engine::Asr::new(&e, &path, 0, gpu) {
            Ok(a) => a,
            Err(err) => return drop(tx.send(Ev::Fatal(err.to_string()))),
        };
        let _ = loaded.send(Loaded);
        while let Ok(first) = rx.recv() {
            // Only the newest request matters when several wait (the utterance only grows).
            let (job, samples) = rx.try_iter().last().unwrap_or(first);
            let text = match asr.transcribe(&samples, fast) {
                Ok(t) => (t.text, t.ms),
                Err(err) => {
                    eprintln!("recognition: {err}");
                    (String::new(), 0.0)
                }
            };
            let _ = tx.send(Ev::Transcript { job, text: text.0, ms: text.1 });
        }
    });
    jobs
}

fn spawn_llm(cfg: &Config, eng: &Engine, tx: Sender<Ev>, sh: Arc<Shared>, loaded: Sender<Loaded>, warm: String) -> Sender<(u64, String, String)> {
    let (jobs, rx) = unbounded::<(u64, String, String)>();
    let (e, path, ctx, layers) = (eng.clone(), cfg.path(&cfg.models.llm), cfg.models.context, cfg.models.gpu_layers);
    std::thread::spawn(move || {
        let mut llm = match engine::Llm::new(&e, &path, ctx, layers) {
            Ok(l) => l,
            Err(err) => return drop(tx.send(Ev::Fatal(err.to_string()))),
        };
        // The system prompt and the tools (about 2,000 tokens) are computed now, while the other
        // models load, so that the first request only computes what is new.
        let _ = llm.generate(&warm, None, None, 1, &mut |_| true);
        let _ = loaded.send(Loaded);
        for (turn, prompt, schema) in rx {
            if sh.cancelled.lock().unwrap().contains(&turn) {
                let _ = tx.send(Ev::Generated { turn, g: Generation { stop: "cancel".into(), ..Default::default() } });
                continue;
            }
            let mut cb = |p: &str| {
                if sh.cancelled.lock().unwrap().contains(&turn) {
                    return false;
                }
                let _ = tx.send(Ev::Piece { turn, text: p.to_string() });
                true
            };
            let g = llm.generate(&prompt, Some(prompt::TRIGGER), (!schema.is_empty()).then_some(schema.as_str()), 300, &mut cb);
            let g = g.unwrap_or_else(|err| {
                eprintln!("language model: {err}");
                Generation { stop: "error".into(), ..Default::default() }
            });
            let _ = tx.send(Ev::Generated { turn, g });
        }
    });
    jobs
}

fn spawn_tts(cfg: &Config, eng: &Engine, out: &Output, tx: Sender<Ev>, sh: Arc<Shared>, loaded: Sender<Loaded>) -> Sender<(u64, String)> {
    let (jobs, rx) = unbounded::<(u64, String)>();
    let (e, voice, dict, speaker, speed) =
        (eng.clone(), cfg.path(&cfg.models.voice), cfg.path(&cfg.models.cmudict), cfg.voice.speaker, cfg.voice.speed);
    let out = match out {
        Output::Device(o) => Output::Device(o.clone()),
        Output::Virtual(v) => Output::Virtual(v.clone()),
    };
    std::thread::spawn(move || {
        let mut tts = match engine::Tts::new(&e, &voice, &dict, 2) {
            Ok(t) => t,
            Err(err) => return drop(tx.send(Ev::Fatal(err.to_string()))),
        };
        let _ = loaded.send(Loaded);
        let mut started: HashSet<u64> = HashSet::new();
        for (turn, text) in rx {
            if turn <= sh.silenced_upto.load(Ordering::SeqCst) {
                continue;
            }
            let pcm = tts.speak(&text, speaker, speed, 1.0).unwrap_or_else(|err| {
                eprintln!("voice: {err}");
                vec![0; 160]
            });
            let secs = pcm.len() as f64 / tts.rate() as f64;
            if started.insert(turn) {
                sh.mark("first_audio", json!({"turn": turn}));
            }
            sh.mark("say", json!({"text": text, "seconds": secs}));
            match &out {
                Output::Device(o) => o.speak(&pcm, tts.rate()),
                Output::Virtual(v) => {
                    let now = sh.now();
                    let mut v = v.lock().unwrap();
                    let start = v.busy_until.max(now);
                    v.busy_until = start + secs * 1000.0;
                    let end = v.busy_until;
                    v.ends.push_back(end);
                }
            }
        }
    });
    jobs
}

/// The tools thread's queue, the tools it loaded, and the plugins that failed to load.
type Tools = (Sender<(u64, ToolCall)>, Vec<ToolDef>, Vec<String>);

fn spawn_tools(cfg: &Config, services: Arc<dyn Services>, tx: Sender<Ev>) -> Result<Tools> {
    let opts = {
        let (dir, data_dir, config) = (cfg.path(&cfg.assistant.plugins_dir), cfg.data_dir(), cfg.plugins.clone());
        let place = cfg.assistant.place.clone();
        move || HostOptions {
            dir: dir.clone(),
            data_dir: Some(data_dir.clone()),
            config: config.clone(),
            services: services.clone(),
            clock: Clock::Real,
            place: place.clone(),
            time_limit: crate::tools::TOOL_TIME,
        }
    };
    // Loaded once here for the tool list, then again on the thread that owns it (Lua stays there).
    let probe = ToolHost::load(opts()).context("loading the plugins")?;
    let (tools, errors) = (probe.tools(), probe.errors.clone());
    drop(probe);
    let (jobs, rx) = unbounded::<(u64, ToolCall)>();
    std::thread::spawn(move || {
        let mut host = match ToolHost::load(opts()) {
            Ok(h) => h,
            Err(err) => return drop(tx.send(Ev::Fatal(err.to_string()))),
        };
        let mut checked = Instant::now();
        loop {
            match rx.recv_timeout(Duration::from_millis(200)) {
                Ok((turn, call)) => {
                    let value = host.call(&call.name, &call.arguments);
                    let _ = tx.send(Ev::ToolResult { turn, value });
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(_) => return,
            }
            for a in host.due_alarms() {
                let _ = tx.send(Ev::Alarm(a.text));
            }
            // New or changed plugins: reload them all (alarms are kept by the host, on disk).
            if checked.elapsed() > Duration::from_secs(1) {
                checked = Instant::now();
                if host.changed() {
                    match ToolHost::load(opts()) {
                        Ok(h) => {
                            host = h;
                            let _ = tx.send(Ev::ToolsChanged(host.tools(), host.errors.clone()));
                        }
                        Err(err) => drop(tx.send(Ev::ToolsChanged(host.tools(), vec![err.to_string()]))),
                    }
                }
            }
        }
    });
    Ok((jobs, tools, errors))
}

/// Runs the assistant on `audio` (blocks of samples at `rate`) until the input ends (with
/// `stop_when_idle`) or the program is stopped. Returns the timeline.
pub fn run(
    cfg: &Config,
    eng: &Engine,
    audio: Receiver<Vec<f32>>,
    rate: u32,
    output: Output,
    played: Receiver<Played>,
    opt: Options,
) -> Result<Vec<Mark>> {
    let sh = Arc::new(Shared {
        start: Mutex::new(Instant::now()),
        marks: Mutex::new(Vec::new()),
        cancelled: Mutex::new(HashSet::new()),
        silenced_upto: AtomicU64::new(0),
        last_spoken: AtomicU64::new(0),
        quiet: opt.quiet,
    });
    let (ev_tx, ev_rx) = unbounded::<Ev>();
    let services: Arc<dyn Services> = Arc::new(System {
        out: match &output {
            Output::Device(o) => Some(o.clone()),
            Output::Virtual(_) => None,
        },
        network: cfg.assistant.network,
    });
    let (tools_tx, tools, errors) = spawn_tools(cfg, services, ev_tx.clone())?;
    for e in errors {
        sh.mark("error", json!({"text": e}));
    }
    let (loaded_tx, loaded_rx) = unbounded::<Loaded>();
    let w = Workers {
        asr: spawn_asr(cfg, eng, ev_tx.clone(), loaded_tx.clone()),
        llm: spawn_llm(cfg, eng, ev_tx.clone(), sh.clone(), loaded_tx.clone(), prompt::system_message(prompt::SYSTEM, &tools)),
        tts: spawn_tts(cfg, eng, &output, ev_tx.clone(), sh.clone(), loaded_tx),
        tools: tools_tx,
    };

    // The ears, on this thread: speech detection and the wake word.
    let mut vad = engine::Vad::new(eng, Some(&cfg.path(&cfg.models.vad)))?;
    let threshold = cfg.wake_threshold();
    let mut wake = engine::Wake::new(eng, &cfg.path(&cfg.models.wake), threshold, cfg.wake.hits, cfg.wake.refractory_ms)?;
    let place = cfg.assistant.place.clone();
    let dcfg = DialogConfig {
        endpoint: cfg.endpoint.clone(),
        speculate: cfg.assistant.speculate,
        live_transcript_ms: if cfg.assistant.live_transcript { 1000.0 } else { 0.0 },
        follow_up_ms: cfg.assistant.follow_up_s * 1000.0,
        listen_ms: cfg.assistant.listen_s * 1000.0,
        place: place.clone(),
        ..Default::default()
    };
    let mut schema = prompt::call_schema(&tools).to_string();
    let n_tools = tools.len();
    let mut dialog = Dialog::new(dcfg, tools, Box::new(move || prompt::clock(chrono::Local::now(), &place)));
    let mut ring = Ring { buf: VecDeque::new(), first: 0 };
    let mut resampler = Resampler::new(rate, 16000);
    let mut recent = VecDeque::from(vec![0.0f32; 4]);
    let mut input_done = false;
    let mut idle_since: Option<Instant> = None;
    let mut last_state = dialog.state();
    let mut audio_ms = 0.0f64;
    // Wait for the three models, then start the clock and drop audio that arrived meanwhile.
    let mut loaded = 0;
    while loaded < 3 {
        select! {
            recv(loaded_rx) -> _ => loaded += 1,
            recv(ev_rx) -> ev => if let Ok(Ev::Fatal(e)) = ev { return Err(anyhow!(e)) },
        }
    }
    *sh.start.lock().unwrap() = Instant::now();
    while audio.try_recv().is_ok() {}
    if let Some(r) = &opt.ready {
        let _ = r.send(());
    }
    sh.mark("ready", json!({"info": format!("wake threshold {threshold:.2}, {n_tools} tools, {}", if eng.has_gpu() { "GPU" } else { "CPU" })}));

    let tick = crossbeam_channel::tick(Duration::from_millis(20));
    loop {
        select! {
            recv(audio) -> block => match block {
                Ok(block) => {
                    let x = resampler.process(&block);
                    ring.push(&x);
                    let probs = vad.process(&x)?;
                    let woke = wake.process(&x)?;
                    let before_ms = audio_ms;
                    audio_ms += x.len() as f64 / 16.0;
                    let mut fx = Fx { sh: &sh, ring: &ring, w: &w, out: &output, schema: &schema };
                    if let Some(&t) = opt.wake_at.iter().find(|&&t| t > before_ms && t <= audio_ms) {
                        sh.mark("wake_given", json!({"ms_audio": t}));
                        dialog.wake((t / 32.0).ceil() as u64, true, &mut fx);
                    }
                    if woke.detections > 0 && !opt.wake_at.is_empty() {
                        sh.mark("wake_detected", json!({"ms_audio": wake.last_ms(), "ignored": true}));
                    } else if woke.detections > 0 {
                        // Speech right after the phrase means the request follows without a pause.
                        let speaking = recent.iter().any(|&p| p > 0.5);
                        sh.mark("wake_detected", json!({"ms_audio": wake.last_ms()}));
                        dialog.wake((wake.last_ms() / 32.0).ceil() as u64, speaking, &mut fx);
                    }
                    for p in probs {
                        recent.pop_front();
                        recent.push_back(p);
                        dialog.vad(p, &mut fx);
                    }
                }
                Err(_) => {
                    input_done = true;
                }
            },
            recv(ev_rx) -> ev => {
                let Ok(ev) = ev else { break };
                let mut fx = Fx { sh: &sh, ring: &ring, w: &w, out: &output, schema: &schema };
                match ev {
                    Ev::Transcript { job, text, ms } => {
                        sh.mark("transcript", json!({"job": job, "text": text, "ms": ms}));
                        dialog.transcript(job, &text, &mut fx);
                    }
                    Ev::Piece { turn, text } => {
                        sh.mark("piece", json!({"turn": turn, "text": text}));
                        dialog.piece(turn, &text, &mut fx);
                    }
                    Ev::Generated { turn, g } => {
                        sh.mark("generated", json!({
                            "turn": turn, "stop": g.stop, "text": g.text, "prompt_tokens": g.prompt_tokens,
                            "reused_tokens": g.reused_tokens, "generated": g.generated, "ms_prompt": g.ms_prompt,
                            "ms_first": g.ms_first, "ms": g.ms
                        }));
                        dialog.generated(turn, &g.text, &g.stop, &mut fx);
                    }
                    Ev::ToolResult { turn, value } => dialog.tool_result(turn, value, &mut fx),
                    Ev::Alarm(text) => {
                        sh.mark("alarm", json!({"text": text}));
                        dialog.alarm(text, &mut fx);
                    }
                    Ev::ToolsChanged(tools, errors) => {
                        for e in errors {
                            sh.mark("error", json!({"text": e}));
                        }
                        sh.mark("plugins", json!({"text": format!("reloaded, {} tools", tools.len())}));
                        schema = prompt::call_schema(&tools).to_string();
                        dialog.set_tools(tools);
                    }
                    Ev::Fatal(e) => return Err(anyhow!(e)),
                }
            },
            recv(played) -> p => {
                if p.is_ok() {
                    let mut fx = Fx { sh: &sh, ring: &ring, w: &w, out: &output, schema: &schema };
                    dialog.spoken(&mut fx);
                }
            },
            recv(tick) -> _ => {
                let mut fx = Fx { sh: &sh, ring: &ring, w: &w, out: &output, schema: &schema };
                match &output {
                    Output::Virtual(v) => {
                        // Pieces whose time is up have been "played".
                        let now = sh.now();
                        let mut done = 0;
                        {
                            let mut v = v.lock().unwrap();
                            while v.ends.front().is_some_and(|&t| t <= now) {
                                v.ends.pop_front();
                                done += 1;
                            }
                        }
                        for _ in 0..done {
                            dialog.spoken(&mut fx);
                        }
                    }
                    Output::Device(o) => o.duck(if matches!(dialog.state(), State::Asleep) { 1.0 } else { 0.15 }),
                }
                let st = dialog.state();
                if st != last_state {
                    // While it talks, its own voice must not wake it: a stricter threshold.
                    wake.set_threshold(if st == State::Speaking { threshold.max(0.97) } else { threshold });
                    last_state = st;
                }
                if opt.stop_when_idle && input_done {
                    let idle = matches!(st, State::Asleep | State::Listening { heard: false, .. });
                    match idle_since {
                        _ if !idle => idle_since = None,
                        None => idle_since = Some(Instant::now()),
                        Some(t) if t.elapsed() > Duration::from_millis(300) => break,
                        _ => {}
                    }
                }
            },
        }
    }
    let marks = sh.marks.lock().unwrap().clone();
    Ok(marks)
}

/// Opens the microphone and the speakers and runs until the program is stopped.
pub fn live(cfg: &Config, eng: &Engine) -> Result<()> {
    let (mic, rx) = crate::audio::open_mic()?;
    let (done_tx, done_rx) = unbounded();
    let speaker = Speaker::open(done_tx)?;
    println!("microphone: {} ({} Hz); output: {} ({} Hz)", mic.name, mic.rate, speaker.name, speaker.out.rate);
    run(
        cfg,
        eng,
        rx,
        mic.rate,
        Output::Device(speaker.out.clone()),
        done_rx,
        Options { quiet: false, stop_when_idle: false, ready: None, wake_at: Vec::new() },
    )?;
    Ok(())
}

/// Feeds a recording in real time (as a microphone would) with the virtual speaker; returns the
/// timeline. `tail_s` of silence follow the recording, for the answer to finish.
pub fn simulate(cfg: &Config, eng: &Engine, samples: Vec<f32>, rate: u32, tail_s: f64, quiet: bool, wake_at: Vec<f64>) -> Result<Vec<Mark>> {
    let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(64);
    let (go_tx, go_rx) = crossbeam_channel::bounded::<()>(1);
    let block = (rate / 50) as usize; // 20 ms, like a device
    let mut all = samples;
    all.extend(std::iter::repeat_n(0.0, (tail_s * rate as f64) as usize));
    std::thread::spawn(move || {
        if go_rx.recv().is_err() {
            return;
        }
        let start = Instant::now();
        for (i, c) in all.chunks(block).enumerate() {
            let due = start + Duration::from_secs_f64((i * block) as f64 / rate as f64);
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
            if tx.send(c.to_vec()).is_err() {
                return;
            }
        }
    });
    let (_never_tx, never_rx) = unbounded::<Played>();
    run(cfg, eng, rx, rate, Output::virtual_speaker(), never_rx, Options { quiet, stop_when_idle: true, ready: Some(go_tx), wake_at })
}
