//! The evaluations behind the README's results.
//!
//!   courier eval tools       requests in text -> the tool call the model makes (or none)
//!   courier eval endpoint    spoken requests with pauses -> where the end of speech was called
//!   courier eval latency     "Hey Courier, ..." spoken by voices the wake word never heard, through
//!                            the whole assistant in real time -> what it heard, did and when
//!
//! The tool and latency sets use a fixed date (Thursday 29 October 2026, 14:00, Bucharest) so that
//! "tomorrow" always means the same day.

use crate::assistant::{self, Mark};
use crate::chat::Agent;
use crate::config::Config;
use crate::dialog::clean_transcript;
use crate::endpoint::{EndpointConfig, Endpointer, Event as Ep, FRAME_MS};
use crate::engine::{self, Engine};
use crate::prompt::{self, Message};
use crate::resample;
use crate::tools::{HostOptions, Recorder, ToolHost};
use anyhow::{Context, Result};
use clap::Subcommand;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub const CLOCK: &str = "Now: Thursday 2026-10-29 14:00 (24-hour time). The user is in Bucharest, Romania.";
const TODAY: (i32, u32, u32) = (2026, 10, 29);

#[derive(Subcommand)]
pub enum What {
    /// Tool calling: which tool the model calls for each request, with which arguments.
    Tools {
        #[arg(long, default_value = "bench/tools.jsonl")]
        set: PathBuf,
        /// Another GGUF model than the configured one.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Generate calls without the JSON grammar.
        #[arg(long)]
        no_grammar: bool,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Endpointing: how soon the end of speech is called, and how often too soon.
    Endpoint {
        #[arg(long, default_value = "bench/endpoint.jsonl")]
        set: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        /// The voice the requests are spoken with (a multi-speaker Piper voice).
        #[arg(long, default_value = "third_party/models/en_GB-vctk-medium.onnx")]
        voice: PathBuf,
    },
    /// The whole assistant on spoken requests, in real time.
    Latency {
        #[arg(long, default_value = "bench/latency.jsonl")]
        set: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value = "third_party/models/en_GB-vctk-medium.onnx")]
        voice: PathBuf,
        /// Do not start the model before the end of speech is confirmed.
        #[arg(long)]
        no_speculation: bool,
        /// Keep the generated audio.
        #[arg(long)]
        wav: Option<PathBuf>,
        /// Wake when the detector fires, rather than at the end of each wake phrase (the detector's
        /// own results are in `courier_wake.evaluate`; this measures everything after it).
        #[arg(long)]
        real_wake: bool,
    },
}

pub fn run(cfg: &Config, what: What) -> Result<()> {
    match what {
        What::Tools { set, model, no_grammar, out } => tools(cfg, &cfg.path(&set), model.as_deref(), !no_grammar, out.as_deref()),
        What::Endpoint { set, out, voice } => endpoint(cfg, &cfg.path(&set), &cfg.path(&voice), out.as_deref()),
        What::Latency { set, out, voice, no_speculation, wav, real_wake } => {
            latency(cfg, &cfg.path(&set), &cfg.path(&voice), !no_speculation, out.as_deref(), wav.as_deref(), real_wake)
        }
    }
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).with_context(|| l.to_string())).collect()
}

fn write_out(out: Option<&Path>, v: &Value) -> Result<()> {
    if let Some(p) = out {
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(p, serde_json::to_string_pretty(v)?)?;
        println!("wrote {}", p.display());
    }
    Ok(())
}

fn percentile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((s.len() - 1) as f64 * q).round() as usize;
    s[i]
}

// --- matching tool calls ------------------------------------------------------------------------

fn words(s: &str) -> Vec<String> {
    const STOP: &[&str] = &["the", "a", "an", "my", "to", "of", "for", "about", "with", "and", "by", "on", "at", "in", "from", "some"];
    s.split(|c: char| !c.is_alphanumeric())
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty() && !STOP.contains(&w.as_str()))
        .map(|w| if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") { w[..w.len() - 1].to_string() } else { w })
        .collect()
}

/// "tomorrow", "friday", "2026-10-30" -> "2026-10-30" (relative to the evaluation's fixed date).
pub fn resolve_date(s: &str) -> Option<String> {
    use chrono::{Datelike, Duration, NaiveDate};
    let today = NaiveDate::from_ymd_opt(TODAY.0, TODAY.1, TODAY.2)?;
    let s = s.trim().to_lowercase();
    let s = s.strip_prefix("next ").or_else(|| s.strip_prefix("this ")).or_else(|| s.strip_prefix("on ")).unwrap_or(&s).to_string();
    match s.as_str() {
        "today" | "tonight" | "now" => return Some(today.to_string()),
        "tomorrow" => return Some((today + Duration::days(1)).to_string()),
        _ => {}
    }
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
        return Some(d.to_string());
    }
    let days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
    let i = days.iter().position(|d| s == *d || s == d[..3])?;
    let ahead = (i as i64 - today.weekday().num_days_from_monday() as i64).rem_euclid(7);
    Some((today + Duration::days(ahead)).to_string())
}

/// "6 pm", "18:00", "6:00 PM" -> "18:00".
pub fn resolve_time(s: &str) -> Option<String> {
    let t = s.trim().to_lowercase().replace([' ', '.'], "");
    let (body, pm, am) = if let Some(b) = t.strip_suffix("pm") {
        (b.to_string(), true, false)
    } else if let Some(b) = t.strip_suffix("am") {
        (b.to_string(), false, true)
    } else {
        (t.clone(), false, false)
    };
    let (h, m) = match body.split_once(':') {
        Some((h, m)) => (h.parse::<u32>().ok()?, m.parse::<u32>().ok()?),
        None => (body.parse::<u32>().ok()?, 0),
    };
    let h = if pm && h < 12 {
        h + 12
    } else if am && h == 12 {
        0
    } else {
        h
    };
    (h < 24 && m < 60).then(|| format!("{h:02}:{m:02}"))
}

fn value_matches(key: &str, want: &Value, got: &Value) -> bool {
    match want {
        Value::Number(n) => got.as_f64().is_some_and(|g| (g - n.as_f64().unwrap()).abs() < 1e-6),
        Value::Bool(b) => got.as_bool() == Some(*b),
        Value::String(w) => {
            let Some(g) = got.as_str().map(str::to_string).or_else(|| got.as_f64().map(|x| x.to_string())) else { return false };
            if key == "date" {
                return resolve_date(w) == resolve_date(&g) && resolve_date(w).is_some();
            }
            if key == "time" {
                return resolve_time(w) == resolve_time(&g) && resolve_time(w).is_some();
            }
            let gw = words(&g);
            words(w).iter().all(|x| gw.iter().any(|y| y == x || (x.len() >= 4 && y.starts_with(x.as_str()))))
        }
        _ => false,
    }
}

/// Whether one argument fits its expectation (see bench/README.md for the forms).
fn arg_ok(key: &str, want: &Value, args: &Value) -> bool {
    let got = args.get(key);
    let Some(spec) = want.as_object() else {
        return got.is_some_and(|g| value_matches(key, want, g));
    };
    if spec.get("absent").is_some() {
        return got.is_none();
    }
    if let Some(v) = spec.get("absent_or") {
        return got.is_none_or(|g| value_matches(key, v, g));
    }
    if spec.get("generic").is_some() {
        const GENERIC: &[&str] = &["music", "something", "anything", "song", "songs", "random", "any"];
        return got.is_none_or(|g| words(g.as_str().unwrap_or("")).iter().all(|w| GENERIC.contains(&w.as_str())));
    }
    if let Some(d) = spec.get("date") {
        return match got {
            None => spec.contains_key("optional"),
            Some(g) => value_matches("date", d, g),
        };
    }
    if spec.contains_key("min") || spec.contains_key("max") {
        let Some(x) = got.and_then(|g| g.as_f64()) else { return false };
        return spec.get("min").and_then(|m| m.as_f64()).is_none_or(|m| x >= m) && spec.get("max").and_then(|m| m.as_f64()).is_none_or(|m| x <= m);
    }
    if let Some(v) = spec.get("optional") {
        return got.is_none_or(|g| v.is_boolean() && v.as_bool() == Some(true) || value_matches(key, v, g));
    }
    false
}

/// Whether a call (or no call) is one of the expected ones. Returns (right tool, right arguments).
/// `_result` checks numbers in what the tool returned (the seconds a timer runs, whatever words the
/// model used for them).
fn judge(expect: &[Value], call: Option<&prompt::ToolCall>, result: Option<&Value>) -> (bool, bool) {
    let wants_none = expect.is_empty() || expect.iter().any(|e| e["tool"].is_null());
    let Some(call) = call else { return (wants_none, wants_none) };
    let mut tool_ok = false;
    for e in expect {
        if e["tool"].as_str() != Some(call.name.as_str()) {
            continue;
        }
        tool_ok = true;
        let empty = json!({});
        let args = e.get("args").unwrap_or(&empty).as_object().cloned().unwrap_or_default();
        let ok = args.iter().all(|(k, v)| {
            if k == "_result" {
                let want = v.as_object().cloned().unwrap_or_default();
                want.iter().all(|(rk, rv)| {
                    let got = result.and_then(|r| r.get(rk)).and_then(|x| x.as_f64());
                    got.is_some_and(|g| (g - rv.as_f64().unwrap_or(f64::NAN)).abs() <= 5.0)
                })
            } else {
                arg_ok(k, v, &call.arguments)
            }
        });
        if ok {
            return (true, true);
        }
    }
    (tool_ok, false)
}

#[derive(Deserialize)]
struct ToolCase {
    id: String,
    text: String,
    expect: Vec<Value>,
    #[serde(default)]
    history: Vec<(String, String)>,
}

fn tools(cfg: &Config, set: &Path, model: Option<&Path>, grammar: bool, out: Option<&Path>) -> Result<()> {
    let cases: Vec<ToolCase> = read_jsonl(set)?;
    let eng = Engine::load(&cfg.root)?;
    let model = model.map(|m| cfg.path(m)).unwrap_or_else(|| cfg.path(&cfg.models.llm));
    let llm = engine::Llm::new(&eng, &model, cfg.models.context, cfg.models.gpu_layers)?;
    // The plugins run, but against a recorder: nothing on this computer is touched.
    let tmp = std::env::temp_dir().join(format!("courier-eval-{}", std::process::id()));
    let mut opt = HostOptions::new(cfg.path(&cfg.assistant.plugins_dir), Arc::new(Recorder::default()));
    opt.data_dir = Some(tmp.clone());
    opt.place = "Bucharest, Romania".into();
    let host = ToolHost::load(opt)?;
    let mut agent = Agent::new(llm, host);
    agent.grammar = grammar;
    agent.clock = Some(CLOCK.to_string());
    agent.today = Some(chrono::NaiveDate::from_ymd_opt(TODAY.0, TODAY.1, TODAY.2).unwrap());
    let mut rows = Vec::new();
    let (mut tool_ok, mut all_ok, mut raw_ok, mut none_cases, mut none_ok, mut call_cases, mut call_ok) = (0, 0, 0, 0, 0, 0, 0);
    let (mut grounded, mut recovered) = (0, 0);
    let mut ms_call = Vec::new();
    let mut ms_first_word = Vec::new();
    let mut invalid = 0;
    for c in &cases {
        agent.history = c.history.iter().map(|(r, t)| if r == "user" { Message::User(t.clone()) } else { Message::Assistant(t.clone()) }).collect();
        let t0 = Instant::now();
        let mut first_word: Option<f64> = None;
        // The whole turn, as the assistant runs it: grounding, and another round after a tool's error.
        let log = agent.respond(&c.text, &mut |piece| {
            if first_word.is_none() && !piece.trim().is_empty() {
                first_word = Some(t0.elapsed().as_secs_f64() * 1000.0);
            }
        })?;
        let first = log.steps.first();
        let raw = first.and_then(|s| s.raw_call.clone());
        // What finally ran: the last call of the turn.
        let last_step = log.steps.iter().rev().find(|s| s.call.is_some());
        let last = last_step.and_then(|s| s.call.clone());
        let last_result = last_step.and_then(|s| s.result.clone());
        if log.steps.iter().any(|s| s.call.is_none() && s.raw_call.is_none() && s.text.is_empty()) {
            invalid += 1;
        }
        // The model alone: its first call as written (results only exist for calls that ran).
        let first_result = first.filter(|s| s.raw_call == s.call).and_then(|s| s.result.clone());
        let (_, r_ok) = judge(&c.expect, raw.as_ref(), first_result.as_ref());
        let (t_ok, a_ok) = judge(&c.expect, last.as_ref(), last_result.as_ref());
        grounded += (first.is_some_and(|s| !s.grounded.is_empty())) as usize;
        recovered += (log.steps.iter().filter(|s| s.call.is_some()).count() > 1 && a_ok) as usize;
        let wants_none = c.expect.is_empty();
        if wants_none {
            none_cases += 1;
            none_ok += a_ok as usize;
        } else {
            call_cases += 1;
            call_ok += a_ok as usize;
        }
        tool_ok += t_ok as usize;
        all_ok += a_ok as usize;
        raw_ok += r_ok as usize;
        match first {
            Some(s) if s.raw_call.is_some() => ms_call.push(s.ms),
            _ => {
                if let Some(f) = first_word {
                    ms_first_word.push(f);
                }
            }
        }
        let mark = if a_ok {
            "ok  "
        } else if t_ok {
            "ARGS"
        } else {
            "FAIL"
        };
        let got = last
            .as_ref()
            .map(|c| format!("{} {}", c.name, c.arguments))
            .unwrap_or_else(|| format!("(no tool) {}", log.reply.chars().take(60).collect::<String>()));
        let fixed = if r_ok != a_ok { if a_ok { " (fixed by grounding/retry)" } else { " (broken by grounding!)" } } else { "" };
        println!("{mark} {} {:<52} {got}{fixed}", c.id, c.text);
        rows.push(json!({"id": c.id, "text": c.text, "tool_ok": t_ok, "ok": a_ok, "raw_ok": r_ok, "raw_call": raw, "call": last, "steps": log.steps, "reply": log.reply}));
    }
    let _ = std::fs::remove_dir_all(&tmp);
    let n = cases.len() as f64;
    let summary = json!({
        "model": model.file_name().map(|f| f.to_string_lossy().into_owned()),
        "grammar": grammar,
        "cases": cases.len(),
        "fully_correct": all_ok as f64 / n,
        "model_alone_correct": raw_ok as f64 / n,
        "right_tool_or_none": tool_ok as f64 / n,
        "calls_correct": call_ok as f64 / call_cases.max(1) as f64,
        "no_tool_correct": none_ok as f64 / none_cases.max(1) as f64,
        "grounding_changed": grounded,
        "fixed_after_tool_error": recovered,
        "invalid_calls": invalid,
        "ms_call_median": percentile(&ms_call, 0.5),
        "ms_call_p90": percentile(&ms_call, 0.9),
        "ms_first_word_median": percentile(&ms_first_word, 0.5),
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    write_out(out, &json!({"summary": summary, "cases": rows}))
}

// --- endpointing ----------------------------------------------------------------------------------

#[derive(Deserialize)]
struct EndpointCase {
    id: String,
    /// What is said, with "|" where the speaker pauses.
    text: String,
    #[serde(default)]
    pause_ms: f64,
    #[serde(default)]
    speaker: u32,
}

fn speak16(tts: &mut engine::Tts, text: &str, speaker: u32) -> Result<Vec<f32>> {
    let pcm = tts.speak(text, speaker, 1.0, 0.8)?;
    let x: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
    Ok(trim(&resample::convert(&x, tts.rate(), 16000)))
}

/// Without leading and trailing silence (40 dB under the peak), as a person's speech would start and stop.
fn trim(x: &[f32]) -> Vec<f32> {
    let frame = 160;
    let e: Vec<f32> = x.chunks(frame).map(|c| c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).collect();
    let peak = e.iter().cloned().fold(0.0f32, f32::max);
    let loud = |v: &f32| *v > peak * 1e-4;
    let (Some(a), Some(b)) = (e.iter().position(loud), e.iter().rposition(loud)) else { return Vec::new() };
    x[a * frame..((b + 1) * frame).min(x.len())].to_vec()
}

fn endpoint(cfg: &Config, set: &Path, voice: &Path, out: Option<&Path>) -> Result<()> {
    let cases: Vec<EndpointCase> = read_jsonl(set)?;
    let eng = Engine::load(&cfg.root)?;
    let mut tts = engine::Tts::new(&eng, voice, &cfg.path(&cfg.models.cmudict), 4)?;
    let mut asr = engine::Asr::new(&eng, &cfg.path(&cfg.models.whisper), 0, cfg.models.whisper_gpu)?;
    let mut vad = engine::Vad::new(&eng, Some(&cfg.path(&cfg.models.vad)))?;
    // Each case: audio, where its speech really ends, and where its pauses are.
    struct Prepared {
        id: String,
        audio: Vec<f32>,
        end: f64,
        pause_ends: Vec<f64>,
        probs: Vec<f32>,
    }
    let mut prepared = Vec::new();
    for c in &cases {
        let mut audio = vec![0.0f32; 8000];
        let mut pause_ends = Vec::new();
        let parts: Vec<&str> = c.text.split('|').map(str::trim).collect();
        for (i, p) in parts.iter().enumerate() {
            if i > 0 {
                audio.extend(std::iter::repeat_n(0.0, (c.pause_ms * 16.0) as usize));
                pause_ends.push(audio.len() as f64 / 16.0);
            }
            audio.extend(speak16(&mut tts, p, c.speaker)?);
        }
        let end = audio.len() as f64 / 16.0;
        audio.extend(std::iter::repeat_n(0.0, 16000 * 3));
        vad.reset();
        let probs = vad.process(&audio)?;
        prepared.push(Prepared { id: c.id.clone(), audio, end, pause_ends, probs });
    }
    let configs: Vec<(&str, EndpointConfig)> = vec![
        ("fixed 300 ms", EndpointConfig::fixed(300.0)),
        ("fixed 500 ms", EndpointConfig::fixed(500.0)),
        ("fixed 800 ms", EndpointConfig::fixed(800.0)),
        ("fixed 1200 ms", EndpointConfig::fixed(1200.0)),
        ("transcript-aware (Courier)", cfg.endpoint.clone()),
    ];
    // Whisper's transcript of each pause, cached: the configurations share pauses.
    let mut cache: std::collections::HashMap<(usize, u64), (String, f64)> = std::collections::HashMap::new();
    let mut results = Vec::new();
    for (name, ec) in &configs {
        let mut cut_early = 0;
        let mut delays = Vec::new();
        let mut paused_cases = 0;
        let mut rows = Vec::new();
        for (ci, p) in prepared.iter().enumerate() {
            let mut ep = Endpointer::new(ec.clone());
            // The transcript arrives as long after the pause as Whisper takes.
            let mut pending: Option<(u64, String)> = None;
            let mut end_ms = None;
            let mut heard = Vec::new();
            for (f, &prob) in p.probs.iter().enumerate() {
                if let Some((at, text)) = pending.as_ref()
                    && f as u64 >= *at
                {
                    ep.set_transcript(text);
                    pending = None;
                }
                match ep.feed(prob) {
                    Some(Ep::Pause(pf)) => {
                        let key = (ci, pf);
                        let (text, ms) = match cache.get(&key) {
                            Some(v) => v.clone(),
                            None => {
                                let t = asr.transcribe(&p.audio[..(pf as usize * 512).min(p.audio.len())], cfg.models.whisper_fast)?;
                                let v = (clean_transcript(&t.text), t.ms);
                                cache.insert(key, v.clone());
                                v
                            }
                        };
                        heard.push(json!({"pause_ms": pf as f64 * FRAME_MS, "text": text, "asr_ms": ms}));
                        pending = Some((f as u64 + (ms / FRAME_MS).ceil() as u64, text));
                    }
                    Some(Ep::End(_)) => {
                        end_ms = Some((f + 1) as f64 * FRAME_MS);
                        break;
                    }
                    _ => {}
                }
            }
            let end_ms = end_ms.unwrap_or(f64::INFINITY);
            let early = end_ms < p.end - 50.0;
            if !p.pause_ends.is_empty() {
                paused_cases += 1;
            }
            if early {
                cut_early += 1;
            } else {
                delays.push(end_ms - p.end);
            }
            rows.push(json!({"id": p.id, "end_ms": end_ms, "speech_end_ms": p.end, "cut_early": early, "pauses": p.pause_ends, "heard": heard}));
        }
        let r = json!({
            "config": name,
            "cases": prepared.len(),
            "with_pauses": paused_cases,
            "cut_early": cut_early,
            "cut_early_rate": cut_early as f64 / prepared.len() as f64,
            "delay_ms_median": percentile(&delays, 0.5),
            "delay_ms_p90": percentile(&delays, 0.9),
            "rows": rows,
        });
        println!(
            "{:<28} cut early {:>2}/{}  delay after speech median {:>5.0} ms, p90 {:>5.0} ms",
            name,
            cut_early,
            prepared.len(),
            r["delay_ms_median"].as_f64().unwrap_or(f64::NAN),
            r["delay_ms_p90"].as_f64().unwrap_or(f64::NAN)
        );
        results.push(r);
    }
    write_out(out, &json!({"results": results}))
}

// --- latency, end to end --------------------------------------------------------------------------

#[derive(Deserialize)]
struct LatencyCase {
    id: String,
    text: String,
    #[serde(default)]
    tool: Option<String>,
    #[serde(default)]
    speaker: u32,
}

const WAKE_IPA: &[&str] = &["[[hˈeɪ]] [[kˈʊɹiɚ]],", "[[hˈeɪ]] [[kˈɜːɹiɚ]],", "[[hˈeɪ]], [[kˈʊɹiɚ]],"];

fn latency(cfg: &Config, set: &Path, voice: &Path, speculate: bool, out: Option<&Path>, wav: Option<&Path>, real_wake: bool) -> Result<()> {
    let cases: Vec<LatencyCase> = read_jsonl(set)?;
    let eng = Engine::load(&cfg.root)?;
    // One long recording: each request 12 s after the last, so every one needs the wake word.
    let mut tts = engine::Tts::new(&eng, voice, &cfg.path(&cfg.models.cmudict), 4)?;
    // 2.5 s of silence first: the wake word detector scores once it has 2 s of audio.
    let mut audio = vec![0.0f32; 40000];
    let mut spans = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let start = audio.len() as f64 / 16.0;
        audio.extend(speak16(&mut tts, WAKE_IPA[i % WAKE_IPA.len()], c.speaker)?);
        let wake_end = audio.len() as f64 / 16.0;
        audio.extend(std::iter::repeat_n(0.0, 16000 * 15 / 100));
        audio.extend(speak16(&mut tts, &c.text, c.speaker)?);
        let end = audio.len() as f64 / 16.0;
        spans.push((start, wake_end, end));
        audio.extend(std::iter::repeat_n(0.0, 16000 * 12));
    }
    drop(tts);
    if let Some(w) = wav {
        crate::wav::write_f32(w, &audio, 16000)?;
    }
    let mut c2 = cfg.clone();
    c2.assistant.speculate = speculate;
    c2.assistant.live_transcript = false;
    // A fixed history-free start and no network: the weather plugin will report it cannot.
    c2.assistant.network = false;
    let tmp = std::env::temp_dir().join(format!("courier-latency-{}", std::process::id()));
    c2.assistant.data_dir = tmp.clone();
    println!("{} requests, {:.0} s of audio, in real time", cases.len(), audio.len() as f64 / 16000.0);
    let wake_at = if real_wake { Vec::new() } else { spans.iter().map(|s| s.1).collect() };
    let marks = assistant::simulate(&c2, &eng, audio, 16000, 2.0, true, wake_at)?;
    let _ = std::fs::remove_dir_all(&tmp);
    let rows = score_latency(&cases, &spans, &marks);
    let get = |k: &str| rows.iter().filter_map(|r| r[k].as_f64()).collect::<Vec<f64>>();
    let n = cases.len() as f64;
    let woke = rows.iter().filter(|r| r["woke"].as_bool() == Some(true)).count();
    let tool_ok = rows.iter().filter(|r| r["tool_ok"].as_bool() == Some(true)).count();
    let first = get("first_audio_after_speech_ms");
    let summary = json!({
        "speculation": speculate,
        "wake": if real_wake { "detector" } else { "given at the end of each wake phrase" },
        "requests": cases.len(),
        "woke": woke as f64 / n,
        "tool_ok": tool_ok as f64 / n,
        "wake_delay_ms_median": percentile(&get("wake_after_phrase_ms"), 0.5),
        "endpoint_ms_median": percentile(&get("endpoint_after_speech_ms"), 0.5),
        "first_audio_ms_median": percentile(&first, 0.5),
        "first_audio_ms_p90": percentile(&first, 0.9),
        "under_1s": first.iter().filter(|&&x| x < 1000.0).count() as f64 / first.len().max(1) as f64,
        "speculation_kept": marks.iter().filter(|m| m.what == "speculation_kept").count(),
        "speculation_dropped": marks.iter().filter(|m| m.what == "speculation_dropped").count(),
    });
    for r in &rows {
        println!(
            "{} woke {:<5} heard {:<48} tool {:<14} first audio {:>5.0} ms after speech",
            r["id"].as_str().unwrap_or(""),
            r["woke"],
            r["heard"].as_str().unwrap_or("-").chars().take(48).collect::<String>(),
            r["tool"].as_str().unwrap_or("-"),
            r["first_audio_after_speech_ms"].as_f64().unwrap_or(f64::NAN)
        );
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    write_out(out, &json!({"summary": summary, "requests": rows, "timeline": marks}))
}

/// Splits the timeline by request and measures each one.
fn score_latency(cases: &[LatencyCase], spans: &[(f64, f64, f64)], marks: &[Mark]) -> Vec<Value> {
    let mut rows = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let (start, wake_end, end) = spans[i];
        let next = spans.get(i + 1).map(|s| s.0).unwrap_or(f64::INFINITY);
        let m: Vec<&Mark> = marks.iter().filter(|m| m.ms >= start && m.ms < next).collect();
        let at = |what: &str| m.iter().find(|x| x.what == what).map(|x| x.ms);
        let woke = m.iter().find(|x| x.what == "wake_detected").and_then(|x| x.data["ms_audio"].as_f64());
        let heard = m.iter().find(|x| x.what == "heard").and_then(|x| x.data["text"].as_str()).map(str::to_string);
        let tool = m.iter().find(|x| x.what == "tool").and_then(|x| x.data["name"].as_str()).map(str::to_string);
        let confirmed = m.iter().find(|x| x.what == "end_of_speech").and_then(|x| x.data["confirmed_ms"].as_f64());
        let first_audio = at("first_audio");
        let tool_ok = tool == c.tool;
        rows.push(json!({
            "id": c.id, "text": c.text, "expected_tool": c.tool, "tool": tool, "tool_ok": tool_ok,
            "woke": woke.is_some(), "wake_after_phrase_ms": woke.map(|w| w - wake_end),
            "heard": heard,
            "endpoint_after_speech_ms": confirmed.map(|c| c - end),
            "first_audio_after_speech_ms": first_audio.map(|f| f - end),
            "said": m.iter().filter(|x| x.what == "say").filter_map(|x| x.data["text"].as_str()).collect::<Vec<_>>().join(" "),
        }));
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_and_times_resolve_against_the_fixed_day() {
        assert_eq!(resolve_date("tomorrow").unwrap(), "2026-10-30");
        assert_eq!(resolve_date("Friday").unwrap(), "2026-10-30");
        assert_eq!(resolve_date("thursday").unwrap(), "2026-10-29");
        assert_eq!(resolve_date("next monday").unwrap(), "2026-11-02");
        assert_eq!(resolve_time("6 pm").unwrap(), "18:00");
        assert_eq!(resolve_time("6:30 PM").unwrap(), "18:30");
        assert_eq!(resolve_time("09:15").unwrap(), "09:15");
        assert_eq!(resolve_time("12am").unwrap(), "00:00");
    }

    #[test]
    fn judging_calls() {
        let call = |name: &str, args: Value| prompt::ToolCall { name: name.into(), arguments: args };
        let timer = vec![json!({"tool": "set_timer", "args": {"_result": {"seconds": 600}}})];
        let ten = call("set_timer", json!({"duration": "ten minutes"}));
        assert_eq!(judge(&timer, Some(&ten), Some(&json!({"seconds": 600}))), (true, true));
        assert_eq!(judge(&timer, Some(&ten), Some(&json!({"seconds": 300}))), (true, false));
        assert_eq!(judge(&timer, Some(&ten), None), (true, false));
        assert_eq!(judge(&timer, None, None), (false, false));
        assert_eq!(judge(&[], None, None), (true, true));
        assert_eq!(judge(&[], Some(&call("flip_coin", json!({}))), None), (false, false));
        let rem = vec![json!({"tool": "add_reminder", "args": {"text": "call mother", "time": "18:00", "date": "2026-10-30"}})];
        assert!(judge(&rem, Some(&call("add_reminder", json!({"text": "Call my mother", "time": "6 pm", "date": "tomorrow"}))), None).1);
        assert!(!judge(&rem, Some(&call("add_reminder", json!({"text": "call father", "time": "18:00", "date": "tomorrow"}))), None).1);
        let w = vec![json!({"tool": "get_weather", "args": {"place": {"absent_or": "bucharest"}, "date": {"date": "2026-10-29", "optional": true}}})];
        assert!(judge(&w, Some(&call("get_weather", json!({}))), None).1);
        assert!(judge(&w, Some(&call("get_weather", json!({"place": "Bucharest", "date": "today"}))), None).1);
        assert!(!judge(&w, Some(&call("get_weather", json!({"place": "Paris"}))), None).1);
        let m = vec![json!({"tool": "play_music", "args": {"query": {"generic": true}}})];
        assert!(judge(&m, Some(&call("play_music", json!({"query": "some music"}))), None).1);
        assert!(!judge(&m, Some(&call("play_music", json!({"query": "jazz"}))), None).1);
    }
}
