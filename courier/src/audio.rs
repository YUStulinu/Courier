//! Sound in and out, through cpal.
//!
//! In: the default microphone, mixed to mono, sent on as it arrives; the main loop resamples it to
//! 16 kHz. Out: one stream that mixes the assistant's voice, chimes and music. Music is ducked while
//! the assistant listens or speaks. The device callbacks never block for long: they take a short lock,
//! copy samples and return.

use crate::dialog::Chime;
use crate::resample::Resampler;
use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub struct Mic {
    _stream: cpal::Stream,
    pub rate: u32,
    pub name: String,
}

/// Opens the default microphone; blocks of mono samples at `rate` arrive on the receiver.
pub fn open_mic() -> Result<(Mic, Receiver<Vec<f32>>)> {
    let host = cpal::default_host();
    let dev = host.default_input_device().ok_or_else(|| anyhow!("no microphone"))?;
    let name = dev.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "microphone".into());
    let cfg = dev.default_input_config().context("the microphone's configuration")?;
    let channels = cfg.channels() as usize;
    let rate = cfg.sample_rate();
    let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(256);
    let err = |e| eprintln!("microphone: {e}");
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream(
            cfg.config(),
            move |d: &[f32], _: &_| {
                let _ = tx.try_send(mono(d, channels, |s| s));
            },
            err,
            None,
        ),
        cpal::SampleFormat::I16 => dev.build_input_stream(
            cfg.config(),
            move |d: &[i16], _: &_| {
                let _ = tx.try_send(mono(d, channels, |s| s as f32 / 32768.0));
            },
            err,
            None,
        ),
        cpal::SampleFormat::I32 => dev.build_input_stream(
            cfg.config(),
            move |d: &[i32], _: &_| {
                let _ = tx.try_send(mono(d, channels, |s| s as f32 / 2147483648.0));
            },
            err,
            None,
        ),
        f => return Err(anyhow!("unsupported microphone sample format {f:?}")),
    }
    .context("opening the microphone")?;
    stream.play().context("starting the microphone")?;
    Ok((Mic { _stream: stream, rate, name }, rx))
}

fn mono<T: Copy>(d: &[T], channels: usize, f: impl Fn(T) -> f32) -> Vec<f32> {
    if channels == 1 {
        return d.iter().map(|&s| f(s)).collect();
    }
    d.chunks(channels).map(|c| c.iter().map(|&s| f(s)).sum::<f32>() / channels as f32).collect()
}

/// A piece of speech, already at the device's rate.
struct Segment {
    samples: Vec<f32>,
    pos: usize,
}

#[derive(Default)]
struct MusicState {
    playlist: Vec<PathBuf>,
    index: usize,
    paused: bool,
    playing: bool,
    /// Stereo frames at the device rate, filled by the decoder thread.
    buffer: VecDeque<(f32, f32)>,
    generation: u64,
    error: Option<String>,
}

struct Mixer {
    speech: VecDeque<Segment>,
    chime: Option<Segment>,
    duck: f32,
    duck_target: f32,
    volume: f32,
}

/// Something finished playing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Played {
    Speech,
}

/// The output device. The stream itself stays with the thread that opened it; everything else goes
/// through `Out`, which can be shared.
pub struct Speaker {
    _stream: cpal::Stream,
    pub name: String,
    pub out: Out,
}

#[derive(Clone)]
pub struct Out {
    pub rate: u32,
    mixer: Arc<Mutex<Mixer>>,
    music: Arc<Mutex<MusicState>>,
}

impl Speaker {
    pub fn open(done: Sender<Played>) -> Result<Speaker> {
        let host = cpal::default_host();
        let dev = host.default_output_device().ok_or_else(|| anyhow!("no audio output"))?;
        let name = dev.description().map(|d| d.name().to_string()).unwrap_or_else(|_| "speakers".into());
        let cfg = dev.default_output_config().context("the output's configuration")?;
        let channels = cfg.channels() as usize;
        let rate = cfg.sample_rate();
        let mixer = Arc::new(Mutex::new(Mixer { speech: VecDeque::new(), chime: None, duck: 1.0, duck_target: 1.0, volume: 1.0 }));
        let music = Arc::new(Mutex::new(MusicState::default()));
        let (m, mu) = (mixer.clone(), music.clone());
        let err = |e| eprintln!("audio output: {e}");
        let mut scratch: Vec<(f32, f32)> = Vec::new();
        let fill = move |out: &mut dyn FnMut(usize, f32, f32), frames: usize| {
            scratch.clear();
            scratch.resize(frames, (0.0, 0.0));
            {
                let mut mus = mu.lock().unwrap();
                if mus.playing && !mus.paused {
                    for s in scratch.iter_mut() {
                        match mus.buffer.pop_front() {
                            Some(v) => *s = v,
                            None => break,
                        }
                    }
                }
            }
            let mut mx = m.lock().unwrap();
            let mut finished = 0;
            for (i, s) in scratch.iter_mut().enumerate() {
                // Duck smoothly (about 50 ms).
                mx.duck += (mx.duck_target - mx.duck) * 0.0005;
                let mut v = 0.0;
                if let Some(seg) = mx.speech.front_mut() {
                    v += seg.samples[seg.pos];
                    seg.pos += 1;
                    if seg.pos >= seg.samples.len() {
                        mx.speech.pop_front();
                        finished += 1;
                    }
                }
                if let Some(c) = mx.chime.as_mut() {
                    v += c.samples[c.pos];
                    c.pos += 1;
                    if c.pos >= c.samples.len() {
                        mx.chime = None;
                    }
                }
                let g = mx.duck * mx.volume;
                out(i, (s.0 * g + v).clamp(-1.0, 1.0), (s.1 * g + v).clamp(-1.0, 1.0));
            }
            drop(mx);
            for _ in 0..finished {
                let _ = done.try_send(Played::Speech);
            }
        };
        let stream = match cfg.sample_format() {
            cpal::SampleFormat::F32 => {
                let mut fill = fill;
                dev.build_output_stream(
                    cfg.config(),
                    move |d: &mut [f32], _: &_| {
                        let frames = d.len() / channels;
                        fill(&mut |i, l, r| write_frame(&mut d[i * channels..(i + 1) * channels], l, r, |x| x), frames);
                    },
                    err,
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let mut fill = fill;
                dev.build_output_stream(
                    cfg.config(),
                    move |d: &mut [i16], _: &_| {
                        let frames = d.len() / channels;
                        fill(&mut |i, l, r| write_frame(&mut d[i * channels..(i + 1) * channels], l, r, |x| (x * 32767.0) as i16), frames);
                    },
                    err,
                    None,
                )
            }
            f => return Err(anyhow!("unsupported output sample format {f:?}")),
        }
        .context("opening the audio output")?;
        stream.play().context("starting the audio output")?;
        Ok(Speaker { _stream: stream, name, out: Out { rate, mixer, music } })
    }
}

impl Out {
    /// Queues speech (16-bit at `rate`).
    pub fn speak(&self, pcm: &[i16], rate: u32) {
        let x: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut r = Resampler::new(rate, self.rate);
        let mut samples = r.process(&x);
        samples.extend(r.process(&vec![0.0; 256]));
        if samples.is_empty() {
            samples.push(0.0);
        }
        self.mixer.lock().unwrap().speech.push_back(Segment { samples, pos: 0 });
    }

    /// Drops queued and playing speech. Returns how many pieces were dropped.
    pub fn stop_speech(&self) -> usize {
        let mut m = self.mixer.lock().unwrap();
        let n = m.speech.len();
        m.speech.clear();
        n
    }

    pub fn chime(&self, kind: Chime) {
        self.mixer.lock().unwrap().chime = Some(Segment { samples: chime(kind, self.rate), pos: 0 });
    }

    /// Music at `level` (0-1) of its volume: low while the assistant listens or talks.
    pub fn duck(&self, level: f32) {
        self.mixer.lock().unwrap().duck_target = level;
    }

    pub fn music(&self, cmd: crate::tools::MusicCmd) -> Result<Value> {
        use crate::tools::MusicCmd::*;
        // Every change of what plays bumps the generation; a decoder stops when it is not its own.
        let start = |s: &Out| {
            let music = s.music.clone();
            let rate = s.rate;
            std::thread::spawn(move || decode_loop(music, rate));
        };
        match cmd {
            Play(list) => {
                if list.is_empty() {
                    return Err(anyhow!("nothing to play"));
                }
                {
                    let mut m = self.music.lock().unwrap();
                    m.playlist = list;
                    m.index = 0;
                    m.paused = false;
                    m.playing = true;
                    m.buffer.clear();
                    m.generation += 1;
                    m.error = None;
                }
                start(self);
            }
            Pause => self.music.lock().unwrap().paused = true,
            Resume => self.music.lock().unwrap().paused = false,
            Stop => {
                let mut m = self.music.lock().unwrap();
                m.playing = false;
                m.buffer.clear();
                m.generation += 1;
            }
            Next | Previous => {
                {
                    let mut m = self.music.lock().unwrap();
                    if m.playlist.is_empty() {
                        return Err(anyhow!("nothing is playing"));
                    }
                    let n = m.playlist.len();
                    m.index = if cmd == Next { (m.index + 1) % n } else { (m.index + n - 1) % n };
                    m.buffer.clear();
                    m.generation += 1;
                    m.playing = true;
                    m.paused = false;
                }
                start(self);
            }
            Status => {}
        }
        let m = self.music.lock().unwrap();
        let track = m.playlist.get(m.index).map(|p| track_name(p));
        Ok(json!({
            "playing": m.playing && !m.paused,
            "paused": m.playing && m.paused,
            "track": track,
            "position": m.index + 1,
            "tracks": m.playlist.len(),
            "error": m.error,
        }))
    }
}

fn write_frame<T: Copy>(out: &mut [T], l: f32, r: f32, conv: impl Fn(f32) -> T) {
    match out.len() {
        0 => {}
        1 => out[0] = conv((l + r) * 0.5),
        _ => {
            out[0] = conv(l);
            out[1] = conv(r);
            for x in out[2..].iter_mut() {
                *x = conv(0.0);
            }
        }
    }
}

/// "Miles Davis - So What.mp3" -> "Miles Davis - So What".
pub fn track_name(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().replace('_', " ")).unwrap_or_default()
}

/// Short synthetic sounds: rising for "I'm listening", falling for "done", three beeps for alarms.
pub fn chime(kind: Chime, rate: u32) -> Vec<f32> {
    let notes: &[(f32, f32)] = match kind {
        Chime::Wake => &[(660.0, 0.07), (880.0, 0.09)],
        Chime::Done => &[(880.0, 0.07), (660.0, 0.09)],
        Chime::Alarm => &[(988.0, 0.15), (0.0, 0.1), (988.0, 0.15), (0.0, 0.1), (988.0, 0.15)],
    };
    let mut out = Vec::new();
    for &(f, d) in notes {
        let n = (d * rate as f32) as usize;
        for i in 0..n {
            let t = i as f32 / rate as f32;
            let env = (t / 0.005).min(1.0) * ((d - t) / 0.02).clamp(0.0, 1.0);
            out.push(if f > 0.0 { 0.25 * env * (2.0 * std::f32::consts::PI * f * t).sin() } else { 0.0 });
        }
    }
    out
}

fn decode_loop(music: Arc<Mutex<MusicState>>, rate: u32) {
    loop {
        let (path, generation) = {
            let m = music.lock().unwrap();
            if !m.playing || m.playlist.is_empty() {
                return;
            }
            (m.playlist[m.index].clone(), m.generation)
        };
        let r = decode_file(&path, rate, &music, generation);
        let mut m = music.lock().unwrap();
        if m.generation != generation {
            return;
        }
        if let Err(e) = r {
            m.error = Some(format!("{}: {e}", track_name(&path)));
        }
        // On to the next track; stop after the last.
        if m.index + 1 < m.playlist.len() {
            m.index += 1;
        } else {
            m.playing = false;
            return;
        }
    }
}

fn decode_file(path: &Path, rate: u32, music: &Arc<Mutex<MusicState>>, generation: u64) -> Result<()> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        hint.with_extension(&ext.to_string_lossy());
    }
    let mut format = symphonia::default::get_probe().probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())?;
    let track = format.default_track(TrackType::Audio).ok_or_else(|| anyhow!("no audio track"))?;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).ok_or_else(|| anyhow!("not audio"))?.clone();
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs().make_audio_decoder(&params, &AudioDecoderOptions::default())?;
    let mut rs: Option<(Resampler, Resampler, u32)> = None;
    let mut interleaved: Vec<f32> = Vec::new();
    while let Some(packet) = format.next_packet()? {
        if music.lock().unwrap().generation != generation {
            return Ok(());
        }
        if packet.track_id != track_id {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec = buf.spec().clone();
        let ch = spec.channels().count().max(1);
        let src_rate = spec.rate();
        interleaved.resize(buf.samples_interleaved(), 0.0);
        buf.copy_to_slice_interleaved(&mut interleaved);
        let (l, r): (Vec<f32>, Vec<f32>) = interleaved.chunks(ch).map(|c| (c[0], if ch > 1 { c[1] } else { c[0] })).unzip();
        if rs.as_ref().is_none_or(|x| x.2 != src_rate) {
            rs = Some((Resampler::new(src_rate, rate), Resampler::new(src_rate, rate), src_rate));
        }
        let (rl, rr, _) = rs.as_mut().unwrap();
        let (l, r) = (rl.process(&l), rr.process(&r));
        // Wait while two seconds are buffered (or while paused).
        loop {
            let full = {
                let m = music.lock().unwrap();
                if m.generation != generation {
                    return Ok(());
                }
                m.buffer.len() > rate as usize * 2
            };
            if !full {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let mut m = music.lock().unwrap();
        m.buffer.extend(l.into_iter().zip(r));
    }
    // Let the buffer play out before the next track starts.
    loop {
        let m = music.lock().unwrap();
        if m.generation != generation || m.buffer.is_empty() {
            return Ok(());
        }
        drop(m);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
