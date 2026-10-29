//! Deciding when you have finished speaking.
//!
//! Silero gives a speech probability every 32 ms. Speech starts after two speech frames in a row. When
//! it goes quiet, the pause is reported at once (so the utterance so far can be transcribed while we
//! wait), and how long to wait before calling it the end depends on what was said:
//!   - it reads like a finished sentence ("Set a timer for ten minutes."): a short wait;
//!   - it stops on a word that needs more ("Remind me to call ..."): a long one;
//!   - no transcript yet, or no punctuation: in between.
//!
//! A fixed silence timeout has to choose between cutting people off mid-thought and answering late;
//! the transcript lets the wait be short only where that is safe.

use serde::{Deserialize, Serialize};

pub const FRAME_MS: f64 = 32.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EndpointConfig {
    /// A frame is speech at or above this probability...
    pub speech: f32,
    /// ...and silence below this one; in between, it continues what it was.
    pub silence: f32,
    /// Speech frames in a row that start an utterance.
    pub start_frames: u32,
    /// Silence that counts as a pause (the utterance is transcribed then).
    pub pause_ms: f64,
    /// Silence that ends a finished-sounding utterance.
    pub end_complete_ms: f64,
    /// ... one that stops on a word that needs more.
    pub end_incomplete_ms: f64,
    /// ... one we know nothing about yet.
    pub end_unsure_ms: f64,
    /// Longest utterance.
    pub max_ms: f64,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        EndpointConfig {
            speech: 0.5,
            silence: 0.35,
            start_frames: 2,
            pause_ms: 160.0,
            end_complete_ms: 384.0,
            end_incomplete_ms: 1280.0,
            end_unsure_ms: 1000.0,
            max_ms: 15000.0,
        }
    }
}

impl EndpointConfig {
    /// A plain silence timeout, for comparison: every utterance waits `ms`.
    pub fn fixed(ms: f64) -> EndpointConfig {
        EndpointConfig { end_complete_ms: ms, end_incomplete_ms: ms, end_unsure_ms: ms, ..Default::default() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completeness {
    Complete,
    Incomplete,
    Unsure,
}

/// Words a sentence does not end on. Mostly function words: articles, prepositions, conjunctions,
/// possessives, and the verbs of commands that need an object.
const CONTINUES: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "so", "because", "to", "for", "of", "with", "at", "in", "on", "by", "from", "into", "about", "my", "your",
    "his", "her", "their", "our", "its", "this", "that", "these", "those", "which", "who", "if", "when", "then", "than", "um", "uh", "er", "erm",
    "hmm", "is", "are", "was", "were", "be", "set", "remind", "play", "open", "find", "search", "call", "add", "turn", "put", "tell", "what's",
    "whats", "hear", "see", "watch", "show", "start", "check", "buy", "get", "make", "send", "listen", "too", "where's", "how's", "me", "some",
    "any", "every", "until", "after", "before", "called", "named", "also", "maybe", "please", "can", "could", "would", "will", "should", "i", "i'm",
    "let's", "let", "just", "very", "really", "more",
];

/// Whether a transcript reads like a finished request.
pub fn completeness(text: &str) -> Completeness {
    let t = text.trim();
    if t.is_empty() {
        return Completeness::Unsure;
    }
    // Whisper's "..." means it heard the voice trail off.
    if t.ends_with("...") || t.ends_with('…') || t.ends_with(',') || t.ends_with('-') {
        return Completeness::Incomplete;
    }
    let last = t.split_whitespace().last().unwrap_or("").trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_lowercase();
    if CONTINUES.contains(&last.as_str()) {
        return Completeness::Incomplete;
    }
    // A question mark is Whisper hearing a question's intonation end ("What's the weather like?").
    if t.ends_with('?') {
        return Completeness::Complete;
    }
    if t.ends_with('.') || t.ends_with('?') || t.ends_with('!') { Completeness::Complete } else { Completeness::Unsure }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// Speech started at this frame (the first speech frame).
    Start(u64),
    /// Silence began at this frame: transcribe up to here.
    Pause(u64),
    /// Speech again after a pause: forget what the pause started.
    Resume,
    /// The utterance is over; it ended (last speech) at this frame.
    End(u64),
}

#[derive(Debug)]
pub struct Endpointer {
    cfg: EndpointConfig,
    frame: u64,
    in_speech: bool,
    speech_run: u32,
    silence_run: u32,
    start: u64,
    last_speech: u64,
    paused: bool,
    completeness: Completeness,
}

impl Endpointer {
    pub fn new(cfg: EndpointConfig) -> Endpointer {
        Endpointer {
            cfg,
            frame: 0,
            in_speech: false,
            speech_run: 0,
            silence_run: 0,
            start: 0,
            last_speech: 0,
            paused: false,
            completeness: Completeness::Unsure,
        }
    }

    pub fn config(&self) -> &EndpointConfig {
        &self.cfg
    }

    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// The index of the next frame.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Restarts at frame `at`, e.g. after the wake word. `speaking`: the person is already talking
    /// (the request follows the wake word without a pause), so the utterance starts here.
    pub fn restart(&mut self, at: u64, speaking: bool) {
        self.frame = at;
        self.in_speech = speaking;
        self.speech_run = 0;
        self.silence_run = 0;
        self.start = at;
        self.last_speech = at;
        self.paused = false;
        self.completeness = Completeness::Unsure;
    }

    /// What the transcript of the current pause says about it.
    pub fn set_transcript(&mut self, text: &str) {
        if self.paused {
            self.completeness = completeness(text);
        }
    }

    fn wait_ms(&self) -> f64 {
        match self.completeness {
            Completeness::Complete => self.cfg.end_complete_ms,
            Completeness::Incomplete => self.cfg.end_incomplete_ms,
            Completeness::Unsure => self.cfg.end_unsure_ms,
        }
    }

    /// Feeds one frame's speech probability.
    pub fn feed(&mut self, p: f32) -> Option<Event> {
        let f = self.frame;
        self.frame += 1;
        let speech = p >= self.cfg.speech;
        let silence = p < self.cfg.silence;
        if !self.in_speech {
            self.speech_run = if speech { self.speech_run + 1 } else { 0 };
            if self.speech_run >= self.cfg.start_frames {
                self.in_speech = true;
                self.silence_run = 0;
                self.start = f + 1 - self.speech_run as u64;
                self.last_speech = f;
                self.paused = false;
                return Some(Event::Start(self.start));
            }
            return None;
        }
        if !silence {
            self.silence_run = 0;
            self.last_speech = f;
            if self.paused && speech {
                self.paused = false;
                self.completeness = Completeness::Unsure;
                return Some(Event::Resume);
            }
            if (f - self.start) as f64 * FRAME_MS >= self.cfg.max_ms {
                return Some(self.end());
            }
            return None;
        }
        self.silence_run += 1;
        let quiet_ms = self.silence_run as f64 * FRAME_MS;
        if !self.paused && quiet_ms >= self.cfg.pause_ms {
            self.paused = true;
            return Some(Event::Pause(self.last_speech + 1));
        }
        if self.paused && quiet_ms >= self.wait_ms() {
            return Some(self.end());
        }
        None
    }

    fn end(&mut self) -> Event {
        self.in_speech = false;
        self.speech_run = 0;
        self.silence_run = 0;
        self.paused = false;
        self.completeness = Completeness::Unsure;
        Event::End(self.last_speech + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(ep: &mut Endpointer, probs: &[f32]) -> Vec<(u64, Event)> {
        let mut out = Vec::new();
        for &p in probs {
            let f = ep.frame();
            if let Some(e) = ep.feed(p) {
                out.push((f, e));
            }
        }
        out
    }

    fn frames(ms: f64) -> usize {
        (ms / FRAME_MS).round() as usize
    }

    #[test]
    fn completeness_reads_the_last_word_and_punctuation() {
        assert_eq!(completeness("Set a timer for ten minutes."), Completeness::Complete);
        assert_eq!(completeness("What's the weather?"), Completeness::Complete);
        assert_eq!(completeness("Remind me to call"), Completeness::Incomplete);
        assert_eq!(completeness("Play some music by the"), Completeness::Incomplete);
        assert_eq!(completeness("Remind me to, um..."), Completeness::Incomplete);
        assert_eq!(completeness("Open the"), Completeness::Incomplete);
        assert_eq!(completeness("Remind me to call."), Completeness::Incomplete);
        assert_eq!(completeness("lights off"), Completeness::Unsure);
        // Whisper hears a question in "What's the weather like in ..." and "remind me to" as "too".
        assert_eq!(completeness("What's the weather like in?"), Completeness::Incomplete);
        assert_eq!(completeness("Can you remind me too?"), Completeness::Incomplete);
        assert_eq!(completeness("I'd like to hear."), Completeness::Incomplete);
        assert_eq!(completeness(""), Completeness::Unsure);
    }

    #[test]
    fn start_pause_end() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        let mut p = vec![0.0; 10];
        p.extend(vec![0.9; 30]);
        p.extend(vec![0.0; 60]);
        let ev = run(&mut ep, &p);
        assert_eq!(ev[0], (11, Event::Start(10)));
        assert_eq!(ev[1].1, Event::Pause(40));
        // No transcript: the "unsure" wait (1000 ms, 32 frames after the last speech frame).
        assert_eq!(ev[2], (39 + (1000.0 / FRAME_MS).ceil() as u64, Event::End(40)));
    }

    #[test]
    fn a_finished_sentence_ends_sooner_and_an_unfinished_one_later() {
        for (text, wait) in [("Set a timer for ten minutes.", 384.0), ("Remind me to call", 1280.0)] {
            let mut ep = Endpointer::new(EndpointConfig::default());
            let mut p = vec![0.9; 20];
            p.extend(vec![0.0; 80]);
            let mut end = None;
            for (i, &x) in p.iter().enumerate() {
                if let Some(e) = ep.feed(x) {
                    match e {
                        Event::Pause(_) => ep.set_transcript(text),
                        Event::End(_) => end = Some(i),
                        _ => {}
                    }
                }
            }
            assert_eq!(end, Some(19 + frames(wait)), "{text}");
        }
    }

    #[test]
    fn speech_after_a_pause_resumes_the_same_utterance() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        let mut p = vec![0.9; 20];
        p.extend(vec![0.0; 10]); // 320 ms: a pause, not an end
        p.extend(vec![0.9; 20]);
        p.extend(vec![0.0; 40]);
        let ev: Vec<Event> = run(&mut ep, &p).into_iter().map(|x| x.1).collect();
        assert!(matches!(ev[..], [Event::Start(0), Event::Pause(20), Event::Resume, Event::Pause(50), Event::End(50)]), "{ev:?}");
    }

    #[test]
    fn in_between_probabilities_keep_the_state() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        let mut p = vec![0.9; 5];
        p.extend(vec![0.4; 100]); // neither speech nor silence: still speaking
        let ev = run(&mut ep, &p);
        assert_eq!(ev.len(), 1);
    }

    #[test]
    fn restart_while_speaking_starts_the_utterance_at_once() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        ep.restart(100, true);
        let ev = run(&mut ep, &[0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(ev[0].1, Event::Pause(102));
    }

    #[test]
    fn long_speech_is_cut() {
        let mut ep = Endpointer::new(EndpointConfig { max_ms: 1000.0, ..Default::default() });
        let ev = run(&mut ep, &vec![0.9; 60]);
        assert!(ev.iter().any(|e| matches!(e.1, Event::End(_))));
    }
}
