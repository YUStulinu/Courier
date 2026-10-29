//! The conversation's state machine.
//!
//! ```text
//!  Asleep ──wake word──► Listening ──end of speech──► Thinking ──first words──► Speaking
//!    ▲                      ▲  │                         │  ▲                     │
//!    │       timeout        │  └─ nothing said ─► Asleep │  └── tool result ◄─────┤
//!    └──────────────────────┴──── follow-up window ◄─────┴────── done speaking ◄──┘
//! ```
//!
//! Getting the first word out quickly is most of the work:
//!   - when speech pauses (160 ms), the utterance is transcribed at once, before we know it ended;
//!   - with that transcript the model starts answering right away, speculatively: what it says is
//!     held back and its tool calls are not run until the end of speech is confirmed. If you go on
//!     talking, that answer is cancelled;
//!   - the answer is spoken a clause at a time as it streams out of the model;
//!   - a tool whose result carries `say` is answered with that text, without a second generation.
//!
//! The wake word interrupts anything (barge-in). The dialog does no work itself: it asks for work
//! through `Effects` and is told the results, which keeps it testable without models or audio.

use crate::endpoint::{EndpointConfig, Endpointer, Event as Ep, FRAME_MS};
use crate::ground;
use crate::prompt::{self, Message, ToolCall, ToolDef};
use crate::speech::Chunker;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chime {
    Wake,
    Done,
    Alarm,
}

/// What the dialog asks the rest of the program to do.
pub trait Effects {
    /// Transcribe the audio between two frames (32 ms each, counted from the start of the stream).
    fn transcribe(&mut self, job: u64, from_frame: u64, to_frame: u64);
    fn generate(&mut self, turn: u64, prompt: String);
    fn cancel_generation(&mut self, turn: u64);
    fn call_tool(&mut self, turn: u64, call: ToolCall);
    fn speak(&mut self, turn: u64, text: String);
    fn stop_speaking(&mut self);
    fn chime(&mut self, kind: Chime);
    fn note(&mut self, what: Note);
}

/// Things worth showing or timing.
#[derive(Debug, Clone, PartialEq)]
pub enum Note {
    Woke,
    Asleep,
    Listening {
        follow_up: bool,
    },
    Heard(String),
    Partial(String),
    /// The end of speech: the frame speech ended at, and the time it was confirmed.
    EndOfSpeech {
        speech_end_ms: f64,
        confirmed_ms: f64,
    },
    /// The model started on a transcript before the end of speech was confirmed.
    Speculating(String),
    SpeculationKept,
    SpeculationDropped,
    Calling(ToolCall),
    /// Arguments changed to match what was said (see ground.rs).
    Grounded(Vec<String>),
    Result(Value),
    Said(String),
    Interrupted,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum State {
    Asleep,
    Listening { follow_up: bool, since_ms: f64, heard: bool },
    Thinking,
    Speaking,
}

#[derive(Debug, Clone)]
pub struct DialogConfig {
    pub endpoint: EndpointConfig,
    /// Start the model at a pause, before the end of speech is confirmed.
    pub speculate: bool,
    /// Transcribe while you speak (for the display), every this many ms of speech; 0 = never.
    pub live_transcript_ms: f64,
    /// How long it listens after the wake word for speech to begin.
    pub listen_ms: f64,
    /// How long it keeps listening after answering, without the wake word.
    pub follow_up_ms: f64,
    /// Audio kept before the start of speech.
    pub preroll_ms: f64,
    /// Tool calls in a row before the model must answer in words.
    pub max_rounds: u32,
    /// Conversation messages kept (older ones are dropped).
    pub max_history: usize,
    /// Minutes of silence after which the conversation starts afresh.
    pub forget_after_ms: f64,
    pub system: String,
    pub place: String,
    /// The day dates are counted from; None: today.
    pub today: Option<chrono::NaiveDate>,
}

impl Default for DialogConfig {
    fn default() -> Self {
        DialogConfig {
            endpoint: EndpointConfig::default(),
            speculate: true,
            live_transcript_ms: 1000.0,
            listen_ms: 8000.0,
            follow_up_ms: 5000.0,
            preroll_ms: 320.0,
            max_rounds: 3,
            max_history: 16,
            forget_after_ms: 10.0 * 60_000.0,
            system: prompt::SYSTEM.to_string(),
            place: String::new(),
            today: None,
        }
    }
}

#[derive(Debug)]
struct Turn {
    id: u64,
    transcript: String,
    /// The end of speech is confirmed: words may be spoken and tools run.
    confirmed: bool,
    chunker: Chunker,
    held: Vec<String>,
    held_call: Option<ToolCall>,
    /// Words the model wrote before the end was confirmed, for the conversation history.
    held_answer: Option<String>,
    generating: bool,
    calling: bool,
    rounds: u32,
    spoken: Vec<String>,
    finished: bool,
}

impl Turn {
    fn new(id: u64, transcript: String, confirmed: bool) -> Turn {
        Turn {
            id,
            transcript,
            confirmed,
            chunker: Chunker::new(),
            held: Vec::new(),
            held_call: None,
            held_answer: None,
            generating: true,
            calling: false,
            rounds: 0,
            spoken: Vec::new(),
            finished: false,
        }
    }
}

pub struct Dialog {
    cfg: DialogConfig,
    state: State,
    ep: Endpointer,
    tools: Vec<ToolDef>,
    history: Vec<Message>,
    last_activity_ms: f64,
    next_id: u64,
    turn: Option<Turn>,
    /// The utterance's first frame.
    utt_start: u64,
    /// The frame the wake word ended at: audio before it is not part of the request.
    wake_frame: u64,
    pause_job: Option<u64>,
    pause_text: Option<String>,
    /// The end was confirmed but the pause's transcript is not back yet.
    awaiting_text: bool,
    end_frame: u64,
    live_job: Option<u64>,
    last_live_frame: u64,
    speaking_queued: usize,
    alarms: Vec<String>,
    now_ms: f64,
    clock: Box<dyn Fn() -> String + Send>,
}

fn frame_ms(f: u64) -> f64 {
    f as f64 * FRAME_MS
}

/// Whisper's transcript without the wake phrase it may have caught at the start, and without its
/// non-speech markers.
pub fn clean_transcript(text: &str) -> String {
    let mut t = text.trim().to_string();
    for marker in ["[BLANK_AUDIO]", "[blank_audio]", "(silence)", "[Silence]", "[MUSIC]", "(music)", "[NOISE]"] {
        t = t.replace(marker, "");
    }
    let re = regex::Regex::new(
        r"(?i)^\s*(?:(?:hey|hi|a|ok|okay)[\s,.!]*)?(?:courier|currier|courir|kurier|career|korea|carrier|curie|courtier)\b[\s,.!?]*",
    )
    .unwrap();
    let t = re.replace(&t, "").to_string();
    let t = t.trim().to_string();
    // A lone "you" or "Thank you." is what Whisper writes for silence or breathing.
    let bare = t.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    if bare.is_empty() || bare == "you" || bare == "thank you" || bare == "bye" {
        return String::new();
    }
    // Capitalize the first letter, as Whisper would have.
    let mut c = t.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

impl Dialog {
    pub fn new(cfg: DialogConfig, tools: Vec<ToolDef>, clock: Box<dyn Fn() -> String + Send>) -> Dialog {
        let ep = Endpointer::new(cfg.endpoint.clone());
        Dialog {
            cfg,
            state: State::Asleep,
            ep,
            tools,
            history: Vec::new(),
            last_activity_ms: 0.0,
            next_id: 1,
            turn: None,
            utt_start: 0,
            wake_frame: 0,
            pause_job: None,
            pause_text: None,
            awaiting_text: false,
            end_frame: 0,
            live_job: None,
            last_live_frame: 0,
            speaking_queued: 0,
            alarms: Vec::new(),
            now_ms: 0.0,
            clock,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn history(&self) -> &[Message] {
        &self.history
    }

    pub fn set_tools(&mut self, tools: Vec<ToolDef>) {
        self.tools = tools;
    }

    pub fn tools(&self) -> &[ToolDef] {
        &self.tools
    }

    fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn prompt(&self, extra: Option<&str>) -> String {
        let mut h = self.history.clone();
        if let Some(u) = extra {
            h.push(Message::User(u.to_string()));
        }
        prompt::build(&self.cfg.system, &self.tools, &(self.clock)(), &h)
    }

    /// Wake word heard at `frame` (the end of the phrase). `speaking`: speech goes on right after it.
    pub fn wake(&mut self, frame: u64, speaking: bool, fx: &mut dyn Effects) {
        if matches!(self.state, State::Thinking | State::Speaking) {
            self.interrupt(fx);
        }
        fx.note(Note::Woke);
        fx.chime(Chime::Wake);
        self.listen(frame, speaking, false, fx);
        self.wake_frame = frame;
    }

    fn listen(&mut self, frame: u64, speaking: bool, follow_up: bool, fx: &mut dyn Effects) {
        self.state = State::Listening { follow_up, since_ms: frame_ms(frame), heard: speaking };
        self.ep.restart(frame, speaking);
        self.utt_start = frame;
        self.wake_frame = frame;
        self.pause_job = None;
        self.pause_text = None;
        self.awaiting_text = false;
        self.live_job = None;
        self.last_live_frame = frame;
        fx.note(Note::Listening { follow_up });
    }

    fn interrupt(&mut self, fx: &mut dyn Effects) {
        if let Some(t) = self.turn.take() {
            if t.generating {
                fx.cancel_generation(t.id);
            }
            // What was said before the interruption stays in the conversation.
            if t.confirmed && !t.spoken.is_empty() {
                self.history.push(Message::Assistant(t.spoken.join(" ")));
            }
        }
        fx.stop_speaking();
        self.speaking_queued = 0;
        fx.note(Note::Interrupted);
    }

    fn sleep(&mut self, fx: &mut dyn Effects) {
        self.state = State::Asleep;
        if let Some(t) = self.turn.take()
            && t.generating
        {
            fx.cancel_generation(t.id);
        }
        fx.note(Note::Asleep);
    }

    /// One frame's speech probability (frames are counted from the start of the stream).
    pub fn vad(&mut self, prob: f32, fx: &mut dyn Effects) {
        let frame = self.ep.frame();
        self.now_ms = frame_ms(frame + 1);
        let State::Listening { follow_up, since_ms, heard } = self.state else {
            // Keep the frame count going while not listening.
            self.ep.feed(0.0);
            return;
        };
        if self.awaiting_text {
            self.ep.feed(0.0);
            return;
        }
        let ev = self.ep.feed(prob);
        match ev {
            Some(Ep::Start(f)) => {
                let preroll = (self.cfg.preroll_ms / FRAME_MS) as u64;
                self.utt_start = f.saturating_sub(preroll).max(self.wake_frame);
                self.state = State::Listening { follow_up, since_ms, heard: true };
                self.last_live_frame = f;
            }
            Some(Ep::Pause(f)) => {
                let job = self.id();
                self.pause_job = Some(job);
                self.pause_text = None;
                fx.transcribe(job, self.utt_start, f);
            }
            Some(Ep::Resume) => {
                self.pause_job = None;
                self.pause_text = None;
                if let Some(t) = self.turn.take() {
                    fx.cancel_generation(t.id);
                    fx.note(Note::SpeculationDropped);
                }
            }
            Some(Ep::End(f)) => {
                self.end_frame = f;
                fx.note(Note::EndOfSpeech { speech_end_ms: frame_ms(f), confirmed_ms: self.now_ms });
                if let Some(text) = self.pause_text.clone() {
                    self.finish_utterance(text, fx);
                } else {
                    // Cut by the length limit, or the transcript is late: wait for it.
                    if self.pause_job.is_none() {
                        let job = self.id();
                        self.pause_job = Some(job);
                        fx.transcribe(job, self.utt_start, f);
                    }
                    self.awaiting_text = true;
                    self.state = State::Thinking;
                }
            }
            None => {
                // Live transcript for the display.
                if self.ep.in_speech()
                    && self.cfg.live_transcript_ms > 0.0
                    && self.live_job.is_none()
                    && frame_ms(frame - self.last_live_frame) >= self.cfg.live_transcript_ms
                {
                    let job = self.id();
                    self.live_job = Some(job);
                    self.last_live_frame = frame;
                    fx.transcribe(job, self.utt_start, frame);
                }
                if !heard && !self.ep.in_speech() {
                    let limit = if follow_up { self.cfg.follow_up_ms } else { self.cfg.listen_ms };
                    if self.now_ms - since_ms > limit {
                        self.sleep(fx);
                    }
                }
            }
        }
    }

    pub fn transcript(&mut self, job: u64, text: &str, fx: &mut dyn Effects) {
        let text = clean_transcript(text);
        if Some(job) == self.live_job {
            self.live_job = None;
            if !text.is_empty() && matches!(self.state, State::Listening { .. }) {
                fx.note(Note::Partial(text));
            }
            return;
        }
        if Some(job) != self.pause_job {
            return; // superseded
        }
        self.pause_text = Some(text.clone());
        if self.awaiting_text {
            self.awaiting_text = false;
            self.finish_utterance(text, fx);
            return;
        }
        self.ep.set_transcript(&text);
        if self.cfg.speculate && !text.is_empty() && matches!(self.state, State::Listening { .. }) {
            let id = self.id();
            fx.note(Note::Speculating(text.clone()));
            fx.generate(id, self.prompt(Some(&text)));
            self.turn = Some(Turn::new(id, text, false));
        }
    }

    fn finish_utterance(&mut self, text: String, fx: &mut dyn Effects) {
        if text.is_empty() {
            if let Some(t) = self.turn.take() {
                fx.cancel_generation(t.id);
            }
            let follow_up = matches!(self.state, State::Listening { follow_up: true, .. });
            if follow_up {
                self.sleep(fx);
            } else {
                // Nothing intelligible after the wake word: listen once more.
                let f = self.ep.frame();
                self.listen(f, false, true, fx);
            }
            return;
        }
        if self.now_ms - self.last_activity_ms > self.cfg.forget_after_ms {
            self.history.clear();
        }
        self.last_activity_ms = self.now_ms;
        fx.note(Note::Heard(text.clone()));
        self.state = State::Thinking;
        match self.turn.as_mut() {
            Some(t) if t.transcript == text && !t.confirmed => {
                t.confirmed = true;
                fx.note(Note::SpeculationKept);
                self.history.push(Message::User(text));
                if let Some(a) = self.turn.as_mut().unwrap().held_answer.take().filter(|a| !a.is_empty()) {
                    self.history.push(Message::Assistant(a));
                }
                let held = std::mem::take(&mut self.turn.as_mut().unwrap().held);
                for chunk in held {
                    self.say(chunk, fx);
                }
                if let Some(call) = self.turn.as_mut().unwrap().held_call.take() {
                    self.run_call(call, fx);
                }
                self.maybe_done(fx);
            }
            _ => {
                if let Some(t) = self.turn.take() {
                    fx.cancel_generation(t.id);
                    fx.note(Note::SpeculationDropped);
                }
                let id = self.id();
                fx.generate(id, self.prompt(Some(&text)));
                self.history.push(Message::User(text.clone()));
                self.turn = Some(Turn::new(id, text, true));
            }
        }
    }

    fn say(&mut self, text: String, fx: &mut dyn Effects) {
        let Some(t) = self.turn.as_mut() else { return };
        t.spoken.push(text.clone());
        self.speaking_queued += 1;
        self.state = State::Speaking;
        fx.speak(t.id, text);
    }

    pub fn piece(&mut self, turn: u64, piece: &str, fx: &mut dyn Effects) {
        let Some(t) = self.turn.as_mut().filter(|t| t.id == turn) else { return };
        let chunks = t.chunker.push(piece);
        if t.confirmed {
            for c in chunks {
                self.say(c, fx);
            }
        } else {
            t.held.extend(chunks);
        }
    }

    pub fn generated(&mut self, turn: u64, text: &str, stop: &str, fx: &mut dyn Effects) {
        let Some(t) = self.turn.as_mut().filter(|t| t.id == turn) else { return };
        t.generating = false;
        if stop == "cancel" {
            return;
        }
        let rest = t.chunker.finish();
        let reply = prompt::parse_reply(text);
        if t.confirmed {
            for c in rest {
                self.say(c, fx);
            }
        } else {
            t.held.extend(rest);
        }
        let t = self.turn.as_mut().unwrap();
        if let Some(call) = reply.call {
            // Words before a call stay in the conversation with it.
            if t.confirmed {
                if !reply.text.is_empty() {
                    self.history.push(Message::Assistant(reply.text.clone()));
                }
                self.run_call(call, fx);
            } else {
                t.held_answer = Some(reply.text.clone());
                t.held_call = Some(call);
            }
        } else if reply.bad_call.is_some() {
            if t.confirmed {
                self.say("Sorry, I could not do that.".into(), fx);
            } else {
                t.held.push("Sorry, I could not do that.".into());
            }
            self.turn.as_mut().unwrap().finished = true;
        } else {
            t.finished = true;
            if t.confirmed {
                self.history.push(Message::Assistant(reply.text));
            } else {
                // Complete before the end of speech was confirmed: it joins the conversation when it is.
                t.held_answer = Some(reply.text);
            }
        }
        if self.turn.as_ref().is_some_and(|t| t.confirmed) {
            self.maybe_done(fx);
        }
    }

    fn run_call(&mut self, mut call: ToolCall, fx: &mut dyn Effects) {
        let today = self.cfg.today.unwrap_or_else(|| chrono::Local::now().date_naive());
        let said = self.history.iter().rev().find_map(|m| if let Message::User(u) = m { Some(u.clone()) } else { None }).unwrap_or_default();
        let notes = ground::ground(&mut call, &said, &self.tools, today);
        if !notes.is_empty() {
            fx.note(Note::Grounded(notes));
        }
        let Some(t) = self.turn.as_mut() else { return };
        t.calling = true;
        t.rounds += 1;
        self.history.push(Message::Call(call.clone()));
        fx.note(Note::Calling(call.clone()));
        fx.call_tool(t.id, call);
    }

    pub fn tool_result(&mut self, turn: u64, result: Value, fx: &mut dyn Effects) {
        let Some(t) = self.turn.as_mut().filter(|t| t.id == turn) else { return };
        t.calling = false;
        fx.note(Note::Result(result.clone()));
        self.history.push(Message::Result(result.clone()));
        let say = result.get("say").and_then(|s| s.as_str()).map(str::to_string);
        let rounds = t.rounds;
        if let Some(s) = say {
            self.history.push(Message::Assistant(s.clone()));
            self.turn.as_mut().unwrap().finished = true;
            self.say(s, fx);
            self.maybe_done(fx);
        } else if rounds < self.cfg.max_rounds {
            let id = self.turn.as_ref().unwrap().id;
            let t = self.turn.as_mut().unwrap();
            t.generating = true;
            t.chunker = Chunker::new();
            let p = self.prompt(None);
            fx.generate(id, p);
        } else {
            self.turn.as_mut().unwrap().finished = true;
            self.say("Sorry, that took too many steps.".into(), fx);
        }
    }

    /// The speaker finished one queued piece.
    pub fn spoken(&mut self, fx: &mut dyn Effects) {
        self.speaking_queued = self.speaking_queued.saturating_sub(1);
        if let Some(t) = self.turn.as_ref()
            && let Some(last) = t.spoken.last()
        {
            fx.note(Note::Said(last.clone()));
        }
        self.maybe_done(fx);
    }

    fn maybe_done(&mut self, fx: &mut dyn Effects) {
        let done = self.turn.as_ref().is_none_or(|t| t.finished && !t.generating && !t.calling);
        if done && self.speaking_queued == 0 && matches!(self.state, State::Thinking | State::Speaking) {
            self.turn = None;
            if let Some(a) = (!self.alarms.is_empty()).then(|| self.alarms.remove(0)) {
                self.announce(a, fx);
                return;
            }
            let f = self.ep.frame();
            self.listen(f, false, true, fx);
        }
    }

    /// An alarm from a tool (a timer, a reminder): said now, or after the current answer.
    pub fn alarm(&mut self, text: String, fx: &mut dyn Effects) {
        match self.state {
            State::Asleep | State::Listening { heard: false, .. } => self.announce(text, fx),
            _ => self.alarms.push(text),
        }
    }

    fn announce(&mut self, text: String, fx: &mut dyn Effects) {
        fx.chime(Chime::Alarm);
        let id = self.id();
        let mut t = Turn::new(id, String::new(), true);
        t.generating = false;
        t.finished = true;
        self.turn = Some(t);
        self.say(text, fx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Default)]
    struct Fx {
        log: Vec<String>,
        notes: Vec<Note>,
    }

    impl Effects for Fx {
        fn transcribe(&mut self, job: u64, from: u64, to: u64) {
            self.log.push(format!("transcribe {job} {from}-{to}"));
        }
        fn generate(&mut self, turn: u64, prompt: String) {
            let last = prompt.rsplit("<|im_start|>").nth(1).unwrap_or("").replace('\n', " ");
            self.log.push(format!("generate {turn} [{}]", last.trim()));
        }
        fn cancel_generation(&mut self, turn: u64) {
            self.log.push(format!("cancel {turn}"));
        }
        fn call_tool(&mut self, turn: u64, call: ToolCall) {
            self.log.push(format!("tool {turn} {}", call.name));
        }
        fn speak(&mut self, turn: u64, text: String) {
            self.log.push(format!("speak {turn} {text}"));
        }
        fn stop_speaking(&mut self) {
            self.log.push("stop".into());
        }
        fn chime(&mut self, kind: Chime) {
            self.log.push(format!("chime {kind:?}"));
        }
        fn note(&mut self, what: Note) {
            self.notes.push(what);
        }
    }

    fn dialog() -> Dialog {
        Dialog::new(DialogConfig { live_transcript_ms: 0.0, ..Default::default() }, vec![], Box::new(|| "Now: test.".into()))
    }

    fn frames(d: &mut Dialog, fx: &mut Fx, p: f32, n: usize) {
        for _ in 0..n {
            d.vad(p, fx);
        }
    }

    fn take(fx: &mut Fx) -> Vec<String> {
        std::mem::take(&mut fx.log)
    }

    #[test]
    fn transcripts_lose_the_wake_phrase_and_noise_markers() {
        assert_eq!(clean_transcript(" Hey Courier, set a timer."), "Set a timer.");
        assert_eq!(clean_transcript("Courier what's the time?"), "What's the time?");
        assert_eq!(clean_transcript("[BLANK_AUDIO]"), "");
        assert_eq!(clean_transcript(" you"), "");
        assert_eq!(clean_transcript("Thank you."), "");
        assert_eq!(clean_transcript("The courier is late."), "The courier is late.");
    }

    #[test]
    fn a_speculative_answer_is_kept_when_speech_really_ended() {
        let (mut d, mut fx) = (dialog(), Fx::default());
        frames(&mut d, &mut fx, 0.0, 10);
        d.wake(10, false, &mut fx);
        assert_eq!(take(&mut fx), vec!["chime Wake"]);
        frames(&mut d, &mut fx, 0.9, 30); // speech from frame 10
        frames(&mut d, &mut fx, 0.0, 5); // a pause: transcribe from the start (wake frame) to 40
        assert_eq!(take(&mut fx), vec!["transcribe 2 10-40"]);
        d.transcript(2, "What's the weather like?", &mut fx);
        assert_eq!(take(&mut fx), vec!["generate 3 [user What's the weather like?<|im_end|>]"]);
        // The model answers before the end is confirmed: nothing is spoken yet.
        d.piece(3, "It is sunny and warm. ", &mut fx);
        d.generated(3, "It is sunny and warm. ", "end", &mut fx);
        assert!(take(&mut fx).is_empty());
        // A finished sentence ends after 384 ms of silence.
        frames(&mut d, &mut fx, 0.0, 8);
        assert_eq!(take(&mut fx), vec!["speak 3 It is sunny and warm."]);
        assert!(fx.notes.contains(&Note::SpeculationKept));
        assert_eq!(d.state(), State::Speaking);
        d.spoken(&mut fx);
        assert!(matches!(d.state(), State::Listening { follow_up: true, .. }));
        assert_eq!(d.history(), &[Message::User("What's the weather like?".into()), Message::Assistant("It is sunny and warm.".into())]);
    }

    #[test]
    fn speaking_again_cancels_the_speculation() {
        let (mut d, mut fx) = (dialog(), Fx::default());
        d.wake(0, false, &mut fx);
        frames(&mut d, &mut fx, 0.9, 20);
        frames(&mut d, &mut fx, 0.0, 5);
        d.transcript(2, "Remind me to call", &mut fx);
        take(&mut fx);
        frames(&mut d, &mut fx, 0.9, 20); // "... my mother"
        assert_eq!(take(&mut fx), vec!["cancel 3"]);
        frames(&mut d, &mut fx, 0.0, 5);
        d.transcript(4, "Remind me to call my mother.", &mut fx);
        frames(&mut d, &mut fx, 0.0, 10);
        let log = take(&mut fx);
        assert_eq!(log, vec!["transcribe 4 0-45", "generate 5 [user Remind me to call my mother.<|im_end|>]"]);
    }

    #[test]
    fn tool_calls_wait_for_the_end_and_say_skips_the_second_generation() {
        let (mut d, mut fx) = (dialog(), Fx::default());
        d.wake(0, true, &mut fx);
        frames(&mut d, &mut fx, 0.9, 20);
        frames(&mut d, &mut fx, 0.0, 5);
        d.transcript(2, "Set a timer for five minutes.", &mut fx);
        let call = "<tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"minutes\": 5}}";
        d.piece(3, call, &mut fx);
        d.generated(3, call, "tool", &mut fx);
        take(&mut fx);
        frames(&mut d, &mut fx, 0.0, 8);
        assert_eq!(take(&mut fx), vec!["tool 3 set_timer"]);
        d.tool_result(3, json!({"ok": true, "say": "Timer set for five minutes."}), &mut fx);
        assert_eq!(take(&mut fx), vec!["speak 3 Timer set for five minutes."]);
        d.spoken(&mut fx);
        assert!(matches!(d.state(), State::Listening { follow_up: true, .. }));
        assert!(matches!(d.history()[1], Message::Call(_)));
        assert_eq!(d.history()[3], Message::Assistant("Timer set for five minutes.".into()));
    }

    #[test]
    fn a_result_without_say_goes_back_to_the_model() {
        let (mut d, mut fx) = (dialog(), Fx::default());
        d.wake(0, true, &mut fx);
        frames(&mut d, &mut fx, 0.9, 10);
        frames(&mut d, &mut fx, 0.0, 5);
        d.transcript(2, "What's on my calendar today?", &mut fx);
        frames(&mut d, &mut fx, 0.0, 8);
        let call = "<tool_call>\n{\"name\": \"calendar_events\", \"arguments\": {}}";
        d.generated(3, call, "tool", &mut fx);
        d.tool_result(3, json!({"events": []}), &mut fx);
        let log = take(&mut fx);
        assert!(log.last().unwrap().starts_with("generate 3 [user <tool_response>"), "{log:?}");
        d.piece(3, "Your calendar is empty today.", &mut fx);
        d.generated(3, "Your calendar is empty today.", "end", &mut fx);
        assert_eq!(take(&mut fx), vec!["speak 3 Your calendar is empty today."]);
    }

    #[test]
    fn the_wake_word_interrupts_an_answer() {
        let (mut d, mut fx) = (dialog(), Fx::default());
        d.wake(0, true, &mut fx);
        frames(&mut d, &mut fx, 0.9, 10);
        frames(&mut d, &mut fx, 0.0, 5);
        d.transcript(2, "Tell me a long story.", &mut fx);
        frames(&mut d, &mut fx, 0.0, 8);
        d.piece(3, "Once upon a time there was a fox. ", &mut fx);
        take(&mut fx);
        d.wake(100, false, &mut fx);
        assert_eq!(take(&mut fx), vec!["cancel 3", "stop", "chime Wake"]);
        assert!(matches!(d.state(), State::Listening { follow_up: false, .. }));
        assert_eq!(d.history().last(), Some(&Message::Assistant("Once upon a time there was a fox.".into())));
    }

    #[test]
    fn silence_after_the_wake_word_goes_back_to_sleep() {
        let (mut d, mut fx) = (dialog(), Fx::default());
        d.wake(0, false, &mut fx);
        frames(&mut d, &mut fx, 0.0, 260); // 8.3 s
        assert_eq!(d.state(), State::Asleep);
    }

    #[test]
    fn alarms_wait_for_the_answer_to_end() {
        let (mut d, mut fx) = (dialog(), Fx::default());
        d.alarm("Your tea is ready.".into(), &mut fx);
        assert_eq!(take(&mut fx), vec!["chime Alarm", "speak 2 Your tea is ready."]);
        d.spoken(&mut fx);
        d.wake(0, true, &mut fx);
        frames(&mut d, &mut fx, 0.9, 10);
        frames(&mut d, &mut fx, 0.0, 5);
        d.transcript(3, "Hello.", &mut fx);
        frames(&mut d, &mut fx, 0.0, 8);
        d.piece(4, "Hi there. ", &mut fx);
        take(&mut fx);
        d.alarm("Time to stretch.".into(), &mut fx);
        assert!(take(&mut fx).is_empty());
        d.generated(4, "Hi there. ", "end", &mut fx);
        d.spoken(&mut fx);
        assert_eq!(take(&mut fx), vec!["chime Alarm", "speak 5 Time to stretch."]);
    }

    #[test]
    fn a_late_transcript_is_waited_for() {
        let (mut d, mut fx) = (dialog(), Fx::default());
        d.wake(0, true, &mut fx);
        frames(&mut d, &mut fx, 0.9, 10);
        frames(&mut d, &mut fx, 0.0, 40); // the end comes before the transcript
        assert_eq!(d.state(), State::Thinking);
        take(&mut fx);
        d.transcript(2, "Open the browser.", &mut fx);
        assert_eq!(take(&mut fx), vec!["generate 3 [user Open the browser.<|im_end|>]"]);
    }
}
