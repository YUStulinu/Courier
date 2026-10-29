//! From the model's streamed text to pieces the voice can say. The first piece goes out as soon as it
//! is a clause long, so the voice starts while the model is still writing the rest; later pieces are
//! whole sentences, which sound better. Tool calls are never spoken.

use crate::prompt::TRIGGER;

#[derive(Debug)]
pub struct Chunker {
    buf: String,
    sent_any: bool,
    in_call: bool,
}

/// Words in the first piece before it may end at a comma.
const FIRST_CLAUSE_WORDS: usize = 5;

impl Default for Chunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker {
    pub fn new() -> Chunker {
        Chunker { buf: String::new(), sent_any: false, in_call: false }
    }

    /// Whether a tool call has started (nothing more will be spoken from this generation).
    pub fn in_call(&self) -> bool {
        self.in_call
    }

    pub fn push(&mut self, piece: &str) -> Vec<String> {
        if self.in_call {
            return Vec::new();
        }
        self.buf.push_str(piece);
        if let Some(at) = self.buf.find(TRIGGER) {
            self.buf.truncate(at);
            self.in_call = true;
            return self.finish();
        }
        let mut out = Vec::new();
        loop {
            // Hold back what might be the start of the trigger.
            let held = (1..TRIGGER.len()).rev().find(|&k| self.buf.ends_with(&TRIGGER[..k])).unwrap_or(0);
            let usable = self.buf.len() - held;
            let Some(end) = self.boundary(&self.buf[..usable]) else { break };
            let chunk: String = self.buf.drain(..end).collect();
            let chunk = chunk.trim().to_string();
            if has_words(&chunk) {
                out.push(chunk);
                self.sent_any = true;
            }
        }
        out
    }

    /// What is left at the end of the generation.
    pub fn finish(&mut self) -> Vec<String> {
        let rest = std::mem::take(&mut self.buf);
        let rest = rest.trim().to_string();
        if has_words(&rest) {
            self.sent_any = true;
            vec![rest]
        } else {
            Vec::new()
        }
    }

    /// Where the next piece ends (exclusive), if it is complete: after a sentence end followed by a
    /// space, or, for the first piece, after a comma once it has enough words.
    fn boundary(&self, s: &str) -> Option<usize> {
        let b = s.as_bytes();
        for i in 0..b.len().saturating_sub(1) {
            let c = b[i];
            let next_space = b[i + 1] == b' ' || b[i + 1] == b'\n';
            if !next_space {
                continue;
            }
            if c == b'.' || c == b'!' || c == b'?' {
                // Not after a single capital ("J. Smith") or a number ("3. ").
                if c == b'.' && i >= 1 && (b[i - 1].is_ascii_digit() || (b[i - 1].is_ascii_uppercase() && (i < 2 || b[i - 2] == b' '))) {
                    continue;
                }
                return Some(i + 1);
            }
            if !self.sent_any && (c == b',' || c == b';' || c == b':') && s[..i].split_whitespace().count() >= FIRST_CLAUSE_WORDS {
                return Some(i + 1);
            }
            if c == b'\n' {
                return Some(i + 1);
            }
        }
        None
    }
}

fn has_words(s: &str) -> bool {
    s.chars().any(|c| c.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(pieces: &[&str]) -> Vec<String> {
        let mut c = Chunker::new();
        let mut out = Vec::new();
        for p in pieces {
            out.extend(c.push(p));
        }
        out.extend(c.finish());
        out
    }

    #[test]
    fn sentences_come_out_as_they_end() {
        let mut c = Chunker::new();
        assert!(c.push("It is sunny").is_empty());
        assert_eq!(c.push(" today. Tomor"), vec!["It is sunny today."]);
        assert!(c.push("row it rains").is_empty());
        assert_eq!(c.finish(), vec!["Tomorrow it rains"]);
    }

    #[test]
    fn the_first_piece_may_end_at_a_comma() {
        let out = feed(&["Sure, ", "I set a timer for ten", " minutes, and I will", " ring when it is done."]);
        assert_eq!(out, vec!["Sure, I set a timer for ten minutes,", "and I will ring when it is done."]);
    }

    #[test]
    fn tool_calls_are_not_spoken() {
        let out = feed(&["Let me check. <tool", "_call>\n{\"name\": \"x\"}"]);
        assert_eq!(out, vec!["Let me check."]);
        let mut c = Chunker::new();
        c.push("<tool_call>");
        assert!(c.in_call());
        assert!(c.push("{\"a\": 1}. And more. ").is_empty());
    }

    #[test]
    fn numbers_and_initials_do_not_end_sentences() {
        let out = feed(&["It is 21.5 degrees. Call J. Smith now."]);
        assert_eq!(out, vec!["It is 21.5 degrees.", "Call J. Smith now."]);
    }

    #[test]
    fn empty_pieces_are_dropped() {
        assert!(feed(&[" ... ", "\n"]).is_empty());
    }
}
