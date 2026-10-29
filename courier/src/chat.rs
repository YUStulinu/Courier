//! A turn without audio: text in, the model, tool calls, text out. `courier chat` uses it, and so
//! does the tool-calling evaluation, which needs the model's decisions without the timing of speech.

use crate::engine::Llm;
use crate::ground;
use crate::prompt::{self, Message, ToolCall, ToolDef};
use crate::tools::ToolHost;
use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

pub struct Agent {
    pub llm: Llm,
    pub host: ToolHost,
    pub tools: Vec<ToolDef>,
    pub history: Vec<Message>,
    pub system: String,
    pub place: String,
    /// Constrain calls with the tools' grammar.
    pub grammar: bool,
    pub max_rounds: u32,
    /// Fixed clock text and day (for evaluations that must not depend on today's date).
    pub clock: Option<String>,
    pub today: Option<chrono::NaiveDate>,
    /// Check calls against what was said (ground.rs).
    pub grounding: bool,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Step {
    /// The call as the model wrote it, and as it ran (after grounding).
    pub raw_call: Option<ToolCall>,
    pub call: Option<ToolCall>,
    pub grounded: Vec<String>,
    pub result: Option<Value>,
    pub text: String,
    pub ms: f64,
    pub ms_first: f64,
    pub prompt_tokens: i64,
    pub reused_tokens: i64,
    pub generated: i64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct TurnLog {
    pub steps: Vec<Step>,
    /// What would be spoken.
    pub reply: String,
}

impl Agent {
    pub fn new(llm: Llm, host: ToolHost) -> Agent {
        let tools = host.tools();
        Agent {
            llm,
            host,
            tools,
            history: Vec::new(),
            system: prompt::SYSTEM.to_string(),
            place: String::new(),
            grammar: true,
            max_rounds: 3,
            clock: None,
            today: None,
            grounding: true,
        }
    }

    pub fn respond(&mut self, user: &str, on_piece: &mut dyn FnMut(&str)) -> Result<TurnLog> {
        self.history.push(Message::User(user.to_string()));
        let schema = prompt::call_schema(&self.tools).to_string();
        let mut log = TurnLog::default();
        for round in 0..=self.max_rounds {
            let clock = self.clock.clone().unwrap_or_else(|| prompt::clock(chrono::Local::now(), &self.place));
            let p = prompt::build(&self.system, &self.tools, &clock, &self.history);
            let mut in_call = false;
            let mut cb = |piece: &str| {
                if piece.contains(prompt::TRIGGER) {
                    in_call = true;
                }
                if !in_call {
                    on_piece(piece);
                }
                true
            };
            let g = self.llm.generate(&p, Some(prompt::TRIGGER), self.grammar.then_some(schema.as_str()), 300, &mut cb)?;
            let reply = prompt::parse_reply(&g.text);
            let mut step = Step {
                text: reply.text.clone(),
                ms: g.ms,
                ms_first: g.ms_first,
                prompt_tokens: g.prompt_tokens,
                reused_tokens: g.reused_tokens,
                generated: g.generated,
                ..Default::default()
            };
            match reply.call {
                Some(mut call) if round < self.max_rounds => {
                    step.raw_call = Some(call.clone());
                    if self.grounding {
                        let today = self.today.unwrap_or_else(|| chrono::Local::now().date_naive());
                        step.grounded = ground::ground(&mut call, user, &self.tools, today);
                    }
                    if !reply.text.is_empty() {
                        self.history.push(Message::Assistant(reply.text.clone()));
                    }
                    self.history.push(Message::Call(call.clone()));
                    let result = self.host.call(&call.name, &call.arguments);
                    self.history.push(Message::Result(result.clone()));
                    step.call = Some(call);
                    step.result = Some(result.clone());
                    log.steps.push(step);
                    if let Some(say) = result.get("say").and_then(|s| s.as_str()) {
                        on_piece(say);
                        log.reply = say.to_string();
                        self.history.push(Message::Assistant(log.reply.clone()));
                        return Ok(log);
                    }
                }
                _ => {
                    let text = if reply.bad_call.is_some() && reply.text.is_empty() { "Sorry, I could not do that.".to_string() } else { reply.text };
                    log.reply = text.clone();
                    log.steps.push(step);
                    self.history.push(Message::Assistant(text));
                    return Ok(log);
                }
            }
        }
        Ok(log)
    }
}
