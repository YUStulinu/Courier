//! The conversation as the model reads it: Qwen2.5's chat format (ChatML) with its tool-calling
//! convention (Hermes style): the tools as JSON signatures inside <tools></tools> in the system
//! message, calls as JSON inside <tool_call></tool_call>, results inside <tool_response></tool_response>.
//!
//! The order is chosen for the context cache: the system prompt and the tools (about a thousand
//! tokens) never change between turns, so they come first and are computed once; the clock, which
//! changes every minute, comes after them.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const TRIGGER: &str = "<tool_call>";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// A JSON schema object: {"type": "object", "properties": {...}, "required": [...]}.
    pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    User(String),
    Assistant(String),
    /// A call the assistant made (the JSON inside the tags).
    Call(ToolCall),
    /// What the tool returned, as JSON.
    Result(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

pub const SYSTEM: &str = "You are Courier, a voice assistant that runs entirely on the user's computer. \
Your replies are spoken aloud, so:
- answer in one or two short sentences, in plain words, without lists, markdown, emoji or links;
- never read out file paths or long numbers unless asked; give the name and say you can open it;
- when you used a tool, say what happened or what you found, briefly.
Use a tool when the request needs one: timers, reminders and alarms, the calendar, files, music, the computer, the weather. \
Questions of fact, arithmetic, advice and conversation need no tool: answer them yourself. \
Fill a tool's arguments only with what the user said; leave out what they did not say. \
In arguments, write times as 24-hour \"HH:MM\" and days as the user said them (\"today\", \"tomorrow\", \"friday\"), \
or as \"YYYY-MM-DD\" for a date with a month and day. Do not work out dates yourself. \
If a request is unclear or a required detail is missing, ask one short question instead of guessing.";

/// The system message: instructions and tools. Stable across turns.
pub fn system_message(system: &str, tools: &[ToolDef]) -> String {
    let mut s = String::from("<|im_start|>system\n");
    s.push_str(system);
    if !tools.is_empty() {
        s.push_str("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n");
        s.push_str("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        for t in tools {
            let sig = json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}});
            s.push_str(&py_json(&sig));
            s.push('\n');
        }
        s.push_str(
            "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n",
        );
        s.push_str("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    }
    s.push_str("<|im_end|>\n");
    s
}

/// The whole prompt: the system message, the clock, the conversation, and the assistant's turn.
pub fn build(system: &str, tools: &[ToolDef], clock: &str, history: &[Message]) -> String {
    let mut s = system_message(system, tools);
    s.push_str("<|im_start|>system\n");
    s.push_str(clock);
    s.push_str("<|im_end|>\n");
    let mut i = 0;
    while i < history.len() {
        match &history[i] {
            Message::User(t) => s.push_str(&format!("<|im_start|>user\n{t}<|im_end|>\n")),
            Message::Assistant(t) => s.push_str(&format!("<|im_start|>assistant\n{t}<|im_end|>\n")),
            Message::Call(c) => {
                // A call, perhaps after some words, is one assistant message.
                s.push_str("<|im_start|>assistant\n");
                s.push_str(&format_call(c));
                s.push_str("<|im_end|>\n");
            }
            Message::Result(_) => {
                // Consecutive results go into one user message, as Qwen's template does.
                s.push_str("<|im_start|>user");
                while let Some(Message::Result(v)) = history.get(i) {
                    s.push_str(&format!("\n<tool_response>\n{}\n</tool_response>", py_json(v)));
                    i += 1;
                }
                s.push_str("<|im_end|>\n");
                continue;
            }
        }
        i += 1;
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

pub fn format_call(c: &ToolCall) -> String {
    format!("<tool_call>\n{}\n</tool_call>", py_json(&json!({"name": c.name, "arguments": c.arguments})))
}

/// JSON as Python's json.dumps writes it (", " and ": "), which is what the model saw in training:
/// Qwen's chat template renders tools and calls that way.
pub fn py_json(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let items: Vec<String> = m.iter().map(|(k, v)| format!("{}: {}", Value::String(k.clone()), py_json(v))).collect();
            format!("{{{}}}", items.join(", "))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        other => other.to_string(),
    }
}

/// The local time as the model is told it.
pub fn clock(now: chrono::DateTime<chrono::Local>, place: &str) -> String {
    let mut s = format!("Now: {} (24-hour time).", now.format("%A %Y-%m-%d %H:%M"));
    if !place.is_empty() {
        s.push_str(&format!(" The user is in {place}."));
    }
    s
}

/// The grammar's schema: one of the tools, by name, with arguments that fit its parameters.
pub fn call_schema(tools: &[ToolDef]) -> Value {
    let one: Vec<Value> = tools
        .iter()
        .map(|t| {
            let mut params = t.parameters.clone();
            if let Some(o) = params.as_object_mut() {
                o.insert("additionalProperties".into(), json!(false));
                o.entry("type").or_insert(json!("object"));
                o.entry("properties").or_insert(json!({}));
            }
            json!({
                "type": "object",
                "properties": {"name": {"const": t.name}, "arguments": params},
                "required": ["name", "arguments"],
                "additionalProperties": false
            })
        })
        .collect();
    json!({"anyOf": one})
}

/// What the model said in one generation: the words before any call, and the call.
#[derive(Debug, Default, PartialEq)]
pub struct Reply {
    pub text: String,
    pub call: Option<ToolCall>,
    /// A call that was not valid JSON or not a known shape.
    pub bad_call: Option<String>,
}

pub fn parse_reply(out: &str) -> Reply {
    let Some(at) = out.find(TRIGGER) else {
        return Reply { text: out.trim().to_string(), ..Default::default() };
    };
    let text = out[..at].trim().to_string();
    let rest = out[at + TRIGGER.len()..].trim();
    let body = rest.split("</tool_call>").next().unwrap_or("").trim();
    match serde_json::from_str::<ToolCall>(body) {
        Ok(mut c) => {
            if c.arguments.is_null() {
                c.arguments = json!({});
            }
            Reply { text, call: Some(c), bad_call: None }
        }
        Err(_) => Reply { text, call: None, bad_call: Some(body.to_string()) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<ToolDef> {
        vec![ToolDef {
            name: "set_timer".into(),
            description: "Start a countdown timer.".into(),
            parameters: json!({"type": "object", "properties": {"minutes": {"type": "number"}}, "required": ["minutes"]}),
        }]
    }

    #[test]
    fn prompt_layout() {
        let h = vec![
            Message::User("Set a timer for 5 minutes".into()),
            Message::Call(ToolCall { name: "set_timer".into(), arguments: json!({"minutes": 5}) }),
            Message::Result(json!({"ok": true})),
            Message::Assistant("Timer set for five minutes.".into()),
            Message::User("Thanks".into()),
        ];
        let p = build("SYS", &tools(), "Now: Monday 2026-10-26 10:00.", &h);
        assert!(p.starts_with("<|im_start|>system\nSYS\n\n# Tools"));
        assert!(p.contains("{\"type\": \"function\", \"function\": {\"name\": \"set_timer\", \"description\": \"Start a countdown timer.\""));
        assert!(p.contains("<|im_start|>assistant\n<tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"minutes\": 5}}\n</tool_call><|im_end|>"));
        assert!(p.contains("<|im_start|>user\n<tool_response>\n{\"ok\": true}\n</tool_response><|im_end|>"));
        assert!(p.ends_with("<|im_start|>user\nThanks<|im_end|>\n<|im_start|>assistant\n"));
        // The clock comes after the tools, so a new minute keeps the tools cached.
        assert!(p.find("Now:").unwrap() > p.find("</tools>").unwrap());
    }

    #[test]
    fn consecutive_results_share_one_message() {
        let h = vec![Message::Result(json!(1)), Message::Result(json!(2))];
        let p = build("S", &[], "C", &h);
        assert_eq!(p.matches("<|im_start|>user").count(), 1);
        assert_eq!(p.matches("<tool_response>").count(), 2);
    }

    #[test]
    fn parsing_replies() {
        assert_eq!(parse_reply(" Hello there. ").text, "Hello there.");
        let r = parse_reply("<tool_call>\n{\"name\": \"set_timer\", \"arguments\": {\"minutes\": 3}}");
        assert_eq!(r.call.unwrap(), ToolCall { name: "set_timer".into(), arguments: json!({"minutes": 3}) });
        let r = parse_reply("Sure.<tool_call>\n{\"name\": \"x\"}\n</tool_call>");
        assert_eq!(r.text, "Sure.");
        assert_eq!(r.call.unwrap().arguments, json!({}));
        assert!(parse_reply("<tool_call>\n{\"name\": ").bad_call.is_some());
    }

    #[test]
    fn schema_allows_only_known_tools() {
        let s = call_schema(&tools());
        assert_eq!(s["anyOf"][0]["properties"]["name"]["const"], "set_timer");
        assert_eq!(s["anyOf"][0]["properties"]["arguments"]["additionalProperties"], false);
    }
}
