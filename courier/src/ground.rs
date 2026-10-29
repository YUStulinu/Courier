//! Checking a call's arguments against what the user actually said, before the tool runs.
//!
//! A 3B model is good at choosing the tool and copying words, and bad at arithmetic: asked for
//! "Saturday" on a Thursday it writes a date that is Friday; asked to remind you to buy milk it may
//! invent "at 14:00". So:
//!   - a `date` argument follows the day the user named ("tomorrow", "saturday"): it is added if the
//!     model left it out, replaced if the model computed another day, and dropped if the user named
//!     no day and the model wrote today's date anyway;
//!   - an optional `time` or `after` whose numbers the user never said is dropped (the tool then asks).
//!
//! The tools read the day words themselves (plugins/lib/when.lua).

use crate::prompt::{ToolCall, ToolDef};
use chrono::{Datelike, Duration, NaiveDate};
use serde_json::Value;

const WEEKDAYS: [&str; 7] = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
const MONTHS: [&str; 12] = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != ':').filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect()
}

/// The day the user named, as a word the tools understand: "today", "tomorrow", "friday", "next friday".
pub fn day_said(said: &str) -> Option<String> {
    let w = words(said);
    let mut found = Vec::new();
    for (i, x) in w.iter().enumerate() {
        let day = match x.as_str() {
            "today" | "tonight" => Some("today".to_string()),
            "tomorrow" => Some("tomorrow".to_string()),
            d if WEEKDAYS.contains(&d) => Some(if i > 0 && w[i - 1] == "next" { format!("next {d}") } else { d.to_string() }),
            _ => None,
        };
        if let Some(d) = day
            && !found.contains(&d)
        {
            found.push(d);
        }
    }
    // Two different days ("from monday to friday") are for the model to sort out.
    (found.len() == 1).then(|| found.remove(0))
}

/// Whether the user named a calendar date ("November 12", "the 3rd", "12/11").
fn date_said(said: &str) -> bool {
    let w = words(said);
    w.iter().any(|x| {
        MONTHS.contains(&x.as_str())
            || (x.len() > 2 && x[..x.len() - 2].chars().all(|c| c.is_ascii_digit()) && ["st", "nd", "rd", "th"].contains(&&x[x.len() - 2..]))
    }) || said.chars().filter(|c| *c == '/').count() >= 1
}

/// The day a day word means, counted from `today`.
pub fn resolve(day: &str, today: NaiveDate) -> Option<NaiveDate> {
    let d = day.trim().to_lowercase();
    if let Ok(x) = NaiveDate::parse_from_str(&d, "%Y-%m-%d") {
        return Some(x);
    }
    let (next, name) = match d.strip_prefix("next ") {
        Some(n) => (true, n.to_string()),
        None => (false, d.strip_prefix("on ").unwrap_or(&d).to_string()),
    };
    match name.as_str() {
        "today" | "tonight" => return Some(today),
        "tomorrow" => return Some(today + Duration::days(1)),
        _ => {}
    }
    let i = WEEKDAYS.iter().position(|w| *w == name || w[..3] == name)? as i64;
    let mut ahead = (i - today.weekday().num_days_from_monday() as i64).rem_euclid(7);
    if next && ahead == 0 {
        ahead = 7;
    }
    Some(today + Duration::days(ahead))
}

const NUMBER_WORDS: [(&str, i64); 29] = [
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
    ("thirteen", 13),
    ("fourteen", 14),
    ("fifteen", 15),
    ("sixteen", 16),
    ("seventeen", 17),
    ("eighteen", 18),
    ("nineteen", 19),
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("ninety", 90),
    ("noon", 12),
    ("midnight", 0),
    ("half", -1),
    ("quarter", -1),
];

/// The numbers in a text, from digits ("9:15" gives 9 and 15) and words ("forty five" gives 40, 5, 45).
fn numbers(s: &str) -> Vec<i64> {
    let mut out = Vec::new();
    let mut last_tens: Option<i64> = None;
    for w in s.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        if w.is_empty() {
            continue;
        }
        let digits: String = w.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<i64>() {
            out.push(n);
            last_tens = None;
            continue;
        }
        if let Some(&(_, n)) = NUMBER_WORDS.iter().find(|(x, _)| *x == w) {
            out.push(n);
            if let (Some(t), true) = (last_tens, n < 10) {
                out.push(t + n);
            }
            last_tens = (n >= 20 && n % 10 == 0).then_some(n);
        } else {
            last_tens = None;
        }
    }
    out
}

/// Whether every number in an argument was said by the user (an hour of 18 counts as 6 too).
fn numbers_said(arg: &str, said: &[i64]) -> bool {
    numbers(arg).iter().filter(|&&n| n > 0).all(|&n| said.contains(&n) || (n > 12 && said.contains(&(n - 12))) || (n == 30 && said.contains(&-1)))
}

/// Grounds a call in place; returns what was changed, for the log.
pub fn ground(call: &mut ToolCall, said: &str, tools: &[ToolDef], today: NaiveDate) -> Vec<String> {
    let mut notes = Vec::new();
    let Some(def) = tools.iter().find(|t| t.name == call.name) else { return notes };
    let props = def.parameters.get("properties").and_then(|p| p.as_object()).cloned().unwrap_or_default();
    let required: Vec<String> = def
        .parameters
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let Some(args) = call.arguments.as_object_mut() else { return notes };

    if props.contains_key("date") {
        let said_day = day_said(said);
        let given = args.get("date").and_then(|d| d.as_str()).map(str::to_string);
        match (said_day, given) {
            (Some(day), None) if !args.contains_key("after") => {
                notes.push(format!("date: added \"{day}\""));
                args.insert("date".into(), Value::String(day));
            }
            (Some(day), Some(g)) if resolve(&day, today) != resolve(&g, today) => {
                notes.push(format!("date: \"{g}\" -> \"{day}\""));
                args.insert("date".into(), Value::String(day));
            }
            (None, Some(g)) if !date_said(said) && resolve(&g, today) == Some(today) && !required.contains(&"date".to_string()) => {
                notes.push(format!("date: dropped \"{g}\" (no day was said)"));
                args.remove("date");
            }
            _ => {}
        }
    }
    let said_numbers = numbers(said);
    for key in ["time", "after"] {
        if required.iter().any(|r| r == key) {
            continue;
        }
        if let Some(v) = args.get(key) {
            let text = match v {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                _ => continue,
            };
            // A delay of zero is no delay: the model filled the field for the sake of it.
            if !numbers_said(&text, &said_numbers) || (key == "after" && numbers(&text).contains(&0)) {
                notes.push(format!("{key}: dropped \"{text}\" (not said)"));
                args.remove(key);
            }
        }
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn thursday() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 10, 29).unwrap()
    }

    fn tools() -> Vec<ToolDef> {
        let p = |props: Value, req: Value| json!({"type": "object", "properties": props, "required": req});
        vec![
            ToolDef { name: "get_weather".into(), description: String::new(), parameters: p(json!({"place": {}, "date": {}}), json!([])) },
            ToolDef {
                name: "add_reminder".into(),
                description: String::new(),
                parameters: p(json!({"text": {}, "time": {}, "date": {}, "after": {}}), json!(["text"])),
            },
            ToolDef {
                name: "add_event".into(),
                description: String::new(),
                parameters: p(json!({"title": {}, "date": {}, "time": {}}), json!(["title", "date"])),
            },
        ]
    }

    fn run(name: &str, args: Value, said: &str) -> Value {
        let mut c = ToolCall { name: name.into(), arguments: args };
        ground(&mut c, said, &tools(), thursday());
        c.arguments
    }

    #[test]
    fn days_said() {
        assert_eq!(day_said("what's the weather like on Sunday?").as_deref(), Some("sunday"));
        assert_eq!(day_said("remind me tomorrow at eight").as_deref(), Some("tomorrow"));
        assert_eq!(day_said("next Friday please").as_deref(), Some("next friday"));
        assert_eq!(day_said("from monday to friday"), None);
        assert_eq!(day_said("set a timer"), None);
        assert_eq!(resolve("saturday", thursday()).unwrap().to_string(), "2026-10-31");
        assert_eq!(resolve("thursday", thursday()).unwrap().to_string(), "2026-10-29");
        assert_eq!(resolve("next thursday", thursday()).unwrap().to_string(), "2026-11-05");
    }

    #[test]
    fn a_computed_date_follows_the_day_said() {
        assert_eq!(run("get_weather", json!({"place": "Rome", "date": "2026-10-30"}), "weather in Rome on saturday")["date"], "saturday");
        assert_eq!(run("get_weather", json!({"date": "2026-10-31"}), "weather on saturday")["date"], "2026-10-31");
        assert_eq!(run("add_reminder", json!({"text": "x", "time": "08:00"}), "remind me tomorrow at eight to x")["date"], "tomorrow");
    }

    #[test]
    fn invented_values_are_dropped() {
        let a =
            run("add_reminder", json!({"text": "buy milk", "time": "14:00", "date": "2026-10-29", "after": "0 minutes"}), "remind me to buy milk");
        assert_eq!(a, json!({"text": "buy milk"}));
        // Said: kept.
        let a = run("add_reminder", json!({"text": "call mother", "time": "18:00"}), "remind me to call my mother at six pm");
        assert_eq!(a["time"], "18:00");
        let a = run("add_reminder", json!({"text": "x", "time": "09:15"}), "remind me at 9:15 to x");
        assert_eq!(a["time"], "09:15");
        let a = run("add_reminder", json!({"text": "x", "after": "forty five minutes"}), "in forty five minutes remind me to x");
        assert_eq!(a["after"], "forty five minutes");
        let a = run("add_reminder", json!({"text": "x", "after": "half an hour"}), "in half an hour remind me to x");
        assert_eq!(a["after"], "half an hour");
        // A required date stays, and a calendar date the user named is not "invented".
        assert_eq!(run("add_event", json!({"title": "x", "date": "2026-10-29"}), "add x")["date"], "2026-10-29");
        assert_eq!(run("get_weather", json!({"date": "2026-10-29"}), "weather on October 29th")["date"], "2026-10-29");
        assert!(run("get_weather", json!({"date": "today"}), "how cold is it").get("date").is_none());
    }

    #[test]
    fn numbers_in_words() {
        assert!(numbers("forty five").contains(&45));
        assert!(numbers("at 9:15").contains(&15));
        assert!(numbers_said("18:00", &numbers("at six pm")));
        assert!(!numbers_said("14:00", &numbers("buy milk")));
        assert!(numbers_said("12:00", &numbers("at noon")));
    }
}
