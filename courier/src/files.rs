//! Searching your files by name and, for text files, by content. The search plugin (Lua) decides
//! where to look and how to answer; the walking and matching are here, where they are fast.

use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

#[derive(Debug, Clone)]
pub struct Query {
    pub roots: Vec<PathBuf>,
    pub terms: Vec<String>,
    /// Extensions to keep (lowercase, no dot); empty keeps every file.
    pub exts: Vec<String>,
    pub limit: usize,
    pub content: bool,
    pub budget: Duration,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub path: String,
    pub name: String,
    pub folder: String,
    pub modified: String,
    pub score: f64,
    /// For a content match: the line it was found on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

const SKIP_DIRS: &[&str] = &["node_modules", "target", "build", "dist", "__pycache__", "venv", ".venv", "appdata", "$recycle.bin"];
const TEXT_EXTS: &[&str] = &["txt", "md", "csv", "json", "log", "ini", "toml", "yaml", "yml", "tex", "rst", "org", "html", "xml"];
const MAX_TEXT_BYTES: u64 = 1 << 20;
const MAX_FILES: usize = 200_000;

/// "Taxes, 2025!" -> ["tax", "2025"]: lowercase words, a plural "s" dropped, short words kept only
/// if they are numbers.
pub fn terms(query: &str) -> Vec<String> {
    const STOP: &[&str] = &["the", "my", "a", "an", "of", "for", "about", "file", "files", "document", "documents", "with", "and", "in", "on", "to"];
    query
        .split(|c: char| !c.is_alphanumeric())
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty() && !STOP.contains(&w.as_str()))
        .filter(|w| w.len() > 1 || w.chars().all(|c| c.is_ascii_digit()))
        .map(|w| stem(&w))
        .collect()
}

fn stem(w: &str) -> String {
    if w.len() > 4 && w.ends_with("ies") {
        return format!("{}y", &w[..w.len() - 3]);
    }
    if w.len() > 4 && (w.ends_with("ses") || w.ends_with("xes")) {
        return w[..w.len() - 2].to_string();
    }
    if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") {
        return w[..w.len() - 1].to_string();
    }
    w.to_string()
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| stem(&w.to_lowercase())).collect()
}

/// How well a name matches: whole words count more than parts of words.
fn name_score(name_words: &[String], name_lower: &str, term: &str) -> f64 {
    if name_words.iter().any(|w| w == term) {
        3.0
    } else if name_words.iter().any(|w| w.starts_with(term)) {
        2.0
    } else if term.len() >= 4 && name_lower.contains(term) {
        1.5
    } else {
        0.0
    }
}

pub fn search(q: &Query) -> Vec<Hit> {
    let start = Instant::now();
    let mut hits = Vec::new();
    let mut seen = 0usize;
    if q.terms.is_empty() {
        return hits;
    }
    let mut stack: Vec<PathBuf> = q.roots.iter().filter(|r| r.is_dir()).cloned().collect();
    while let Some(dir) = stack.pop() {
        if start.elapsed() > q.budget || seen > MAX_FILES {
            break;
        }
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let lower = name.to_lowercase();
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                if !lower.starts_with('.') && !SKIP_DIRS.contains(&lower.as_str()) {
                    stack.push(entry.path());
                }
                continue;
            }
            if !ft.is_file() {
                continue;
            }
            seen += 1;
            let ext = Path::new(&lower).extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
            if !q.exts.is_empty() && !q.exts.contains(&ext) {
                continue;
            }
            let path = entry.path();
            if let Some(h) = score_file(q, &path, &name, &lower, &ext) {
                hits.push(h);
            }
        }
    }
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then(b.modified.cmp(&a.modified)));
    hits.truncate(q.limit);
    hits
}

fn score_file(q: &Query, path: &Path, name: &str, lower: &str, ext: &str) -> Option<Hit> {
    let stem_lower = lower.rsplit_once('.').map_or(lower, |(s, _)| s);
    let name_words = words(stem_lower);
    let folder = path.parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    let folder_words = words(&folder.to_lowercase());
    let mut score = 0.0;
    let mut missing = Vec::new();
    for t in &q.terms {
        let s = name_score(&name_words, stem_lower, t);
        if s > 0.0 {
            score += s;
        } else if folder_words.iter().any(|w| w == t) {
            score += 1.0;
        } else {
            missing.push(t.clone());
        }
    }
    let mut snippet = None;
    // Terms not in the name or folder may be in the text.
    if !missing.is_empty()
        && q.content
        && TEXT_EXTS.contains(&ext)
        && let Ok(meta) = fs::metadata(path)
        && meta.len() <= MAX_TEXT_BYTES
        && let Ok(text) = fs::read_to_string(path)
    {
        let text_lower = text.to_lowercase();
        let found: Vec<String> = missing.iter().filter(|t| text_lower.contains(t.as_str())).cloned().collect();
        if !found.is_empty() {
            score += found.len() as f64;
            let first = found[0].as_str();
            snippet = text.lines().find(|l| l.to_lowercase().contains(first)).map(|l| l.trim().chars().take(120).collect());
            missing.retain(|t| !found.contains(t));
        }
    }
    // Every term must be somewhere; a name match on all of them ranks first.
    if !missing.is_empty() || score == 0.0 {
        return None;
    }
    let modified = fs::metadata(path).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
    let modified: chrono::DateTime<chrono::Local> = modified.into();
    Some(Hit {
        path: path.to_string_lossy().into_owned(),
        name: name.to_string(),
        folder,
        modified: modified.format("%Y-%m-%d %H:%M").to_string(),
        score,
        snippet,
    })
}

/// Files under `dir` with one of `exts` (lowercase), at most `max`, in path order.
pub fn list(dir: &Path, exts: &[String], max: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = fs::read_dir(&d) else { continue };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.file_name()));
        for e in entries {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if exts.is_empty() || p.extension().is_some_and(|x| exts.contains(&x.to_string_lossy().to_lowercase())) {
                out.push(p);
                if out.len() >= max {
                    return out;
                }
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("courier-files-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("Documents/Taxes")).unwrap();
        fs::create_dir_all(d.join("node_modules/pkg")).unwrap();
        fs::write(d.join("Documents/Taxes/2025 return.pdf"), b"%PDF").unwrap();
        fs::write(d.join("Documents/notes.md"), "Groceries\nCall the plumber about the boiler\n").unwrap();
        fs::write(d.join("Documents/boiler manual.txt"), "manual").unwrap();
        fs::write(d.join("node_modules/pkg/boiler.txt"), "skip me").unwrap();
        d
    }

    fn query(d: &Path, q: &str) -> Vec<Hit> {
        search(&Query { roots: vec![d.to_path_buf()], terms: terms(q), exts: vec![], limit: 10, content: true, budget: Duration::from_secs(5) })
    }

    #[test]
    fn terms_drop_filler_and_plurals() {
        assert_eq!(terms("my notes about the taxes, 2025"), vec!["note", "tax", "2025"]);
        assert_eq!(terms("Batteries"), vec!["battery"]);
    }

    #[test]
    fn names_rank_above_contents_and_skipped_folders_are_skipped() {
        let d = tree("rank");
        let hits = query(&d, "boiler");
        let names: Vec<&str> = hits.iter().map(|h| h.name.as_str()).collect();
        assert_eq!(names, vec!["boiler manual.txt", "notes.md"]);
        assert!(hits[1].snippet.as_deref().unwrap().contains("plumber"));
    }

    #[test]
    fn folders_count_and_every_term_is_needed() {
        let d = tree("folders");
        let hits = query(&d, "taxes 2025");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "2025 return.pdf");
        assert!(query(&d, "boiler 2025").is_empty());
    }

    #[test]
    fn list_filters_by_extension() {
        let d = tree("list");
        let l = list(&d, &["txt".into()], 10);
        assert_eq!(l.len(), 2);
    }
}
