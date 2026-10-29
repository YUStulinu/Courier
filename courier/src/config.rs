//! courier.toml: where the models are, how the assistant behaves, and each plugin's settings. Paths
//! are relative to the file's folder. Everything has a default; an empty file works in a checkout.

use crate::endpoint::EndpointConfig;
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Models {
    pub llm: PathBuf,
    pub whisper: PathBuf,
    pub voice: PathBuf,
    pub cmudict: PathBuf,
    pub vad: PathBuf,
    pub wake: PathBuf,
    /// Layers of the language model on the GPU (0: CPU only).
    pub gpu_layers: i32,
    pub context: i32,
    /// Whisper on the GPU, if the engine was built with CUDA.
    pub whisper_gpu: bool,
    /// Whisper's encoder window cut to the audio's length: faster, a little less accurate.
    pub whisper_fast: bool,
}

impl Default for Models {
    fn default() -> Self {
        Models {
            llm: "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf".into(),
            whisper: "third_party/models/ggml-small.en-q5_1.bin".into(),
            voice: "third_party/models/en_US-ljspeech-medium.onnx".into(),
            cmudict: "third_party/models/cmudict.dict".into(),
            vad: "third_party/models/silero_vad-6.2.3.onnx".into(),
            wake: "models/wake/hey_courier.onnx".into(),
            gpu_layers: 99,
            context: 4096,
            whisper_gpu: true,
            whisper_fast: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Wake {
    /// Empty: the threshold chosen in training (models/wake/hey_courier.json).
    pub threshold: Option<f32>,
    pub hits: i32,
    pub refractory_ms: i32,
}

impl Default for Wake {
    fn default() -> Self {
        Wake { threshold: None, hits: 2, refractory_ms: 1500 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Voice {
    pub speaker: u32,
    pub speed: f32,
}

impl Default for Voice {
    fn default() -> Self {
        Voice { speaker: 0, speed: 1.0 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Assistant {
    /// Where you are, for the weather and the model ("Bucharest, Romania").
    pub place: String,
    /// Whether plugins may use the network (only the weather does).
    pub network: bool,
    pub speculate: bool,
    pub follow_up_s: f64,
    pub listen_s: f64,
    /// Show what Whisper hears while you speak.
    pub live_transcript: bool,
    /// Where plugins keep their data; empty: the system's data folder (%APPDATA%\Courier).
    pub data_dir: PathBuf,
    pub plugins_dir: PathBuf,
}

impl Default for Assistant {
    fn default() -> Self {
        Assistant {
            place: String::new(),
            network: true,
            speculate: true,
            follow_up_s: 5.0,
            listen_s: 8.0,
            live_transcript: true,
            data_dir: PathBuf::new(),
            plugins_dir: "plugins".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub models: Models,
    pub wake: Wake,
    pub voice: Voice,
    pub assistant: Assistant,
    pub endpoint: EndpointConfig,
    /// [plugins.<file name>] tables, given to each plugin as it loads.
    pub plugins: BTreeMap<String, Value>,
    #[serde(skip)]
    pub root: PathBuf,
}

impl Config {
    /// Reads `path` if it exists; the defaults otherwise. Paths are resolved against its folder.
    pub fn load(path: &Path) -> Result<Config> {
        let root = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let root = if root.as_os_str().is_empty() { PathBuf::from(".") } else { root };
        let mut c: Config = if path.exists() {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            toml::from_str(&text).with_context(|| format!("in {}", path.display()))?
        } else {
            Config::default()
        };
        c.root = std::fs::canonicalize(&root).map(plain).unwrap_or(root);
        Ok(c)
    }

    pub fn path(&self, p: &Path) -> PathBuf {
        if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) }
    }

    pub fn data_dir(&self) -> PathBuf {
        if self.assistant.data_dir.as_os_str().is_empty() {
            dirs::data_dir().unwrap_or_else(std::env::temp_dir).join("Courier")
        } else {
            self.path(&self.assistant.data_dir)
        }
    }

    /// The wake threshold: the configured one, else the one chosen in training, else 0.5.
    pub fn wake_threshold(&self) -> f32 {
        if let Some(t) = self.wake.threshold {
            return t;
        }
        let info = self.path(&self.models.wake).with_extension("json");
        std::fs::read_to_string(info)
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v["threshold"].as_f64())
            .map(|t| t as f32)
            .unwrap_or(0.5)
    }
}

/// Without Windows' verbatim prefix (`\\?\C:\...`), which other programs and Lua's `require` choke on.
pub fn plain(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_overrides() {
        let d = std::env::temp_dir().join(format!("courier-config-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("courier.toml");
        std::fs::write(&p, "[assistant]\nplace = \"Bucharest\"\n[endpoint]\nend_complete_ms = 300\n[plugins.music]\ndir = \"D:/Music\"\n").unwrap();
        let c = Config::load(&p).unwrap();
        assert_eq!(c.assistant.place, "Bucharest");
        assert_eq!(c.endpoint.end_complete_ms, 300.0);
        assert_eq!(c.endpoint.end_incomplete_ms, EndpointConfig::default().end_incomplete_ms);
        assert_eq!(c.plugins["music"]["dir"], "D:/Music");
        assert!(c.path(&c.models.llm).starts_with(plain(std::fs::canonicalize(&d).unwrap())));
        assert!(!c.root.to_string_lossy().starts_with(r"\\?\"));
        let none = Config::load(&d.join("missing.toml")).unwrap();
        assert!(none.assistant.speculate);
    }
}
