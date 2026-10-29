//! The C++ engine (engine/include/courier.h), loaded at run time. Loading it late keeps the
//! orchestrator's own logic (dialog, endpointing, prompts, plugins) buildable and testable without it.

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;
use serde::Deserialize;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type PieceCb = unsafe extern "C" fn(*const c_char, *mut c_void) -> c_int;

#[allow(non_snake_case)]
struct Api {
    last_error: unsafe extern "C" fn() -> *const c_char,
    free_string: unsafe extern "C" fn(*mut c_char),
    free_pcm: unsafe extern "C" fn(*mut i16),
    has_gpu: unsafe extern "C" fn() -> c_int,
    vad_new: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    vad_process: unsafe extern "C" fn(*mut c_void, *const f32, c_int, *mut f32, c_int) -> c_int,
    vad_reset: unsafe extern "C" fn(*mut c_void),
    vad_free: unsafe extern "C" fn(*mut c_void),
    wake_new: unsafe extern "C" fn(*const c_char, f32, c_int, c_int) -> *mut c_void,
    wake_process: unsafe extern "C" fn(*mut c_void, *const f32, c_int, *mut f32, c_int, *mut c_int) -> c_int,
    wake_last_ms: unsafe extern "C" fn(*const c_void) -> f64,
    wake_set_threshold: unsafe extern "C" fn(*mut c_void, f32),
    wake_reset: unsafe extern "C" fn(*mut c_void),
    wake_free: unsafe extern "C" fn(*mut c_void),
    asr_new: unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void,
    asr_transcribe: unsafe extern "C" fn(*mut c_void, *const f32, c_int, c_int) -> *mut c_char,
    asr_free: unsafe extern "C" fn(*mut c_void),
    llm_new: unsafe extern "C" fn(*const c_char, c_int, c_int) -> *mut c_void,
    llm_generate: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, *const c_char, c_int, Option<PieceCb>, *mut c_void) -> *mut c_char,
    llm_count_tokens: unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int,
    llm_free: unsafe extern "C" fn(*mut c_void),
    tts_new: unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, c_int) -> *mut c_void,
    tts_rate: unsafe extern "C" fn(*const c_void) -> c_int,
    tts_speakers: unsafe extern "C" fn(*const c_void) -> c_int,
    tts_speak: unsafe extern "C" fn(*mut c_void, *const c_char, c_int, f32, f32, *mut *mut i16) -> c_int,
    tts_explain: unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_char,
    tts_free: unsafe extern "C" fn(*mut c_void),
    _lib: Library,
}

/// The loaded library. Cheap to clone; every model handle keeps one.
#[derive(Clone)]
pub struct Engine(Arc<Api>);

fn library_name() -> &'static str {
    if cfg!(windows) { "courier_engine.dll" } else { "libcourier_engine.so" }
}

/// Where the engine library may be: COURIER_ENGINE, next to the program, then the build folders of
/// a checkout (the CUDA build first).
pub fn candidates(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("COURIER_ENGINE") {
        out.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        out.push(dir.join(library_name()));
    }
    for b in ["engine/build-cuda", "engine/build/Release", "engine/build"] {
        out.push(root.join(b).join(library_name()));
    }
    out
}

impl Engine {
    pub fn load(root: &Path) -> Result<Engine> {
        let path = candidates(root)
            .into_iter()
            .find(|p| p.is_file())
            .ok_or_else(|| anyhow!("the engine library ({}) was not found; build engine/ first or set COURIER_ENGINE", library_name()))?;
        Self::load_from(&path)
    }

    pub fn load_from(path: &Path) -> Result<Engine> {
        // SAFETY: loading our own library; its initialisers only set up static tables.
        let lib = unsafe { open(path) }.with_context(|| format!("loading {}", path.display()))?;
        macro_rules! sym {
            ($name:literal) => {
                // SAFETY: the signatures match engine/include/courier.h.
                unsafe { *lib.get(concat!($name, "\0").as_bytes()).with_context(|| format!("{} is missing {}", path.display(), $name))? }
            };
        }
        let api = Api {
            last_error: sym!("courier_last_error"),
            free_string: sym!("courier_free_string"),
            free_pcm: sym!("courier_free_pcm"),
            has_gpu: sym!("courier_has_gpu"),
            vad_new: sym!("courier_vad_new"),
            vad_process: sym!("courier_vad_process"),
            vad_reset: sym!("courier_vad_reset"),
            vad_free: sym!("courier_vad_free"),
            wake_new: sym!("courier_wake_new"),
            wake_process: sym!("courier_wake_process"),
            wake_last_ms: sym!("courier_wake_last_ms"),
            wake_set_threshold: sym!("courier_wake_set_threshold"),
            wake_reset: sym!("courier_wake_reset"),
            wake_free: sym!("courier_wake_free"),
            asr_new: sym!("courier_asr_new"),
            asr_transcribe: sym!("courier_asr_transcribe"),
            asr_free: sym!("courier_asr_free"),
            llm_new: sym!("courier_llm_new"),
            llm_generate: sym!("courier_llm_generate"),
            llm_count_tokens: sym!("courier_llm_count_tokens"),
            llm_free: sym!("courier_llm_free"),
            tts_new: sym!("courier_tts_new"),
            tts_rate: sym!("courier_tts_rate"),
            tts_speakers: sym!("courier_tts_speakers"),
            tts_speak: sym!("courier_tts_speak"),
            tts_explain: sym!("courier_tts_explain"),
            tts_free: sym!("courier_tts_free"),
            _lib: lib,
        };
        Ok(Engine(Arc::new(api)))
    }

    pub fn has_gpu(&self) -> bool {
        // SAFETY: no arguments.
        unsafe { (self.0.has_gpu)() != 0 }
    }

    fn error(&self, what: &str) -> anyhow::Error {
        // SAFETY: the engine returns a valid C string owned by the calling thread.
        let msg = unsafe { CStr::from_ptr((self.0.last_error)()) }.to_string_lossy().into_owned();
        anyhow!("{what}: {msg}")
    }

    fn take_string(&self, p: *mut c_char, what: &str) -> Result<String> {
        if p.is_null() {
            return Err(self.error(what));
        }
        // SAFETY: a malloc'ed, NUL-terminated string from the engine, freed right after copying.
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        unsafe { (self.0.free_string)(p) };
        Ok(s)
    }
}

/// Opens the library so that its own folder is searched first for what it needs: Windows 11 ships an
/// older onnxruntime.dll in System32, which would otherwise be found before the one next to ours.
unsafe fn open(path: &Path) -> std::result::Result<Library, libloading::Error> {
    #[cfg(windows)]
    {
        use libloading::os::windows::{LOAD_WITH_ALTERED_SEARCH_PATH, Library as WinLibrary};
        // The altered search only works on a full path with backslashes ("/c/src/..." from a bash
        // prompt is not one).
        let full = std::fs::canonicalize(path).map(crate::config::plain).unwrap_or_else(|_| path.to_path_buf());
        let path = full.as_path();
        // SAFETY: as for Library::new; the flag only changes where dependencies are looked for.
        unsafe { WinLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }.map(Library::from)
    }
    #[cfg(not(windows))]
    {
        // SAFETY: as above.
        unsafe { Library::new(path) }
    }
}

fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| anyhow!("text with a NUL byte"))
}

fn cpath(p: &Path) -> Result<CString> {
    if !p.exists() {
        bail!("{} not found", p.display());
    }
    cstr(&p.to_string_lossy())
}

/// Speech probability per 32 ms frame (512 samples).
pub struct Vad {
    e: Engine,
    h: *mut c_void,
}
// SAFETY: the handle is used from one thread at a time (the owner moves it, never shares it).
unsafe impl Send for Vad {}

impl Vad {
    pub const FRAME: usize = 512;

    /// Silero with a model file, or the energy detector without one.
    pub fn new(e: &Engine, model: Option<&Path>) -> Result<Vad> {
        let m = model.map(cpath).transpose()?;
        // SAFETY: a valid path or NULL.
        let h = unsafe { (e.0.vad_new)(m.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())) };
        if h.is_null() {
            return Err(e.error("speech detector"));
        }
        Ok(Vad { e: e.clone(), h })
    }

    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; samples.len() / Self::FRAME + 2];
        // SAFETY: buffers sized as passed.
        let n = unsafe { (self.e.0.vad_process)(self.h, samples.as_ptr(), samples.len() as c_int, out.as_mut_ptr(), out.len() as c_int) };
        if n < 0 {
            return Err(self.e.error("speech detection"));
        }
        out.truncate(n as usize);
        Ok(out)
    }

    pub fn reset(&mut self) {
        // SAFETY: valid handle.
        unsafe { (self.e.0.vad_reset)(self.h) }
    }
}

impl Drop for Vad {
    fn drop(&mut self) {
        // SAFETY: freed once.
        unsafe { (self.e.0.vad_free)(self.h) }
    }
}

/// The wake word detector.
pub struct Wake {
    e: Engine,
    h: *mut c_void,
}
// SAFETY: as for Vad.
unsafe impl Send for Wake {}

pub struct WakeResult {
    pub scores: Vec<f32>,
    pub detections: usize,
}

impl Wake {
    pub fn new(e: &Engine, model: &Path, threshold: f32, hits: i32, refractory_ms: i32) -> Result<Wake> {
        let m = cpath(model)?;
        // SAFETY: valid path.
        let h = unsafe { (e.0.wake_new)(m.as_ptr(), threshold, hits, refractory_ms) };
        if h.is_null() {
            return Err(e.error("wake word detector"));
        }
        Ok(Wake { e: e.clone(), h })
    }

    pub fn process(&mut self, samples: &[f32]) -> Result<WakeResult> {
        let mut scores = vec![0f32; samples.len() / 1280 + 4];
        let mut n: c_int = 0;
        // SAFETY: buffers sized as passed.
        let d =
            unsafe { (self.e.0.wake_process)(self.h, samples.as_ptr(), samples.len() as c_int, scores.as_mut_ptr(), scores.len() as c_int, &mut n) };
        if d < 0 {
            return Err(self.e.error("wake word detection"));
        }
        scores.truncate(n as usize);
        Ok(WakeResult { scores, detections: d as usize })
    }

    /// Where the last detection was, in ms since the stream started.
    pub fn last_ms(&self) -> f64 {
        // SAFETY: valid handle.
        unsafe { (self.e.0.wake_last_ms)(self.h) }
    }

    pub fn set_threshold(&mut self, t: f32) {
        // SAFETY: valid handle.
        unsafe { (self.e.0.wake_set_threshold)(self.h, t) }
    }

    pub fn reset(&mut self) {
        // SAFETY: valid handle.
        unsafe { (self.e.0.wake_reset)(self.h) }
    }
}

impl Drop for Wake {
    fn drop(&mut self) {
        // SAFETY: freed once.
        unsafe { (self.e.0.wake_free)(self.h) }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Transcript {
    pub text: String,
    pub ms: f64,
}

pub struct Asr {
    e: Engine,
    h: *mut c_void,
}
// SAFETY: as for Vad.
unsafe impl Send for Asr {}

impl Asr {
    pub fn new(e: &Engine, model: &Path, threads: i32, gpu: bool) -> Result<Asr> {
        let m = cpath(model)?;
        // SAFETY: valid path.
        let h = unsafe { (e.0.asr_new)(m.as_ptr(), threads, gpu as c_int) };
        if h.is_null() {
            return Err(e.error("speech recognizer"));
        }
        Ok(Asr { e: e.clone(), h })
    }

    pub fn transcribe(&mut self, samples: &[f32], fast: bool) -> Result<Transcript> {
        // SAFETY: buffer of the given length.
        let p = unsafe { (self.e.0.asr_transcribe)(self.h, samples.as_ptr(), samples.len() as c_int, fast as c_int) };
        let s = self.e.take_string(p, "speech recognition")?;
        Ok(serde_json::from_str(&s)?)
    }
}

impl Drop for Asr {
    fn drop(&mut self) {
        // SAFETY: freed once.
        unsafe { (self.e.0.asr_free)(self.h) }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Generation {
    pub text: String,
    /// "end", "tool", "length", "cancel" or "error".
    pub stop: String,
    pub prompt_tokens: i64,
    pub reused_tokens: i64,
    pub generated: i64,
    pub ms_prompt: f64,
    pub ms_first: f64,
    pub ms: f64,
}

pub struct Llm {
    e: Engine,
    h: *mut c_void,
}
// SAFETY: as for Vad.
unsafe impl Send for Llm {}

struct CbState<'a> {
    f: &'a mut dyn FnMut(&str) -> bool,
}

unsafe extern "C" fn piece_trampoline(piece: *const c_char, user: *mut c_void) -> c_int {
    // SAFETY: `user` is the CbState passed to llm_generate, alive for the call; `piece` is a C string.
    let st = unsafe { &mut *(user as *mut CbState) };
    let s = unsafe { CStr::from_ptr(piece) }.to_string_lossy();
    if (st.f)(&s) { 0 } else { 1 }
}

impl Llm {
    pub fn new(e: &Engine, model: &Path, context: i32, gpu_layers: i32) -> Result<Llm> {
        let m = cpath(model)?;
        // SAFETY: valid path.
        let h = unsafe { (e.0.llm_new)(m.as_ptr(), context, gpu_layers) };
        if h.is_null() {
            return Err(e.error("language model"));
        }
        Ok(Llm { e: e.clone(), h })
    }

    /// Generates from a whole prompt. `on_piece` returns false to stop.
    pub fn generate(
        &mut self,
        prompt: &str,
        trigger: Option<&str>,
        tool_schema: Option<&str>,
        max_tokens: i32,
        on_piece: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Generation> {
        let p = cstr(prompt)?;
        let t = trigger.map(cstr).transpose()?;
        let s = tool_schema.map(cstr).transpose()?;
        let mut st = CbState { f: on_piece };
        // SAFETY: all pointers valid for the call; the callback state outlives it.
        let r = unsafe {
            (self.e.0.llm_generate)(
                self.h,
                p.as_ptr(),
                t.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                s.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                max_tokens,
                Some(piece_trampoline),
                &mut st as *mut CbState as *mut c_void,
            )
        };
        let j = self.e.take_string(r, "generation")?;
        Ok(serde_json::from_str(&j)?)
    }

    pub fn count_tokens(&mut self, text: &str) -> Result<usize> {
        let t = cstr(text)?;
        // SAFETY: valid string.
        let n = unsafe { (self.e.0.llm_count_tokens)(self.h, t.as_ptr()) };
        if n < 0 {
            return Err(self.e.error("tokenizing"));
        }
        Ok(n as usize)
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        // SAFETY: freed once.
        unsafe { (self.e.0.llm_free)(self.h) }
    }
}

pub struct Tts {
    e: Engine,
    h: *mut c_void,
    rate: u32,
    speakers: u32,
}
// SAFETY: as for Vad.
unsafe impl Send for Tts {}

#[derive(Debug, Deserialize)]
pub struct Explained {
    pub text: String,
    pub ipa: String,
}

impl Tts {
    pub fn new(e: &Engine, voice: &Path, cmudict: &Path, threads: i32) -> Result<Tts> {
        let v = cpath(voice)?;
        let c = cpath(&PathBuf::from(format!("{}.json", voice.display())))?;
        let d = cpath(cmudict)?;
        // SAFETY: valid paths.
        let h = unsafe { (e.0.tts_new)(v.as_ptr(), c.as_ptr(), d.as_ptr(), threads) };
        if h.is_null() {
            return Err(e.error("voice"));
        }
        // SAFETY: valid handle.
        let (rate, speakers) = unsafe { ((e.0.tts_rate)(h) as u32, (e.0.tts_speakers)(h) as u32) };
        Ok(Tts { e: e.clone(), h, rate, speakers })
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn speakers(&self) -> u32 {
        self.speakers
    }

    pub fn speak(&mut self, text: &str, speaker: u32, speed: f32, variation: f32) -> Result<Vec<i16>> {
        let t = cstr(text)?;
        let mut pcm: *mut i16 = std::ptr::null_mut();
        // SAFETY: valid string; the engine allocates *pcm.
        let n = unsafe { (self.e.0.tts_speak)(self.h, t.as_ptr(), speaker as c_int, speed, variation, &mut pcm) };
        if n < 0 {
            return Err(self.e.error("speaking"));
        }
        // SAFETY: n samples at pcm, freed after copying.
        let out = unsafe { std::slice::from_raw_parts(pcm, n as usize) }.to_vec();
        unsafe { (self.e.0.free_pcm)(pcm) };
        Ok(out)
    }

    pub fn explain(&mut self, text: &str) -> Result<Explained> {
        let t = cstr(text)?;
        // SAFETY: valid string.
        let p = unsafe { (self.e.0.tts_explain)(self.h, t.as_ptr()) };
        Ok(serde_json::from_str(&self.e.take_string(p, "explaining")?)?)
    }
}

impl Drop for Tts {
    fn drop(&mut self) {
        // SAFETY: freed once.
        unsafe { (self.e.0.tts_free)(self.h) }
    }
}
