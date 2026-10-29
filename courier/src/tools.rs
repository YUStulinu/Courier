//! Tools are Lua plugins: every `*.lua` file in the plugins folder returns a table of tools, each with
//! a name, a description, a JSON schema for its arguments and a `run` function. Adding a command means
//! dropping a file in the folder; the host notices and reloads, no rebuild.
//!
//! ```lua
//! local config = ...           -- this plugin's [plugins.<file name>] table from courier.toml
//! return {
//!   tools = {
//!     { name = "roll_dice", description = "Roll dice.",
//!       parameters = { type = "object", properties = { sides = { type = "integer" } } },
//!       run = function(args) return { result = math.random(args.sides or 6) } end },
//!   },
//!   tests = { rolls_in_range = function(t) ... end },   -- optional: `courier plugins --test`
//! }
//! ```
//!
//! What a plugin may do goes through the `courier` table (alarms, storage, file search, opening
//! things, keys, music, HTTP). A `say` field in a tool's result is spoken as the answer as it is,
//! without asking the model to phrase it, which saves a second generation.

use crate::files;
use crate::prompt::ToolDef;
use anyhow::{Context, Result, anyhow};
use mlua::{Function, HookTriggers, Lua, LuaSerdeExt, Table, Value as LuaValue, VmState};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq)]
pub enum MusicCmd {
    Play(Vec<PathBuf>),
    Pause,
    Resume,
    Stop,
    Next,
    Previous,
    Status,
}

/// What plugins can make happen outside Lua. The assistant implements it for real; tests record calls.
pub trait Services: Send + Sync {
    fn music(&self, cmd: MusicCmd) -> Result<Value>;
    fn key(&self, name: &str) -> Result<()>;
    fn open(&self, target: &str) -> Result<()>;
    fn run(&self, program: &str, args: &[String]) -> Result<(i32, String)>;
    fn http_get(&self, url: &str) -> Result<String>;
    fn lock(&self) -> Result<()>;
}

/// The plugins' clock: real, or set by hand in tests.
#[derive(Clone)]
pub enum Clock {
    Real,
    Fixed(Arc<Mutex<f64>>),
}

impl Clock {
    pub fn now(&self) -> f64 {
        match self {
            Clock::Real => SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0),
            Clock::Fixed(t) => *t.lock().unwrap(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Alarm {
    pub id: String,
    /// Seconds since 1970 (UTC).
    pub at: i64,
    pub text: String,
    pub kind: String,
}

/// Alarms (timers, reminders) are kept by the host, not by Lua code, so they survive a plugin reload
/// and a restart (they are saved to alarms.json).
pub struct Alarms {
    path: Option<PathBuf>,
    list: Vec<Alarm>,
    next: u64,
}

impl Alarms {
    fn open(path: Option<PathBuf>) -> Alarms {
        let list: Vec<Alarm> = path.as_ref().and_then(|p| fs::read_to_string(p).ok()).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        let next = list.iter().filter_map(|a| a.id.rsplit(':').next()?.parse::<u64>().ok()).max().unwrap_or(0) + 1;
        Alarms { path, list, next }
    }

    fn save(&self) {
        if let Some(p) = &self.path
            && let Ok(s) = serde_json::to_string_pretty(&self.list)
        {
            let _ = fs::write(p, s);
        }
    }

    fn add(&mut self, at: i64, text: String, kind: String) -> String {
        let id = format!("{kind}:{}", self.next);
        self.next += 1;
        self.list.push(Alarm { id: id.clone(), at, text, kind });
        self.list.sort_by_key(|a| a.at);
        self.save();
        id
    }

    fn cancel(&mut self, id: &str) -> bool {
        let n = self.list.len();
        self.list.retain(|a| a.id != id);
        self.save();
        self.list.len() != n
    }

    /// Removes and returns the alarms due at `now`.
    pub fn due(&mut self, now: f64) -> Vec<Alarm> {
        let (due, later): (Vec<Alarm>, Vec<Alarm>) = std::mem::take(&mut self.list).into_iter().partition(|a| a.at as f64 <= now);
        self.list = later;
        if !due.is_empty() {
            self.save();
        }
        due
    }
}

struct HostState {
    services: Arc<dyn Services>,
    clock: Clock,
    alarms: Alarms,
    data_dir: Option<PathBuf>,
    deadline: Option<Instant>,
    log: Vec<String>,
}

struct Tool {
    def: ToolDef,
    run: Function,
}

struct PluginFile {
    path: PathBuf,
    modified: Option<SystemTime>,
}

pub struct ToolHost {
    lua: Lua,
    tools: Vec<Tool>,
    tests: Vec<(String, String, Function)>,
    files: Vec<PluginFile>,
    dir: PathBuf,
    /// Plugins that failed to load, with why.
    pub errors: Vec<String>,
    limit: Duration,
}

/// How long one tool call may run its Lua code, unless the options say otherwise.
pub const TOOL_TIME: Duration = Duration::from_secs(10);

pub struct HostOptions {
    pub dir: PathBuf,
    /// Where plugins keep data and alarms are saved; None keeps everything in memory (tests).
    pub data_dir: Option<PathBuf>,
    /// [plugins.<name>] tables from the configuration.
    pub config: BTreeMap<String, Value>,
    pub services: Arc<dyn Services>,
    pub clock: Clock,
    /// Where the user is ([assistant] place), for plugins that need it.
    pub place: String,
    pub time_limit: Duration,
}

impl HostOptions {
    pub fn new(dir: PathBuf, services: Arc<dyn Services>) -> HostOptions {
        HostOptions { dir, data_dir: None, config: BTreeMap::new(), services, clock: Clock::Real, place: String::new(), time_limit: TOOL_TIME }
    }
}

impl ToolHost {
    pub fn load(opt: HostOptions) -> Result<ToolHost> {
        if let Some(d) = &opt.data_dir {
            fs::create_dir_all(d.join("plugins")).with_context(|| format!("creating {}", d.display()))?;
        }
        let lua = Lua::new();
        let alarms = Alarms::open(opt.data_dir.as_ref().map(|d| d.join("alarms.json")));
        lua.set_app_data(HostState {
            services: opt.services,
            clock: opt.clock,
            alarms,
            data_dir: opt.data_dir.clone(),
            deadline: None,
            log: Vec::new(),
        });
        let limit = opt.time_limit;
        lua.set_hook(HookTriggers::new().every_nth_instruction(10_000), |lua, _| {
            let late = lua.app_data_ref::<HostState>().and_then(|s| s.deadline).is_some_and(|d| Instant::now() > d);
            if late {
                return Err(mlua::Error::runtime("the plugin took too long"));
            }
            Ok(VmState::Continue)
        })?;
        install_api(&lua, &opt.data_dir, &opt.place)?;
        // `require` finds shared code in the plugins' lib folder.
        let lib = opt.dir.join("lib").join("?.lua").to_string_lossy().replace('\\', "/");
        lua.load(format!("package.path = {:?} .. ';' .. package.path", lib)).exec()?;
        let mut host = ToolHost { lua, tools: Vec::new(), tests: Vec::new(), files: Vec::new(), dir: opt.dir.clone(), errors: Vec::new(), limit };
        let mut paths: Vec<PathBuf> = fs::read_dir(&opt.dir)
            .with_context(|| format!("reading the plugins folder {}", opt.dir.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "lua"))
            .collect();
        paths.sort();
        for p in paths {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            let cfg = opt.config.get(&name).cloned().unwrap_or(json!({}));
            if let Err(e) = host.load_plugin(&p, &name, &cfg) {
                host.errors.push(format!("{}: {e}", p.display()));
            }
            host.files.push(PluginFile { modified: fs::metadata(&p).and_then(|m| m.modified()).ok(), path: p });
        }
        Ok(host)
    }

    fn load_plugin(&mut self, path: &Path, name: &str, cfg: &Value) -> Result<()> {
        let code = fs::read_to_string(path)?;
        let cfg = self.lua.to_value(cfg)?;
        self.set_deadline(Some(self.limit));
        let r = self.lua.load(&code).set_name(format!("@{}", path.display())).call::<Table>(cfg);
        self.set_deadline(None);
        let t = r.map_err(|e| anyhow!("{}", short_error(&e.to_string())))?;
        let tools: Table = t.get("tools").map_err(|_| anyhow!("the plugin returns no `tools` table"))?;
        let mut defs = Vec::new();
        for item in tools.sequence_values::<Table>() {
            let item = item?;
            let tname: String = item.get("name").map_err(|_| anyhow!("a tool without a name"))?;
            if !tname.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || tname.is_empty() {
                return Err(anyhow!("tool name {tname:?}: use letters, digits and _"));
            }
            if self.tools.iter().any(|x| x.def.name == tname) || defs.iter().any(|x: &Tool| x.def.name == tname) {
                return Err(anyhow!("tool {tname} is defined twice"));
            }
            let description: String = item.get("description").map_err(|_| anyhow!("{tname} has no description"))?;
            let run: Function = item.get("run").map_err(|_| anyhow!("{tname} has no run function"))?;
            let params: LuaValue = item.get("parameters")?;
            let params: Value = if params.is_nil() { json!({}) } else { self.lua.from_value(params)? };
            defs.push(Tool { def: ToolDef { name: tname, description, parameters: normalize_schema(params) }, run });
        }
        if defs.is_empty() {
            return Err(anyhow!("the plugin defines no tools"));
        }
        self.tools.extend(defs);
        if let Ok(tests) = t.get::<Table>("tests") {
            for pair in tests.pairs::<String, Function>() {
                let (test, f) = pair?;
                self.tests.push((name.to_string(), test, f));
            }
        }
        Ok(())
    }

    fn set_deadline(&self, d: Option<Duration>) {
        if let Some(mut s) = self.lua.app_data_mut::<HostState>() {
            s.deadline = d.map(|d| Instant::now() + d);
        }
    }

    pub fn tools(&self) -> Vec<ToolDef> {
        self.tools.iter().map(|t| t.def.clone()).collect()
    }

    /// Runs a tool. Problems (unknown tool, bad arguments, a Lua error) come back as {"error": ...},
    /// which the model reads like any other result.
    pub fn call(&self, name: &str, args: &Value) -> Value {
        let Some(tool) = self.tools.iter().find(|t| t.def.name == name) else {
            return json!({"error": format!("there is no tool called {name}")});
        };
        if let Err(e) = validate(args, &tool.def.parameters) {
            return json!({"error": e});
        }
        let lua_args =
            match self.lua.to_value_with(args, mlua::serde::SerializeOptions::new().serialize_none_to_null(false).serialize_unit_to_null(false)) {
                Ok(v) => v,
                Err(e) => return json!({"error": e.to_string()}),
            };
        self.set_deadline(Some(self.limit));
        let r = tool.run.call::<LuaValue>(lua_args);
        self.set_deadline(None);
        match r {
            Ok(v) => match self.lua.from_value::<Value>(v) {
                Ok(Value::Null) => json!({"ok": true}),
                Ok(Value::String(s)) => json!({"say": s}),
                Ok(v) => v,
                Err(e) => json!({"error": format!("the tool returned something that is not data: {e}")}),
            },
            Err(e) => json!({"error": short_error(&e.to_string())}),
        }
    }

    /// Whether a plugin file was added, removed or changed since loading.
    pub fn changed(&self) -> bool {
        let Ok(rd) = fs::read_dir(&self.dir) else { return false };
        let now: Vec<(PathBuf, Option<SystemTime>)> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "lua"))
            .map(|p| (p.clone(), fs::metadata(&p).and_then(|m| m.modified()).ok()))
            .collect();
        now.len() != self.files.len() || now.iter().any(|(p, m)| !self.files.iter().any(|f| &f.path == p && &f.modified == m))
    }

    pub fn due_alarms(&self) -> Vec<Alarm> {
        let Some(mut s) = self.lua.app_data_mut::<HostState>() else { return Vec::new() };
        let now = s.clock.now();
        s.alarms.due(now)
    }

    pub fn take_log(&self) -> Vec<String> {
        self.lua.app_data_mut::<HostState>().map(|mut s| std::mem::take(&mut s.log)).unwrap_or_default()
    }

    /// Runs the plugins' own tests: (plugin, test, error if it failed).
    pub fn run_tests(&self) -> Vec<(String, String, Option<String>)> {
        let mut out = Vec::new();
        for (plugin, name, f) in &self.tests {
            let t = self.test_helpers().expect("test helpers");
            self.set_deadline(Some(self.limit));
            let r = f.call::<()>(t);
            self.set_deadline(None);
            out.push((plugin.clone(), name.clone(), r.err().map(|e| short_error(&e.to_string()))));
        }
        out
    }

    /// `t` in a plugin test: t.call(tool, args) runs a tool as the assistant would (validation
    /// included); t.eq(a, b, what) and t.ok(cond, what) check.
    fn test_helpers(&self) -> mlua::Result<Table> {
        let lua = &self.lua;
        let t = lua.create_table()?;
        let tools: Vec<(String, Value, Function)> =
            self.tools.iter().map(|x| (x.def.name.clone(), x.def.parameters.clone(), x.run.clone())).collect();
        t.set(
            "call",
            lua.create_function(move |lua, (name, args): (String, LuaValue)| {
                let (_, params, run) = tools.iter().find(|x| x.0 == name).ok_or_else(|| mlua::Error::runtime(format!("no tool {name}")))?;
                let a: Value = if args.is_nil() { json!({}) } else { lua.from_value(args.clone())? };
                if let Err(e) = validate(&a, params) {
                    return lua.to_value(&json!({"error": e}));
                }
                run.call::<LuaValue>(args)
            })?,
        )?;
        t.set(
            "eq",
            lua.create_function(|lua, (a, b, what): (LuaValue, LuaValue, Option<String>)| {
                let (x, y): (Value, Value) = (lua.from_value(a)?, lua.from_value(b)?);
                if whole_numbers(x.clone()) != whole_numbers(y.clone()) {
                    return Err(mlua::Error::runtime(format!("{}: got {x}, expected {y}", what.unwrap_or_default())));
                }
                Ok(())
            })?,
        )?;
        t.set(
            "ok",
            lua.create_function(|_, (cond, what): (bool, Option<String>)| {
                if !cond {
                    return Err(mlua::Error::runtime(format!("{} is false", what.unwrap_or_else(|| "condition".into()))));
                }
                Ok(())
            })?,
        )?;
        Ok(t)
    }
}

/// 900.0 and 900 are the same number to Lua; make them the same to JSON's equality too.
fn whole_numbers(v: Value) -> Value {
    match v {
        Value::Number(n) => match n.as_f64() {
            Some(f) if f.fract() == 0.0 && f.abs() < 9e15 => json!(f as i64),
            _ => Value::Number(n),
        },
        Value::Array(a) => Value::Array(a.into_iter().map(whole_numbers).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, whole_numbers(v))).collect()),
        other => other,
    }
}

/// The first line of a Lua error, without the traceback.
fn short_error(e: &str) -> String {
    e.lines().next().unwrap_or(e).replace("runtime error: ", "").to_string()
}

/// Makes a plugin's schema a well-formed object schema: an empty Lua table becomes `{}` whatever it
/// was meant to be, so `required = {}` must be turned back into a list.
fn normalize_schema(mut s: Value) -> Value {
    if !s.is_object() {
        s = json!({});
    }
    let o = s.as_object_mut().unwrap();
    o.entry("type").or_insert(json!("object"));
    let props = o.entry("properties").or_insert(json!({}));
    if props.as_array().is_some_and(|a| a.is_empty()) {
        *props = json!({});
    }
    match o.get("required") {
        Some(Value::Array(_)) => {}
        Some(_) | None => {
            o.insert("required".into(), json!([]));
        }
    }
    s
}

/// Checks arguments against the parts of JSON schema plugins use: required, type, enum, minimum,
/// maximum, unknown properties.
pub fn validate(args: &Value, schema: &Value) -> Result<(), String> {
    let Some(obj) = args.as_object() else { return Err("the arguments must be an object".into()) };
    let props = schema.get("properties").and_then(|p| p.as_object());
    if let Some(req) = schema.get("required").and_then(|r| r.as_array()) {
        for r in req.iter().filter_map(|r| r.as_str()) {
            if !obj.contains_key(r) {
                return Err(format!("missing argument {r}"));
            }
        }
    }
    for (k, v) in obj {
        let Some(p) = props.and_then(|p| p.get(k)) else {
            return Err(format!("unknown argument {k}"));
        };
        if let Some(t) = p.get("type").and_then(|t| t.as_str()) {
            let ok = match t {
                "string" => v.is_string(),
                "number" => v.is_number(),
                "integer" => v.as_f64().is_some_and(|x| x.fract() == 0.0),
                "boolean" => v.is_boolean(),
                "array" => v.is_array(),
                "object" => v.is_object(),
                _ => true,
            };
            if !ok {
                return Err(format!("{k} must be a {t}"));
            }
        }
        if let Some(e) = p.get("enum").and_then(|e| e.as_array())
            && !e.contains(v)
        {
            return Err(format!("{k} must be one of {}", Value::Array(e.clone())));
        }
        if let (Some(min), Some(x)) = (p.get("minimum").and_then(|m| m.as_f64()), v.as_f64())
            && x < min
        {
            return Err(format!("{k} must be at least {min}"));
        }
        if let (Some(max), Some(x)) = (p.get("maximum").and_then(|m| m.as_f64()), v.as_f64())
            && x > max
        {
            return Err(format!("{k} must be at most {max}"));
        }
    }
    Ok(())
}

fn state(lua: &Lua) -> mlua::Result<mlua::AppDataRefMut<'_, HostState>> {
    lua.app_data_mut::<HostState>().ok_or_else(|| mlua::Error::runtime("no host state"))
}

fn services(lua: &Lua) -> mlua::Result<Arc<dyn Services>> {
    Ok(state(lua)?.services.clone())
}

fn lua_err(e: anyhow::Error) -> mlua::Error {
    mlua::Error::runtime(e.to_string())
}

fn safe_key(key: &str) -> mlua::Result<&str> {
    if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err(mlua::Error::runtime(format!("storage key {key:?}: use letters, digits, _ and -")));
    }
    Ok(key)
}

fn install_api(lua: &Lua, data_dir: &Option<PathBuf>, place: &str) -> mlua::Result<()> {
    let c = lua.create_table()?;
    c.set("place", place)?;
    c.set(
        "os",
        if cfg!(windows) {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else {
            "linux"
        },
    )?;
    c.set("home", dirs::home_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default())?;
    if let Some(d) = data_dir {
        c.set("data_dir", d.to_string_lossy().into_owned())?;
    }
    for (name, dir) in [
        ("music_dir", dirs::audio_dir()),
        ("documents_dir", dirs::document_dir()),
        ("desktop_dir", dirs::desktop_dir()),
        ("downloads_dir", dirs::download_dir()),
        ("pictures_dir", dirs::picture_dir()),
    ] {
        if let Some(d) = dir {
            c.set(name, d.to_string_lossy().into_owned())?;
        }
    }

    // Whole seconds: Lua's os.date takes integers.
    c.set("now", lua.create_function(|lua, ()| Ok(state(lua)?.clock.now().floor() as i64))?)?;
    c.set(
        "log",
        lua.create_function(|lua, msg: String| {
            state(lua)?.log.push(msg);
            Ok(())
        })?,
    )?;

    // Alarms: courier.alarm{at = epoch, text = "...", kind = "timer"} -> id
    c.set(
        "alarm",
        lua.create_function(|lua, t: Table| {
            let at: f64 = t.get("at")?;
            let at = at.round() as i64;
            let text: String = t.get("text")?;
            let kind: String = t.get::<Option<String>>("kind")?.unwrap_or_else(|| "alarm".into());
            Ok(state(lua)?.alarms.add(at, text, kind))
        })?,
    )?;
    c.set(
        "alarms",
        lua.create_function(|lua, kind: Option<String>| {
            let list: Vec<Alarm> = state(lua)?.alarms.list.iter().filter(|a| kind.as_ref().is_none_or(|k| &a.kind == k)).cloned().collect();
            lua.to_value(&list)
        })?,
    )?;
    c.set("cancel_alarm", lua.create_function(|lua, id: String| Ok(state(lua)?.alarms.cancel(&id)))?)?;

    // Storage: one JSON file per key.
    c.set(
        "load",
        lua.create_function(|lua, key: String| {
            let key = safe_key(&key)?.to_string();
            let Some(dir) = state(lua)?.data_dir.clone() else { return Ok(LuaValue::Nil) };
            match fs::read_to_string(dir.join("plugins").join(format!("{key}.json"))) {
                Ok(s) => {
                    let v: Value = serde_json::from_str(&s).map_err(mlua::Error::external)?;
                    lua.to_value_with(&v, mlua::serde::SerializeOptions::new().serialize_none_to_null(false))
                }
                Err(_) => Ok(LuaValue::Nil),
            }
        })?,
    )?;
    c.set(
        "save",
        lua.create_function(|lua, (key, v): (String, LuaValue)| {
            let key = safe_key(&key)?.to_string();
            let v: Value = lua.from_value(v)?;
            if let Some(dir) = state(lua)?.data_dir.clone() {
                fs::write(dir.join("plugins").join(format!("{key}.json")), serde_json::to_string_pretty(&v).unwrap())
                    .map_err(mlua::Error::external)?;
            }
            Ok(())
        })?,
    )?;

    // Files.
    c.set(
        "find_files",
        lua.create_function(|lua, t: Table| {
            let roots: Vec<String> = t.get::<Option<Vec<String>>>("roots")?.unwrap_or_default();
            let query: String = t.get("query")?;
            let exts: Vec<String> = t.get::<Option<Vec<String>>>("exts")?.unwrap_or_default();
            let q = files::Query {
                roots: roots.into_iter().map(PathBuf::from).collect(),
                terms: files::terms(&query),
                exts: exts.into_iter().map(|e| e.trim_start_matches('.').to_lowercase()).collect(),
                limit: t.get::<Option<usize>>("limit")?.unwrap_or(5),
                content: t.get::<Option<bool>>("content")?.unwrap_or(true),
                budget: Duration::from_secs_f64(t.get::<Option<f64>>("seconds")?.unwrap_or(4.0)),
            };
            lua.to_value(&files::search(&q))
        })?,
    )?;
    c.set(
        "list_files",
        lua.create_function(|lua, (dir, exts, max): (String, Option<Vec<String>>, Option<usize>)| {
            let exts: Vec<String> = exts.unwrap_or_default().into_iter().map(|e| e.trim_start_matches('.').to_lowercase()).collect();
            let list: Vec<String> =
                files::list(Path::new(&dir), &exts, max.unwrap_or(20_000)).into_iter().map(|p| p.to_string_lossy().into_owned()).collect();
            lua.to_value(&list)
        })?,
    )?;

    // The computer.
    c.set("open", lua.create_function(|lua, target: String| services(lua)?.open(&target).map_err(lua_err))?)?;
    c.set("key", lua.create_function(|lua, name: String| services(lua)?.key(&name).map_err(lua_err))?)?;
    c.set("lock", lua.create_function(|lua, ()| services(lua)?.lock().map_err(lua_err))?)?;
    c.set(
        "run",
        lua.create_function(|lua, (program, args): (String, Option<Vec<String>>)| {
            services(lua)?.run(&program, &args.unwrap_or_default()).map_err(lua_err)
        })?,
    )?;
    c.set(
        "http_get",
        lua.create_function(|lua, url: String| match services(lua)?.http_get(&url) {
            Ok(body) => Ok((Some(body), None)),
            Err(e) => Ok((None, Some(e.to_string()))),
        })?,
    )?;
    c.set(
        "music",
        lua.create_function(|lua, (cmd, arg): (String, LuaValue)| {
            let cmd = match cmd.as_str() {
                "play" => MusicCmd::Play(lua.from_value::<Vec<String>>(arg)?.into_iter().map(PathBuf::from).collect()),
                "pause" => MusicCmd::Pause,
                "resume" => MusicCmd::Resume,
                "stop" => MusicCmd::Stop,
                "next" => MusicCmd::Next,
                "previous" => MusicCmd::Previous,
                "status" => MusicCmd::Status,
                other => return Err(mlua::Error::runtime(format!("unknown music command {other}"))),
            };
            let v = services(lua)?.music(cmd).map_err(lua_err)?;
            lua.to_value_with(&v, mlua::serde::SerializeOptions::new().serialize_none_to_null(false))
        })?,
    )?;

    // JSON.
    c.set(
        "json_decode",
        lua.create_function(|lua, s: String| {
            let v: Value = serde_json::from_str(&s).map_err(mlua::Error::external)?;
            lua.to_value_with(&v, mlua::serde::SerializeOptions::new().serialize_none_to_null(false))
        })?,
    )?;
    c.set(
        "json_encode",
        lua.create_function(|lua, v: LuaValue| {
            let v: Value = lua.from_value(v)?;
            Ok(v.to_string())
        })?,
    )?;
    lua.globals().set("courier", c)?;
    Ok(())
}

/// Services that do nothing but remember what they were asked: for tests and `courier plugins`.
#[derive(Default)]
pub struct Recorder {
    pub calls: Mutex<Vec<String>>,
    /// Answers for http_get, by URL prefix.
    pub http: Mutex<Vec<(String, String)>>,
}

impl Services for Recorder {
    fn music(&self, cmd: MusicCmd) -> Result<Value> {
        self.calls.lock().unwrap().push(format!("music {cmd:?}"));
        Ok(json!({"playing": matches!(cmd, MusicCmd::Play(_) | MusicCmd::Resume | MusicCmd::Next | MusicCmd::Previous)}))
    }
    fn key(&self, name: &str) -> Result<()> {
        self.calls.lock().unwrap().push(format!("key {name}"));
        Ok(())
    }
    fn open(&self, target: &str) -> Result<()> {
        self.calls.lock().unwrap().push(format!("open {target}"));
        Ok(())
    }
    fn run(&self, program: &str, args: &[String]) -> Result<(i32, String)> {
        self.calls.lock().unwrap().push(format!("run {program} {}", args.join(" ")));
        Ok((0, String::new()))
    }
    fn http_get(&self, url: &str) -> Result<String> {
        self.calls.lock().unwrap().push(format!("get {url}"));
        let answers = self.http.lock().unwrap();
        answers.iter().find(|(p, _)| url.starts_with(p.as_str())).map(|(_, a)| a.clone()).ok_or_else(|| anyhow!("no network in tests"))
    }
    fn lock(&self) -> Result<()> {
        self.calls.lock().unwrap().push("lock".into());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(dir: &Path, clock: Clock) -> ToolHost {
        let mut o = HostOptions::new(dir.to_path_buf(), Arc::new(Recorder::default()));
        o.clock = clock;
        o.time_limit = Duration::from_secs(1);
        ToolHost::load(o).unwrap()
    }

    fn plugin_dir(name: &str, files: &[(&str, &str)]) -> PathBuf {
        let d = std::env::temp_dir().join(format!("courier-tools-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        for (f, code) in files {
            fs::write(d.join(f), code).unwrap();
        }
        d
    }

    const DICE: &str = r#"
        local config = ...
        return { tools = {
          { name = "roll", description = "Roll a die.",
            parameters = { type = "object", properties = { sides = { type = "integer", minimum = 2 } }, required = {} },
            run = function(a) return { result = (a.sides or config.sides or 6), say = "Rolled." } end },
          { name = "spin", description = "Loops forever.", run = function() while true do end end },
          { name = "remind", description = "Sets an alarm.",
            parameters = { type = "object", properties = { text = { type = "string" } }, required = { "text" } },
            run = function(a) return { id = courier.alarm{ at = courier.now() + 60, text = a.text, kind = "reminder" } } end },
        } }"#;

    #[test]
    fn plugins_define_tools_with_schemas() {
        let d = plugin_dir("defs", &[("dice.lua", DICE)]);
        let h = host(&d, Clock::Real);
        assert!(h.errors.is_empty(), "{:?}", h.errors);
        let t = h.tools();
        assert_eq!(t.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(), vec!["roll", "spin", "remind"]);
        assert_eq!(t[0].parameters["required"], json!([]));
        assert_eq!(t[1].parameters, json!({"type": "object", "properties": {}, "required": []}));
    }

    #[test]
    fn calls_validate_and_return_data() {
        let d = plugin_dir("calls", &[("dice.lua", DICE)]);
        let h = host(&d, Clock::Real);
        assert_eq!(h.call("roll", &json!({"sides": 20})), json!({"result": 20, "say": "Rolled."}));
        assert_eq!(h.call("roll", &json!({})), json!({"result": 6, "say": "Rolled."}));
        assert_eq!(h.call("roll", &json!({"sides": 1}))["error"], "sides must be at least 2");
        assert_eq!(h.call("roll", &json!({"faces": 3}))["error"], "unknown argument faces");
        assert_eq!(h.call("remind", &json!({}))["error"], "missing argument text");
        assert!(h.call("nope", &json!({}))["error"].as_str().unwrap().contains("no tool"));
    }

    #[test]
    fn a_runaway_plugin_is_stopped() {
        let d = plugin_dir("spin", &[("dice.lua", DICE)]);
        let h = host(&d, Clock::Real);
        let t = Instant::now();
        let r = h.call("spin", &json!({}));
        assert!(r["error"].as_str().unwrap().contains("took too long"), "{r}");
        assert!(t.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn alarms_fire_when_due() {
        let d = plugin_dir("alarms", &[("dice.lua", DICE)]);
        let now = Arc::new(Mutex::new(1000.0));
        let h = host(&d, Clock::Fixed(now.clone()));
        let id = h.call("remind", &json!({"text": "stretch"}))["id"].as_str().unwrap().to_string();
        assert_eq!(id, "reminder:1");
        assert!(h.due_alarms().is_empty());
        *now.lock().unwrap() = 1061.0;
        let due = h.due_alarms();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].text, "stretch");
        assert!(h.due_alarms().is_empty());
    }

    #[test]
    fn broken_plugins_are_reported_and_the_rest_load() {
        let d = plugin_dir("broken", &[("a.lua", "return {"), ("b.lua", DICE), ("c.lua", "return { tools = {} }")]);
        let h = host(&d, Clock::Real);
        assert_eq!(h.errors.len(), 2, "{:?}", h.errors);
        assert_eq!(h.tools().len(), 3);
    }

    #[test]
    fn changes_are_noticed() {
        let d = plugin_dir("watch", &[("dice.lua", DICE)]);
        let h = host(&d, Clock::Real);
        assert!(!h.changed());
        fs::write(d.join("more.lua"), "return { tools = {} }").unwrap();
        assert!(h.changed());
    }

    #[test]
    fn validation() {
        let s = json!({"type": "object", "properties": {"n": {"type": "integer"}, "k": {"type": "string", "enum": ["a", "b"]}}, "required": ["n"]});
        assert!(validate(&json!({"n": 3}), &s).is_ok());
        assert!(validate(&json!({"n": 3.5}), &s).is_err());
        assert!(validate(&json!({"n": 3, "k": "c"}), &s).is_err());
        assert!(validate(&json!([]), &s).is_err());
    }
}
