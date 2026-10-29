//! What plugins can do to the computer, for real: media and volume keys, opening files and programs,
//! locking the screen, running a program, fetching a web page (the weather plugin; off when offline).

use crate::audio::Out;
use crate::tools::{MusicCmd, Services};
use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub struct System {
    pub out: Option<Out>,
    pub network: bool,
}

impl Services for System {
    fn music(&self, cmd: MusicCmd) -> Result<Value> {
        match &self.out {
            Some(o) => o.music(cmd),
            None => Ok(json!({"error": "no audio output"})),
        }
    }

    fn key(&self, name: &str) -> Result<()> {
        press(name)
    }

    fn open(&self, target: &str) -> Result<()> {
        #[cfg(windows)]
        let mut c = {
            let mut c = Command::new("cmd");
            // `start` with an empty title, so a quoted path is not taken for one.
            c.args(["/C", "start", "", target]);
            c
        };
        #[cfg(not(windows))]
        let mut c = {
            let mut c = Command::new(if cfg!(target_os = "macos") { "open" } else { "xdg-open" });
            c.arg(target);
            c
        };
        c.stdout(Stdio::null()).stderr(Stdio::null()).spawn()?;
        Ok(())
    }

    fn run(&self, program: &str, args: &[String]) -> Result<(i32, String)> {
        let mut child = Command::new(program).args(args).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
        let start = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                let mut out = String::new();
                if let Some(mut o) = child.stdout.take() {
                    use std::io::Read;
                    let _ = o.read_to_string(&mut out);
                }
                return Ok((status.code().unwrap_or(-1), out));
            }
            if start.elapsed() > Duration::from_secs(10) {
                let _ = child.kill();
                bail!("{program} did not finish in 10 s");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn http_get(&self, url: &str) -> Result<String> {
        if !self.network {
            bail!("the network is off (network = false in courier.toml)");
        }
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(6))).build().into();
        let mut r = agent.get(url).header("User-Agent", "Courier voice assistant").call()?;
        Ok(r.body_mut().read_to_string()?)
    }

    fn lock(&self) -> Result<()> {
        #[cfg(windows)]
        {
            // SAFETY: no arguments; returns 0 on failure.
            let ok = unsafe { windows_sys::Win32::System::Shutdown::LockWorkStation() };
            if ok == 0 {
                bail!("the screen could not be locked");
            }
            Ok(())
        }
        #[cfg(not(windows))]
        {
            let st = Command::new("loginctl").arg("lock-session").status()?;
            if !st.success() {
                bail!("loginctl lock-session failed");
            }
            Ok(())
        }
    }
}

/// Presses a media key: volume_up, volume_down, mute, play_pause, next, previous, stop.
#[cfg(windows)]
fn press(name: &str) -> Result<()> {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::*;
    let vk: u16 = match name {
        "volume_up" => VK_VOLUME_UP,
        "volume_down" => VK_VOLUME_DOWN,
        "mute" => VK_VOLUME_MUTE,
        "play_pause" => VK_MEDIA_PLAY_PAUSE,
        "next" => VK_MEDIA_NEXT_TRACK,
        "previous" => VK_MEDIA_PREV_TRACK,
        "stop" => VK_MEDIA_STOP,
        other => return Err(anyhow!("unknown key {other}")),
    };
    let key = |flags: u32| INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 { ki: KEYBDINPUT { wVk: vk, wScan: 0, dwFlags: flags | KEYEVENTF_EXTENDEDKEY, time: 0, dwExtraInfo: 0 } },
    };
    let inputs = [key(0), key(KEYEVENTF_KEYUP)];
    // SAFETY: two well-formed keyboard inputs.
    let n = unsafe { SendInput(2, inputs.as_ptr(), std::mem::size_of::<INPUT>() as i32) };
    if n != 2 {
        bail!("the key could not be pressed");
    }
    Ok(())
}

#[cfg(not(windows))]
fn press(name: &str) -> Result<()> {
    let (prog, args): (&str, Vec<&str>) = match name {
        "volume_up" => ("amixer", vec!["-q", "set", "Master", "5%+"]),
        "volume_down" => ("amixer", vec!["-q", "set", "Master", "5%-"]),
        "mute" => ("amixer", vec!["-q", "set", "Master", "toggle"]),
        "play_pause" => ("playerctl", vec!["play-pause"]),
        "next" => ("playerctl", vec!["next"]),
        "previous" => ("playerctl", vec!["previous"]),
        "stop" => ("playerctl", vec!["stop"]),
        other => return Err(anyhow!("unknown key {other}")),
    };
    let st = Command::new(prog).args(&args).status()?;
    if !st.success() {
        bail!("{prog} failed");
    }
    Ok(())
}
