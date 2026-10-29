//! The shipped plugins load, and their own tests (the `tests` table in each file) pass.

use courier::tools::{HostOptions, Recorder, ToolHost};
use std::path::PathBuf;
use std::sync::Arc;

#[test]
fn shipped_plugins_load_and_pass_their_tests() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("plugins");
    let data = std::env::temp_dir().join(format!("courier-plugins-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let mut opt = HostOptions::new(dir, Arc::new(Recorder::default()));
    opt.data_dir = Some(data.clone());
    opt.place = "Bucharest".into();
    let host = ToolHost::load(opt).unwrap();
    assert!(host.errors.is_empty(), "{:?}", host.errors);
    let names: Vec<String> = host.tools().into_iter().map(|t| t.name).collect();
    for want in ["set_timer", "add_reminder", "calendar_events", "search_files", "volume", "play_music", "get_weather", "roll_dice"] {
        assert!(names.iter().any(|n| n == want), "{want} missing from {names:?}");
    }
    let results = host.run_tests();
    assert!(results.len() >= 15, "only {} plugin tests", results.len());
    let failed: Vec<_> = results.iter().filter(|r| r.2.is_some()).collect();
    assert!(failed.is_empty(), "{failed:?}");
    let _ = std::fs::remove_dir_all(&data);
}
