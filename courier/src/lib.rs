//! Courier: an offline voice assistant. This crate is the orchestrator: audio in and out, the
//! conversation's state machine, endpointing, prompts and tool calls, and the Lua plugin host. The
//! models run in the C++ engine (engine/), loaded at run time.

pub mod assistant;
pub mod audio;
pub mod chat;
pub mod config;
pub mod dialog;
pub mod endpoint;
pub mod engine;
pub mod eval;
pub mod files;
pub mod ground;
pub mod prompt;
pub mod resample;
pub mod services;
pub mod speech;
pub mod tools;
pub mod wav;
