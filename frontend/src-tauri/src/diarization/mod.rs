//! Speaker identification ("who spoke when") for recorded meetings, built on the
//! `meetily-diarization` engine crate.

pub mod commands;
pub mod job;

#[cfg(test)]
mod e2e_tests;

use std::path::PathBuf;
use tauri::{AppHandle, Manager, Runtime};

/// `<app data>/models/diarization`, next to the Parakeet and summary models.
pub fn models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map(|dir| dir.join("models").join("diarization"))
        .map_err(|e| format!("Failed to resolve app data directory: {e}"))
}
