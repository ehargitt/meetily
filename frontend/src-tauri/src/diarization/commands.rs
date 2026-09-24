//! Tauri commands for speaker identification, per-meeting speaker editing, the self
//! voiceprint and the diarization model download.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use log::{info, warn};
use meetily_diarization::models::{self, DownloadProgress};
use meetily_diarization::{DiarizationConfig, DiarizationError};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Runtime};

use super::job::{self, JobState, StartSpeakerIdResult, Trigger};
use crate::audio::common::find_audio_file;
use crate::database::repositories::meeting::MeetingsRepository;
use crate::database::repositories::speaker::{MeetingSpeaker, SpeakerRepository};
use crate::state::AppState;

static DOWNLOAD_IN_PROGRESS: AtomicBool = AtomicBool::new(false);
static DOWNLOAD_CANCELLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SpeakerIdOptions {
    /// Force this many speakers instead of detecting the count.
    pub num_speakers: Option<u32>,
    /// `"fast"` (default) or `"accurate"`.
    pub quality: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeakerIdStatus {
    pub meeting_id: String,
    #[serde(flatten)]
    pub state: JobState,
    pub audio_available: bool,
    pub models_installed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiarizationModelsStatus {
    pub installed: bool,
    /// File names of models not yet downloaded.
    pub missing: Vec<String>,
    /// Download size of all models.
    pub total_bytes: u64,
    pub models_dir: String,
}

fn parse_trigger(trigger: &str) -> Result<Trigger, String> {
    match trigger {
        "auto" => Ok(Trigger::Auto),
        "manual" => Ok(Trigger::Manual),
        other => Err(format!(
            "Unknown trigger '{other}'; expected 'auto' or 'manual'"
        )),
    }
}

fn diarization_config(options: Option<SpeakerIdOptions>) -> Result<DiarizationConfig, String> {
    let options = options.unwrap_or_default();
    let mut config = match options.quality.as_deref() {
        None | Some("fast") => DiarizationConfig::fast(),
        Some("accurate") => DiarizationConfig::accurate(),
        Some(other) => {
            return Err(format!(
                "Unknown quality '{other}'; expected 'fast' or 'accurate'"
            ))
        }
    };
    config.num_speakers = match options.num_speakers {
        Some(0) => return Err("num_speakers must be at least 1".to_string()),
        count => count.map(|n| n as usize),
    };
    Ok(config)
}

/// Start identifying the speakers of a meeting. `trigger: "auto"` (after recording) is
/// skipped quietly when the setting is off; both triggers are skipped when the meeting has
/// no audio, the models are missing, retranscription is running, or a job already exists.
#[tauri::command]
pub async fn start_speaker_identification<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    trigger: String,
    options: Option<SpeakerIdOptions>,
) -> Result<StartSpeakerIdResult, String> {
    let trigger = parse_trigger(&trigger)?;
    let config = diarization_config(options)?;
    job::start(&app, &meeting_id, trigger, config).await
}

#[tauri::command]
pub async fn cancel_speaker_identification(meeting_id: String) -> Result<bool, String> {
    Ok(job::cancel(&meeting_id))
}

#[tauri::command]
pub async fn get_speaker_identification_status<R: Runtime>(
    app: AppHandle<R>,
    state: tauri::State<'_, AppState>,
    meeting_id: String,
) -> Result<SpeakerIdStatus, String> {
    let pool = state.db_manager.pool();
    let stored = SpeakerRepository::get_job(pool, &meeting_id)
        .await
        .map_err(|e| e.to_string())?;
    let folder = MeetingsRepository::get_meeting_metadata(pool, &meeting_id)
        .await
        .map_err(|e| e.to_string())?
        .and_then(|meeting| meeting.folder_path)
        .map(PathBuf::from);
    let audio_available = folder.is_some_and(|f| find_audio_file(&f).is_ok());
    let models_installed = models::missing(&super::models_dir(&app)?).is_empty();

    Ok(SpeakerIdStatus {
        state: job::job_state(&meeting_id, stored),
        meeting_id,
        audio_available,
        models_installed,
    })
}

#[tauri::command]
pub async fn api_get_meeting_speakers(
    state: tauri::State<'_, AppState>,
    meeting_id: String,
) -> Result<Vec<MeetingSpeaker>, String> {
    SpeakerRepository::get_meeting_speakers(state.db_manager.pool(), &meeting_id)
        .await
        .map_err(|e| e.to_string())
}

/// `display_name`: `None` leaves it unchanged, `""` resets to the default label.
/// `is_self: true` marks the speaker as the local user (and enrolls the voiceprint).
#[tauri::command]
pub async fn api_update_meeting_speaker(
    state: tauri::State<'_, AppState>,
    meeting_id: String,
    speaker_key: String,
    display_name: Option<String>,
    is_self: Option<bool>,
) -> Result<MeetingSpeaker, String> {
    SpeakerRepository::update_meeting_speaker(
        state.db_manager.pool(),
        &meeting_id,
        &speaker_key,
        display_name.as_deref(),
        is_self,
    )
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn api_merge_meeting_speakers(
    state: tauri::State<'_, AppState>,
    meeting_id: String,
    from_key: String,
    into_key: String,
) -> Result<(), String> {
    SpeakerRepository::merge_meeting_speakers(
        state.db_manager.pool(),
        &meeting_id,
        &from_key,
        &into_key,
    )
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn forget_self_voiceprint(state: tauri::State<'_, AppState>) -> Result<(), String> {
    SpeakerRepository::delete_self_voiceprint(state.db_manager.pool())
        .await
        .map_err(|e| e.to_string())?;
    info!("Self voiceprint deleted");
    Ok(())
}

#[tauri::command]
pub async fn get_diarization_models_status<R: Runtime>(
    app: AppHandle<R>,
) -> Result<DiarizationModelsStatus, String> {
    let models_dir = super::models_dir(&app)?;
    let missing: Vec<String> = models::missing(&models_dir)
        .iter()
        .map(|spec| spec.file_name.to_string())
        .collect();
    Ok(DiarizationModelsStatus {
        installed: missing.is_empty(),
        missing,
        total_bytes: models::total_bytes(),
        models_dir: models_dir.display().to_string(),
    })
}

/// Clears [`DOWNLOAD_IN_PROGRESS`] however the download ends.
struct DownloadGuard;

impl DownloadGuard {
    fn acquire() -> Result<Self, String> {
        DOWNLOAD_IN_PROGRESS
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map(|_| DownloadGuard)
            .map_err(|_| "Speaker model download already in progress".to_string())
    }
}

impl Drop for DownloadGuard {
    fn drop(&mut self) {
        DOWNLOAD_IN_PROGRESS.store(false, Ordering::SeqCst);
    }
}

#[derive(Debug, Clone, Serialize)]
struct DownloadProgressEvent {
    file: &'static str,
    downloaded_bytes: u64,
    total_bytes: u64,
}

/// Download the missing diarization models (sha256-verified). Emits
/// `diarization-models-download-progress` `{file, downloaded_bytes, total_bytes}`, then
/// `-complete` `{models_dir}` or `-error` `{error, cancelled}`.
#[tauri::command]
pub async fn download_diarization_models<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    let _guard = DownloadGuard::acquire()?;
    DOWNLOAD_CANCELLED.store(false, Ordering::SeqCst);
    let models_dir = super::models_dir(&app)?;
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Failed to create {}: {e}", models_dir.display()))?;

    // One event per whole percent of each file; the stream reports every network chunk.
    let last_reported: Mutex<Option<(&'static str, u64)>> = Mutex::new(None);
    let progress_app = app.clone();
    let on_progress = move |progress: DownloadProgress| {
        let percent = progress.downloaded_bytes * 100 / progress.total_bytes.max(1);
        {
            let mut last = last_reported.lock().unwrap_or_else(|e| e.into_inner());
            if *last == Some((progress.file_name, percent)) {
                return;
            }
            *last = Some((progress.file_name, percent));
        }
        let event = DownloadProgressEvent {
            file: progress.file_name,
            downloaded_bytes: progress.downloaded_bytes,
            total_bytes: progress.total_bytes,
        };
        if let Err(e) = progress_app.emit("diarization-models-download-progress", event) {
            warn!("Failed to emit diarization model download progress: {}", e);
        }
    };

    let result = models::download_all(&models_dir, &DOWNLOAD_CANCELLED, &on_progress).await;
    let (event, payload) = match &result {
        Ok(()) => {
            info!("Diarization models installed in {}", models_dir.display());
            (
                "diarization-models-download-complete",
                serde_json::json!({ "models_dir": models_dir.display().to_string() }),
            )
        }
        Err(e) => {
            warn!("Diarization model download failed: {}", e);
            (
                "diarization-models-download-error",
                serde_json::json!({
                    "error": e.to_string(),
                    "cancelled": matches!(e, DiarizationError::Cancelled),
                }),
            )
        }
    };
    if let Err(e) = app.emit(event, payload) {
        warn!("Failed to emit {}: {}", event, e);
    }
    result.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn cancel_diarization_models_download() -> Result<(), String> {
    DOWNLOAD_CANCELLED.store(true, Ordering::SeqCst);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_to_fast_with_detected_speaker_count() {
        assert_eq!(diarization_config(None).unwrap(), DiarizationConfig::fast());
    }

    #[test]
    fn options_select_quality_and_speaker_count() {
        let config = diarization_config(Some(SpeakerIdOptions {
            num_speakers: Some(3),
            quality: Some("accurate".to_string()),
        }))
        .unwrap();
        assert_eq!(config.step_secs, DiarizationConfig::accurate().step_secs);
        assert_eq!(config.num_speakers, Some(3));
    }

    #[test]
    fn options_reject_unknown_quality_and_zero_speakers() {
        let bad_quality = SpeakerIdOptions {
            num_speakers: None,
            quality: Some("best".into()),
        };
        assert!(diarization_config(Some(bad_quality)).is_err());
        let zero = SpeakerIdOptions {
            num_speakers: Some(0),
            quality: None,
        };
        assert!(diarization_config(Some(zero)).is_err());
    }

    #[test]
    fn trigger_accepts_auto_and_manual_only() {
        assert_eq!(parse_trigger("auto").unwrap(), Trigger::Auto);
        assert_eq!(parse_trigger("manual").unwrap(), Trigger::Manual);
        assert!(parse_trigger("Auto").is_err());
    }
}
