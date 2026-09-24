//! Speaker identification jobs: one job at a time, per-meeting dedupe, cancellation, progress
//! events and the DB job row.
//!
//! [`run_identification`] is the job itself and needs no `AppHandle`, so it can run headless
//! against any database; [`start`] wraps it with the registry, the queue and Tauri events.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use log::{error, info, warn};
use meetily_diarization::{
    models, timeline, DiarizationConfig, DiarizationEngine, DiarizationError, Stage,
};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio::sync::Semaphore;

use crate::audio::common::find_audio_file;
use crate::audio::decoder::decode_audio_file;
use crate::audio::recording_preferences::load_recording_preferences;
use crate::audio::retranscription::is_retranscription_in_progress;
use crate::database::repositories::meeting::MeetingsRepository;
use crate::database::repositories::speaker::{
    JobStatus, SavedIdentification, SpeakerIdJobRow, SpeakerRepoError, SpeakerRepository,
};
use crate::state::AppState;

/// Workers while a new recording is live: Parakeet transcribes on the same CPU.
const LIVE_RECORDING_WORKERS: usize = 2;
const MAX_WORKERS: usize = 8;

/// One identification at a time: each job holds ~1 GB of ONNX sessions and saturates the CPU.
static JOB_SLOT: Semaphore = Semaphore::const_new(1);
static JOBS: LazyLock<Mutex<HashMap<String, JobEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct JobEntry {
    cancel: Arc<AtomicBool>,
    running: bool,
    stage: JobStage,
    progress: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStage {
    LoadingModels,
    Decoding,
    Segmenting,
    Embedding,
    Clustering,
    Saving,
}

impl JobStage {
    fn message(self) -> &'static str {
        match self {
            Self::LoadingModels => "Loading speaker models...",
            Self::Decoding => "Decoding audio...",
            Self::Segmenting => "Detecting speech...",
            Self::Embedding => "Analysing voices...",
            Self::Clustering => "Grouping speakers...",
            Self::Saving => "Saving speakers...",
        }
    }
}

/// Receives the stage and the overall 0–100 progress.
pub type ProgressFn = dyn Fn(JobStage, u32) + Send + Sync;

#[derive(Debug, Clone, PartialEq)]
pub struct JobOptions {
    pub config: DiarizationConfig,
    pub workers: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum JobError {
    #[error("This meeting has no recorded audio")]
    NoAudio,
    #[error("Speaker identification models are not installed")]
    ModelsMissing,
    #[error("Failed to decode audio: {0}")]
    DecodeFailed(String),
    #[error("Speaker identification was cancelled")]
    Cancelled,
    #[error("The meeting was deleted")]
    MeetingDeleted,
    #[error("Retranscription is in progress; try again when it finishes")]
    Busy,
    #[error("{0}")]
    Internal(String),
}

impl JobError {
    /// The `code` of the `speaker-identification-error` event.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NoAudio => "no_audio",
            Self::ModelsMissing => "models_missing",
            Self::DecodeFailed(_) => "decode_failed",
            Self::Cancelled => "cancelled",
            Self::MeetingDeleted => "meeting_deleted",
            Self::Busy => "busy",
            Self::Internal(_) => "internal",
        }
    }
}

impl From<SpeakerRepoError> for JobError {
    fn from(error: SpeakerRepoError) -> Self {
        match error {
            SpeakerRepoError::MeetingDeleted => Self::MeetingDeleted,
            other => Self::Internal(other.to_string()),
        }
    }
}

/// Identify the speakers of one meeting and store the result. Records the job row
/// (`running`, then `completed`, `failed` or `cancelled`); a deleted meeting leaves no row.
pub async fn run_identification(
    pool: &SqlitePool,
    meeting_id: &str,
    models_dir: &Path,
    options: &JobOptions,
    cancel: Arc<AtomicBool>,
    progress: Arc<ProgressFn>,
) -> Result<SavedIdentification, JobError> {
    let result = identify(pool, meeting_id, models_dir, options, cancel, progress).await;
    if let Err(job_error) = &result {
        let status = match job_error {
            JobError::MeetingDeleted => None,
            JobError::Cancelled => Some(JobStatus::Cancelled),
            _ => Some(JobStatus::Failed),
        };
        if let Some(status) = status {
            let message = job_error.to_string();
            if let Err(e) =
                SpeakerRepository::set_job_status(pool, meeting_id, status, Some(&message)).await
            {
                warn!(
                    "Failed to record speaker identification status for {}: {}",
                    meeting_id, e
                );
            }
        }
    }
    result
}

async fn identify(
    pool: &SqlitePool,
    meeting_id: &str,
    models_dir: &Path,
    options: &JobOptions,
    cancel: Arc<AtomicBool>,
    progress: Arc<ProgressFn>,
) -> Result<SavedIdentification, JobError> {
    ensure_not_cancelled(&cancel)?;
    if is_retranscription_in_progress() {
        return Err(JobError::Busy);
    }
    SpeakerRepository::set_job_status(pool, meeting_id, JobStatus::Running, None).await?;

    let folder = recording_folder(pool, meeting_id)
        .await?
        .ok_or(JobError::NoAudio)?;
    let audio_path = find_audio_file(&folder).map_err(|_| JobError::NoAudio)?;
    if !models::missing(models_dir).is_empty() {
        return Err(JobError::ModelsMissing);
    }

    progress(JobStage::Decoding, 0);
    let samples = tokio::task::spawn_blocking(move || {
        // The 48 kHz buffer is dropped when this closure returns, before diarization starts.
        decode_audio_file(&audio_path).map(|decoded| decoded.to_whisper_format())
    })
    .await
    .map_err(|e| JobError::Internal(format!("Decode task panicked: {e}")))?
    .map_err(|e| JobError::DecodeFailed(e.to_string()))?;
    progress(JobStage::Decoding, 10);
    ensure_not_cancelled(&cancel)?;

    progress(JobStage::LoadingModels, 10);
    let output = {
        let models_dir = models_dir.to_path_buf();
        let options = options.clone();
        let cancel = cancel.clone();
        let progress = progress.clone();
        tokio::task::spawn_blocking(move || {
            diarize_blocking(&models_dir, &options, samples, &cancel, &*progress)
        })
        .await
        .map_err(|e| JobError::Internal(format!("Diarization task panicked: {e}")))??
    };
    ensure_not_cancelled(&cancel)?;

    progress(JobStage::Saving, 90);
    let clock = timeline::detect_clock(&folder);
    let saved = SpeakerRepository::save_result(pool, meeting_id, &output, clock).await?;
    progress(JobStage::Saving, 100);
    Ok(saved)
}

/// Load the sessions for this job only (they hold ~1 GB) and run the engine on a dedicated
/// rayon pool, so its parallelism stays within `options.workers`.
fn diarize_blocking(
    models_dir: &Path,
    options: &JobOptions,
    samples: Vec<f32>,
    cancel: &AtomicBool,
    progress: &ProgressFn,
) -> Result<meetily_diarization::DiarizationOutput, JobError> {
    crate::ensure_onnx_runtime_available().map_err(|e| JobError::Internal(e.to_string()))?;
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .thread_name(|i| format!("diarization-{i}"))
        .build()
        .map_err(|e| JobError::Internal(format!("Failed to start diarization threads: {e}")))?;
    let engine = DiarizationEngine::load(models_dir, options.workers).map_err(engine_error)?;
    let on_progress = |stage: Stage, fraction: f32| {
        let (job_stage, percent) = engine_progress(stage, fraction);
        progress(job_stage, percent);
    };
    thread_pool
        .install(|| engine.diarize(&samples, &options.config, cancel, &on_progress))
        .map_err(engine_error)
}

fn engine_error(error: DiarizationError) -> JobError {
    match error {
        DiarizationError::Cancelled => JobError::Cancelled,
        DiarizationError::ModelsMissing(_) => JobError::ModelsMissing,
        other => JobError::Internal(other.to_string()),
    }
}

/// Map an engine stage and its 0..=1 fraction onto the overall scale: decoding 0–10,
/// segmenting 10–25, embedding 25–85, clustering 85–90, saving 90–100.
fn engine_progress(stage: Stage, fraction: f32) -> (JobStage, u32) {
    let (job_stage, start, span) = match stage {
        Stage::Segmenting => (JobStage::Segmenting, 10.0, 15.0),
        Stage::Embedding => (JobStage::Embedding, 25.0, 60.0),
        Stage::Clustering => (JobStage::Clustering, 85.0, 5.0),
    };
    let percent = start + span * fraction.clamp(0.0, 1.0);
    (job_stage, percent.round() as u32)
}

fn ensure_not_cancelled(cancel: &AtomicBool) -> Result<(), JobError> {
    if cancel.load(Ordering::SeqCst) {
        Err(JobError::Cancelled)
    } else {
        Ok(())
    }
}

/// The meeting's recording folder, `None` for meetings saved without one.
async fn recording_folder(
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<Option<PathBuf>, JobError> {
    let meeting = MeetingsRepository::get_meeting_metadata(pool, meeting_id)
        .await
        .map_err(|e| JobError::Internal(e.to_string()))?
        .ok_or(JobError::MeetingDeleted)?;
    Ok(meeting.folder_path.map(PathBuf::from))
}

/// Parallelism for a job: [`LIVE_RECORDING_WORKERS`] while recording, otherwise a third of
/// the cores (the prototype's sweet spot), capped at [`MAX_WORKERS`].
fn worker_count(recording_live: bool, cores: usize) -> usize {
    if recording_live {
        LIVE_RECORDING_WORKERS
    } else {
        (cores / 3).clamp(1, MAX_WORKERS)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// After a recording or import; skipped quietly when not applicable.
    Auto,
    Manual,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StartStatus {
    Started,
    Queued,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StartSpeakerIdResult {
    pub status: StartStatus,
    /// Why the job was skipped: `already_running`, `retranscription_running`, `disabled`
    /// (auto only), `no_audio` or `models_missing`.
    pub reason: Option<String>,
}

impl StartSpeakerIdResult {
    fn skipped(reason: &str) -> Self {
        Self {
            status: StartStatus::Skipped,
            reason: Some(reason.to_string()),
        }
    }
}

/// Queue identification for a meeting and return at once. Progress, completion and errors
/// arrive as `speaker-identification-*` events.
pub async fn start<R: Runtime>(
    app: &AppHandle<R>,
    meeting_id: &str,
    trigger: Trigger,
    config: DiarizationConfig,
) -> Result<StartSpeakerIdResult, String> {
    if is_active(meeting_id) {
        return Ok(StartSpeakerIdResult::skipped("already_running"));
    }
    if is_retranscription_in_progress() {
        return Ok(StartSpeakerIdResult::skipped("retranscription_running"));
    }
    if trigger == Trigger::Auto {
        let enabled = load_recording_preferences(app)
            .await
            .map(|prefs| prefs.auto_identify_speakers)
            .unwrap_or_else(|e| {
                warn!("Failed to load recording preferences: {}", e);
                false
            });
        if !enabled {
            return Ok(StartSpeakerIdResult::skipped("disabled"));
        }
    }

    let pool = app
        .try_state::<AppState>()
        .ok_or("Database is not initialized")?
        .db_manager
        .pool()
        .clone();
    let folder = match recording_folder(&pool, meeting_id).await {
        Ok(folder) => folder,
        Err(JobError::MeetingDeleted) => return Err(format!("Meeting not found: {meeting_id}")),
        Err(e) => return Err(e.to_string()),
    };
    if folder.and_then(|f| find_audio_file(&f).ok()).is_none() {
        return Ok(StartSpeakerIdResult::skipped("no_audio"));
    }
    let models_dir = super::models_dir(app)?;
    if !models::missing(&models_dir).is_empty() {
        return Ok(StartSpeakerIdResult::skipped("models_missing"));
    }

    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
        if jobs.contains_key(meeting_id) {
            return Ok(StartSpeakerIdResult::skipped("already_running"));
        }
        jobs.insert(
            meeting_id.to_string(),
            JobEntry {
                cancel: cancel.clone(),
                running: false,
                stage: JobStage::Decoding,
                progress: 0,
            },
        );
    }
    let registration = Registration(meeting_id.to_string());

    if let Err(e) =
        SpeakerRepository::set_job_status(&pool, meeting_id, JobStatus::Queued, None).await
    {
        return Err(match e {
            SpeakerRepoError::MeetingDeleted => format!("Meeting not found: {meeting_id}"),
            other => other.to_string(),
        });
    }
    let status = if JOB_SLOT.available_permits() == 0 {
        StartStatus::Queued
    } else {
        StartStatus::Started
    };
    info!(
        "Speaker identification for {} {:?} ({:?})",
        meeting_id, status, trigger
    );

    let app = app.clone();
    let meeting_id = meeting_id.to_string();
    tauri::async_runtime::spawn(async move {
        let _registration = registration;
        let result = match JOB_SLOT.acquire().await {
            Ok(_permit) => {
                set_running(&meeting_id);
                let options = JobOptions {
                    config,
                    workers: worker_count(
                        crate::audio::recording_commands::is_recording().await,
                        std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
                    ),
                };
                info!(
                    "Identifying speakers for {} with {} workers",
                    meeting_id, options.workers
                );
                let progress = progress_emitter(app.clone(), meeting_id.clone());
                run_identification(&pool, &meeting_id, &models_dir, &options, cancel, progress)
                    .await
            }
            Err(e) => Err(JobError::Internal(format!("Job queue closed: {e}"))),
        };
        emit_result(&app, &meeting_id, result);
    });

    Ok(StartSpeakerIdResult {
        status,
        reason: None,
    })
}

/// The auto trigger for Rust-side entry points (import). Never fails the caller.
pub async fn maybe_start_auto<R: Runtime>(app: &AppHandle<R>, meeting_id: &str) {
    match start(app, meeting_id, Trigger::Auto, DiarizationConfig::default()).await {
        Ok(result) => info!(
            "Auto speaker identification for {}: {:?} {:?}",
            meeting_id, result.status, result.reason
        ),
        Err(e) => warn!(
            "Auto speaker identification for {} not started: {}",
            meeting_id, e
        ),
    }
}

/// Request cancellation of a queued or running job. Returns whether one was registered.
pub fn cancel(meeting_id: &str) -> bool {
    let jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
    match jobs.get(meeting_id) {
        Some(entry) => {
            entry.cancel.store(true, Ordering::SeqCst);
            true
        }
        None => false,
    }
}

/// Whether a job for this meeting is queued or running.
pub fn is_active(meeting_id: &str) -> bool {
    JOBS.lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(meeting_id)
}

/// Removes the registry entry when the job ends, however it ends.
struct Registration(String);

impl Drop for Registration {
    fn drop(&mut self) {
        JOBS.lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.0);
    }
}

fn set_running(meeting_id: &str) {
    if let Some(entry) = JOBS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_mut(meeting_id)
    {
        entry.running = true;
    }
}

#[derive(Debug, Clone, Serialize)]
struct ProgressEvent {
    meeting_id: String,
    stage: JobStage,
    progress_percentage: u32,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct CompleteEvent {
    meeting_id: String,
    speaker_count: usize,
    labeled_segments: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ErrorEvent {
    meeting_id: String,
    code: &'static str,
    error: String,
}

/// Records progress in the registry and emits an event whenever the stage or whole percent
/// changes (the engine reports far more often than that).
fn progress_emitter<R: Runtime>(app: AppHandle<R>, meeting_id: String) -> Arc<ProgressFn> {
    Arc::new(move |stage: JobStage, percent: u32| {
        let changed = {
            let mut jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
            match jobs.get_mut(&meeting_id) {
                Some(entry) if entry.stage != stage || entry.progress != percent => {
                    entry.stage = stage;
                    entry.progress = percent;
                    true
                }
                _ => false,
            }
        };
        if changed {
            let event = ProgressEvent {
                meeting_id: meeting_id.clone(),
                stage,
                progress_percentage: percent,
                message: stage.message().to_string(),
            };
            if let Err(e) = app.emit("speaker-identification-progress", event) {
                warn!("Failed to emit speaker identification progress: {}", e);
            }
        }
    })
}

fn emit_result<R: Runtime>(
    app: &AppHandle<R>,
    meeting_id: &str,
    result: Result<SavedIdentification, JobError>,
) {
    let emitted = match result {
        Ok(saved) => {
            info!(
                "Speaker identification for {} complete: {} speakers, {} segments",
                meeting_id, saved.speaker_count, saved.labeled_segments
            );
            app.emit(
                "speaker-identification-complete",
                CompleteEvent {
                    meeting_id: meeting_id.to_string(),
                    speaker_count: saved.speaker_count,
                    labeled_segments: saved.labeled_segments,
                },
            )
        }
        Err(job_error) => {
            match &job_error {
                JobError::Cancelled | JobError::MeetingDeleted => {
                    info!(
                        "Speaker identification for {} stopped: {}",
                        meeting_id, job_error
                    )
                }
                _ => error!(
                    "Speaker identification for {} failed: {}",
                    meeting_id, job_error
                ),
            }
            app.emit(
                "speaker-identification-error",
                ErrorEvent {
                    meeting_id: meeting_id.to_string(),
                    code: job_error.code(),
                    error: job_error.to_string(),
                },
            )
        }
    };
    if let Err(e) = emitted {
        warn!("Failed to emit speaker identification result: {}", e);
    }
}

/// A job's state as reported by `get_speaker_identification_status`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JobState {
    /// `none`, `queued`, `running`, `completed`, `failed`, `cancelled` or `interrupted`.
    pub status: String,
    pub stage: Option<JobStage>,
    pub progress_percentage: Option<u32>,
    pub speaker_count: Option<i64>,
    pub error: Option<String>,
}

/// Combine the live registry with the stored row. A stored `queued`/`running` row without a
/// live job means the app quit mid-run.
pub fn job_state(meeting_id: &str, stored: Option<SpeakerIdJobRow>) -> JobState {
    let live = {
        let jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
        jobs.get(meeting_id)
            .map(|entry| (entry.running, entry.stage, entry.progress))
    };
    resolve_job_state(live, stored)
}

fn resolve_job_state(
    live: Option<(bool, JobStage, u32)>,
    stored: Option<SpeakerIdJobRow>,
) -> JobState {
    if let Some((running, stage, progress)) = live {
        let status = if running { "running" } else { "queued" };
        return JobState {
            status: status.to_string(),
            stage: running.then_some(stage),
            progress_percentage: running.then_some(progress),
            speaker_count: None,
            error: None,
        };
    }
    match stored {
        Some(row) => {
            let status = match row.status.as_str() {
                "queued" | "running" => "interrupted".to_string(),
                other => other.to_string(),
            };
            JobState {
                status,
                stage: None,
                progress_percentage: None,
                speaker_count: row.speaker_count,
                error: row.error,
            }
        }
        None => JobState {
            status: "none".to_string(),
            stage: None,
            progress_percentage: None,
            speaker_count: None,
            error: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_drop_to_two_while_recording() {
        assert_eq!(worker_count(true, 24), 2);
    }

    #[test]
    fn workers_use_a_third_of_the_cores_capped_at_eight() {
        assert_eq!(worker_count(false, 24), 8);
        assert_eq!(worker_count(false, 32), 8);
        assert_eq!(worker_count(false, 12), 4);
        assert_eq!(worker_count(false, 2), 1);
        assert_eq!(worker_count(false, 1), 1);
    }

    #[test]
    fn engine_progress_follows_stage_weights() {
        assert_eq!(
            engine_progress(Stage::Segmenting, 0.0),
            (JobStage::Segmenting, 10)
        );
        assert_eq!(
            engine_progress(Stage::Segmenting, 1.0),
            (JobStage::Segmenting, 25)
        );
        assert_eq!(
            engine_progress(Stage::Embedding, 0.5),
            (JobStage::Embedding, 55)
        );
        assert_eq!(
            engine_progress(Stage::Embedding, 1.0),
            (JobStage::Embedding, 85)
        );
        assert_eq!(
            engine_progress(Stage::Clustering, 1.0),
            (JobStage::Clustering, 90)
        );
        assert_eq!(
            engine_progress(Stage::Clustering, 7.0),
            (JobStage::Clustering, 90)
        );
    }

    #[test]
    fn stale_running_row_without_live_job_is_interrupted() {
        let row = |status: &str| SpeakerIdJobRow {
            status: status.to_string(),
            speaker_count: None,
            error: None,
        };
        assert_eq!(
            resolve_job_state(None, Some(row("running"))).status,
            "interrupted"
        );
        assert_eq!(
            resolve_job_state(None, Some(row("queued"))).status,
            "interrupted"
        );
        assert_eq!(
            resolve_job_state(None, Some(row("completed"))).status,
            "completed"
        );
        assert_eq!(resolve_job_state(None, None).status, "none");
    }

    #[test]
    fn live_job_reports_queue_or_progress() {
        let queued = resolve_job_state(Some((false, JobStage::Decoding, 0)), None);
        assert_eq!(queued.status, "queued");
        assert_eq!(queued.progress_percentage, None);

        let running = resolve_job_state(Some((true, JobStage::Embedding, 40)), None);
        assert_eq!(running.status, "running");
        assert_eq!(running.stage, Some(JobStage::Embedding));
        assert_eq!(running.progress_percentage, Some(40));
    }

    #[test]
    fn cancel_flags_only_registered_jobs() {
        let meeting_id = "meeting-cancel-test";
        assert!(!cancel(meeting_id));

        let flag = Arc::new(AtomicBool::new(false));
        JOBS.lock().unwrap().insert(
            meeting_id.to_string(),
            JobEntry {
                cancel: flag.clone(),
                running: true,
                stage: JobStage::Decoding,
                progress: 0,
            },
        );
        let registration = Registration(meeting_id.to_string());
        assert!(is_active(meeting_id));
        assert!(cancel(meeting_id));
        assert!(flag.load(Ordering::SeqCst));

        drop(registration);
        assert!(!is_active(meeting_id));
    }

    #[test]
    fn error_codes_match_the_event_contract() {
        assert_eq!(JobError::NoAudio.code(), "no_audio");
        assert_eq!(JobError::MeetingDeleted.code(), "meeting_deleted");
        assert_eq!(
            JobError::from(SpeakerRepoError::MeetingDeleted).code(),
            "meeting_deleted"
        );
        assert_eq!(JobError::Busy.code(), "busy");
    }
}
