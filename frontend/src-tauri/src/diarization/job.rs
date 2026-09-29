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
use tokio::sync::{Notify, Semaphore, SemaphorePermit};

use crate::audio::common::find_audio_file;
use crate::audio::decoder::decode_audio_file;
use crate::audio::recording_preferences::load_recording_preferences;
use crate::audio::retranscription::{is_retranscribing, is_retranscription_in_progress};
use crate::database::repositories::meeting::MeetingsRepository;
use crate::database::repositories::speaker::{
    JobStatus, SavedIdentification, SpeakerIdJobRow, SpeakerRepoError, SpeakerRepository,
};
use crate::state::AppState;

/// Workers while a new recording is live: Parakeet transcribes on the same CPU.
const LIVE_RECORDING_WORKERS: usize = 2;
const MAX_WORKERS: usize = 8;
/// Recordings longer than this take minutes to identify and need well over 1 GB of memory.
pub const LONG_RECORDING_SECS: f64 = 3.0 * 3600.0;
/// Sample rate of the audio the engine receives.
const ENGINE_SAMPLE_RATE: f64 = 16_000.0;

/// One identification at a time: each job holds ~1 GB of ONNX sessions and saturates the CPU.
static JOB_SLOT: Semaphore = Semaphore::const_new(1);
static JOBS: LazyLock<Mutex<HashMap<String, JobEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct JobEntry {
    cancel: JobCancel,
    running: bool,
    stage: JobStage,
    progress: u32,
}

/// Cancellation of one job: the flag the engine polls, and a wake-up for a job still waiting
/// for [`JOB_SLOT`].
#[derive(Clone, Default)]
struct JobCancel {
    flag: Arc<AtomicBool>,
    wake: Arc<Notify>,
}

impl JobCancel {
    fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        // Stores a permit when nobody waits yet, so a cancel just before the wait still counts.
        self.wake.notify_one();
    }
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
            Self::Embedding => "Analyzing voices...",
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
    #[error("This meeting is being retranscribed; try again when it finishes")]
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
        record_failure(pool, meeting_id, job_error).await;
    }
    result
}

/// Record how a job ended without a result; a deleted meeting has no row to write.
async fn record_failure(pool: &SqlitePool, meeting_id: &str, job_error: &JobError) {
    let status = match job_error {
        JobError::MeetingDeleted => return,
        JobError::Cancelled => JobStatus::Cancelled,
        _ => JobStatus::Failed,
    };
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

async fn identify(
    pool: &SqlitePool,
    meeting_id: &str,
    models_dir: &Path,
    options: &JobOptions,
    cancel: Arc<AtomicBool>,
    progress: Arc<ProgressFn>,
) -> Result<SavedIdentification, JobError> {
    ensure_not_cancelled(&cancel)?;
    if is_retranscribing(meeting_id) {
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
    let duration_secs = samples.len() as f64 / ENGINE_SAMPLE_RATE;
    if duration_secs > LONG_RECORDING_SECS {
        warn!(
            "Identifying speakers in a {:.1} h recording for {}; this takes several minutes and over 1 GB of memory",
            duration_secs / 3600.0,
            meeting_id
        );
    }
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

/// Load the sessions for this job only (they hold ~1 GB) and run the engine, whose own pool
/// keeps its parallelism within `options.workers`.
fn diarize_blocking(
    models_dir: &Path,
    options: &JobOptions,
    samples: Vec<f32>,
    cancel: &AtomicBool,
    progress: &ProgressFn,
) -> Result<meetily_diarization::DiarizationOutput, JobError> {
    crate::ensure_onnx_runtime_available().map_err(|e| JobError::Internal(e.to_string()))?;
    let engine = DiarizationEngine::load(models_dir, options.workers).map_err(engine_error)?;
    let on_progress = |stage: Stage, fraction: f32| {
        let (job_stage, percent) = engine_progress(stage, fraction);
        progress(job_stage, percent);
    };
    engine
        .diarize(&samples, &options.config, cancel, &on_progress)
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

/// Parallelism for a job: [`LIVE_RECORDING_WORKERS`] while a recording or a retranscription
/// (of any meeting) is transcribing on the same CPU, otherwise a third of the cores (the
/// prototype's sweet spot), capped at [`MAX_WORKERS`].
fn worker_count(recording_live: bool, retranscribing: bool, cores: usize) -> usize {
    if recording_live || retranscribing {
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

/// What [`start`] knows about a meeting when asked to identify its speakers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StartConditions {
    trigger: Trigger,
    /// The "Automatically identify speakers" preference; only `Auto` consults it.
    auto_enabled: bool,
    already_active: bool,
    /// This meeting is being retranscribed (another meeting's retranscription does not count).
    retranscribing: bool,
    audio_found: bool,
    models_installed: bool,
}

/// Why a start request is skipped, or `None` when the job should run.
fn skip_reason(conditions: &StartConditions) -> Option<&'static str> {
    if conditions.already_active {
        Some("already_running")
    } else if conditions.retranscribing {
        Some("retranscription_running")
    } else if conditions.trigger == Trigger::Auto && !conditions.auto_enabled {
        Some("disabled")
    } else if !conditions.audio_found {
        Some("no_audio")
    } else if !conditions.models_installed {
        Some("models_missing")
    } else {
        None
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
    let auto_enabled = match trigger {
        Trigger::Auto => load_recording_preferences(app)
            .await
            .map(|prefs| prefs.auto_identify_speakers)
            .unwrap_or_else(|e| {
                warn!("Failed to load recording preferences: {}", e);
                false
            }),
        Trigger::Manual => true,
    };
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
    let models_dir = super::models_dir(app)?;
    let conditions = StartConditions {
        trigger,
        auto_enabled,
        already_active: is_active(meeting_id),
        retranscribing: is_retranscribing(meeting_id),
        audio_found: folder.and_then(|f| find_audio_file(&f).ok()).is_some(),
        models_installed: models::missing(&models_dir).is_empty(),
    };
    if let Some(reason) = skip_reason(&conditions) {
        return Ok(StartSpeakerIdResult::skipped(reason));
    }

    let Some((registration, cancel)) = register(meeting_id) else {
        return Ok(StartSpeakerIdResult::skipped("already_running"));
    };
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
        let progress = progress_emitter(app.clone(), meeting_id.clone());
        let result = run_registered_job(
            registration,
            &JOB_SLOT,
            &pool,
            &models_dir,
            config,
            &cancel,
            progress,
        )
        .await;
        emit_result(&app, &meeting_id, result);
    });

    Ok(StartSpeakerIdResult {
        status,
        reason: None,
    })
}

/// Wait for `slot`, then run the job. A job cancelled while it waits ends at once as
/// `cancelled` rather than when the running job frees the slot. The registration is released
/// before returning, so the result event never reports a job that still blocks a new start.
async fn run_registered_job(
    registration: Registration,
    slot: &Semaphore,
    pool: &SqlitePool,
    models_dir: &Path,
    config: DiarizationConfig,
    cancel: &JobCancel,
    progress: Arc<ProgressFn>,
) -> Result<SavedIdentification, JobError> {
    let meeting_id = registration.0.as_str();
    let _permit = match wait_for_slot(slot, cancel).await {
        Ok(permit) => permit,
        Err(job_error) => {
            record_failure(pool, meeting_id, &job_error).await;
            return Err(job_error);
        }
    };
    set_running(meeting_id);
    let options = JobOptions {
        config,
        workers: worker_count(
            crate::audio::recording_commands::is_recording().await,
            is_retranscription_in_progress(),
            std::thread::available_parallelism().map_or(1, NonZeroUsize::get),
        ),
    };
    info!(
        "Identifying speakers for {} with {} workers",
        meeting_id, options.workers
    );
    run_identification(
        pool,
        meeting_id,
        models_dir,
        &options,
        cancel.flag.clone(),
        progress,
    )
    .await
}

async fn wait_for_slot<'a>(
    slot: &'a Semaphore,
    cancel: &JobCancel,
) -> Result<SemaphorePermit<'a>, JobError> {
    tokio::select! {
        biased;
        _ = cancel.wake.notified() => Err(JobError::Cancelled),
        permit = slot.acquire() => {
            permit.map_err(|e| JobError::Internal(format!("Job queue closed: {e}")))
        }
    }
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
            entry.cancel.cancel();
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

/// Add a queued job for the meeting to the registry; `None` when it already has one.
fn register(meeting_id: &str) -> Option<(Registration, JobCancel)> {
    let mut jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
    if jobs.contains_key(meeting_id) {
        return None;
    }
    let cancel = JobCancel::default();
    jobs.insert(
        meeting_id.to_string(),
        JobEntry {
            cancel: cancel.clone(),
            running: false,
            stage: JobStage::Decoding,
            progress: 0,
        },
    );
    Some((Registration(meeting_id.to_string()), cancel))
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

/// The state of the meeting's queued or running job, `None` when it has none. Callers read
/// this before the stored row: a job writes its final row before it leaves the registry, so a
/// miss here followed by a DB read never reports a just-finished job as interrupted.
pub fn live_job_state(meeting_id: &str) -> Option<JobState> {
    let jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
    jobs.get(meeting_id)
        .map(|entry| live_state(entry.running, entry.stage, entry.progress))
}

fn live_state(running: bool, stage: JobStage, progress: u32) -> JobState {
    JobState {
        status: if running { "running" } else { "queued" }.to_string(),
        stage: running.then_some(stage),
        progress_percentage: running.then_some(progress),
        speaker_count: None,
        error: None,
    }
}

/// The state of `stored`, the row read after [`live_job_state`] found no job. A job registered
/// between the two reads has already written a `queued` row, so an unfinished row is checked
/// against the registry again before it is reported as interrupted.
pub fn stored_or_live_job_state(meeting_id: &str, stored: Option<SpeakerIdJobRow>) -> JobState {
    let unfinished = stored
        .as_ref()
        .is_some_and(|row| matches!(row.status.as_str(), "queued" | "running"));
    if unfinished {
        if let Some(live) = live_job_state(meeting_id) {
            return live;
        }
    }
    stored_job_state(stored)
}

/// The state recorded for a meeting with no live job. A `queued`/`running` row then means the
/// app quit mid-run.
pub fn stored_job_state(stored: Option<SpeakerIdJobRow>) -> JobState {
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
        assert_eq!(worker_count(true, false, 24), 2);
    }

    #[test]
    fn workers_drop_to_two_while_another_meeting_is_retranscribed() {
        assert_eq!(worker_count(false, true, 24), 2);
        assert_eq!(worker_count(true, true, 24), 2);
    }

    #[test]
    fn workers_use_a_third_of_the_cores_capped_at_eight() {
        assert_eq!(worker_count(false, false, 24), 8);
        assert_eq!(worker_count(false, false, 32), 8);
        assert_eq!(worker_count(false, false, 12), 4);
        assert_eq!(worker_count(false, false, 2), 1);
        assert_eq!(worker_count(false, false, 1), 1);
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
        assert_eq!(stored_job_state(Some(row("running"))).status, "interrupted");
        assert_eq!(stored_job_state(Some(row("queued"))).status, "interrupted");
        assert_eq!(stored_job_state(Some(row("completed"))).status, "completed");
        assert_eq!(stored_job_state(None).status, "none");
    }

    #[test]
    fn a_queued_row_of_a_job_registered_after_the_registry_miss_is_not_interrupted() {
        let meeting_id = "meeting-registered-after-miss-test";
        let row = |status: &str| {
            Some(SpeakerIdJobRow {
                status: status.to_string(),
                speaker_count: None,
                error: None,
            })
        };
        assert_eq!(live_job_state(meeting_id), None);
        let (registration, _) = register(meeting_id).unwrap();

        assert_eq!(
            stored_or_live_job_state(meeting_id, row("queued")).status,
            "queued"
        );
        set_running(meeting_id);
        assert_eq!(
            stored_or_live_job_state(meeting_id, row("running")).status,
            "running"
        );

        drop(registration);
        assert_eq!(
            stored_or_live_job_state(meeting_id, row("running")).status,
            "interrupted"
        );
        assert_eq!(
            stored_or_live_job_state(meeting_id, row("completed")).status,
            "completed"
        );
        assert_eq!(stored_or_live_job_state(meeting_id, None).status, "none");
    }

    #[test]
    fn live_job_reports_queue_or_progress() {
        let queued = live_state(false, JobStage::Decoding, 0);
        assert_eq!(queued.status, "queued");
        assert_eq!(queued.progress_percentage, None);

        let running = live_state(true, JobStage::Embedding, 40);
        assert_eq!(running.status, "running");
        assert_eq!(running.stage, Some(JobStage::Embedding));
        assert_eq!(running.progress_percentage, Some(40));
    }

    #[test]
    fn cancel_flags_only_registered_jobs() {
        let meeting_id = "meeting-cancel-test";
        assert!(!cancel(meeting_id));

        let (registration, job_cancel) = register(meeting_id).unwrap();
        assert!(is_active(meeting_id));
        assert!(
            register(meeting_id).is_none(),
            "a second job for the meeting is deduped"
        );
        assert!(cancel(meeting_id));
        assert!(job_cancel.flag.load(Ordering::SeqCst));

        drop(registration);
        assert!(!is_active(meeting_id));
    }

    #[test]
    fn live_state_is_read_from_the_registry() {
        let meeting_id = "meeting-live-state-test";
        assert_eq!(live_job_state(meeting_id), None);
        let (registration, _) = register(meeting_id).unwrap();
        assert_eq!(live_job_state(meeting_id).unwrap().status, "queued");
        set_running(meeting_id);
        assert_eq!(live_job_state(meeting_id).unwrap().status, "running");
        drop(registration);
        assert_eq!(live_job_state(meeting_id), None);
    }

    async fn pool_with_meeting(meeting_id: &str) -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO meetings (id, title, created_at, updated_at) VALUES (?, 'Test', ?, ?)",
        )
        .bind(meeting_id)
        .bind(chrono::Utc::now())
        .bind(chrono::Utc::now())
        .execute(&pool)
        .await
        .unwrap();
        pool
    }

    #[tokio::test]
    async fn cancelling_a_queued_job_ends_it_without_waiting_for_the_slot() {
        let meeting_id = "meeting-queued-cancel-test";
        let pool = pool_with_meeting(meeting_id).await;
        SpeakerRepository::set_job_status(&pool, meeting_id, JobStatus::Queued, None)
            .await
            .unwrap();
        let slot = Semaphore::new(1);
        let _other_job = slot.acquire().await.unwrap();
        let (registration, job_cancel) = register(meeting_id).unwrap();

        let job = run_registered_job(
            registration,
            &slot,
            &pool,
            Path::new("/nonexistent"),
            DiarizationConfig::default(),
            &job_cancel,
            Arc::new(|_, _| {}),
        );
        let cancel_it = async {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            assert_eq!(live_job_state(meeting_id).unwrap().status, "queued");
            assert!(cancel(meeting_id));
        };
        let (result, ()) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(job, cancel_it)
        })
        .await
        .expect("a cancelled queued job must not wait for the running job");

        assert!(matches!(result, Err(JobError::Cancelled)), "{result:?}");
        assert!(
            !is_active(meeting_id),
            "the meeting can be started or retranscribed again"
        );
        let row = SpeakerRepository::get_job(&pool, meeting_id).await.unwrap();
        assert_eq!(stored_job_state(row).status, "cancelled");
    }

    fn runnable(trigger: Trigger) -> StartConditions {
        StartConditions {
            trigger,
            auto_enabled: true,
            already_active: false,
            retranscribing: false,
            audio_found: true,
            models_installed: true,
        }
    }

    #[test]
    fn start_runs_when_nothing_blocks_it() {
        assert_eq!(skip_reason(&runnable(Trigger::Auto)), None);
        assert_eq!(skip_reason(&runnable(Trigger::Manual)), None);
    }

    #[test]
    fn auto_start_is_skipped_when_the_setting_is_off_but_manual_is_not() {
        let auto = StartConditions {
            auto_enabled: false,
            ..runnable(Trigger::Auto)
        };
        assert_eq!(skip_reason(&auto), Some("disabled"));
        let manual = StartConditions {
            auto_enabled: false,
            ..runnable(Trigger::Manual)
        };
        assert_eq!(skip_reason(&manual), None);
    }

    #[test]
    fn start_is_skipped_for_each_blocking_condition() {
        for trigger in [Trigger::Auto, Trigger::Manual] {
            let base = runnable(trigger);
            let cases = [
                (
                    StartConditions {
                        models_installed: false,
                        ..base
                    },
                    "models_missing",
                ),
                (
                    StartConditions {
                        audio_found: false,
                        ..base
                    },
                    "no_audio",
                ),
                (
                    StartConditions {
                        already_active: true,
                        ..base
                    },
                    "already_running",
                ),
                (
                    StartConditions {
                        retranscribing: true,
                        ..base
                    },
                    "retranscription_running",
                ),
            ];
            for (conditions, reason) in cases {
                assert_eq!(skip_reason(&conditions), Some(reason), "{conditions:?}");
            }
        }
    }

    #[test]
    fn an_already_running_job_wins_over_other_reasons() {
        let conditions = StartConditions {
            already_active: true,
            retranscribing: true,
            audio_found: false,
            models_installed: false,
            ..runnable(Trigger::Auto)
        };
        assert_eq!(skip_reason(&conditions), Some("already_running"));
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
