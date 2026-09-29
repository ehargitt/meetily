// audio/recording_commands.rs
//
// Slim Tauri command layer for recording functionality.
// Delegates to transcription and recording modules for actual implementation.

use anyhow::Result;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio::task::{AbortHandle, JoinHandle};
use tokio_util::sync::CancellationToken;

use super::{
    recording_manager::RecordingStartError,
    parse_audio_device,
    default_input_device,   // Get default microphone
    default_output_device,  // Get default system audio
    RecordingManager,
};
use super::device_monitor::{DeviceEvent, DeviceMonitorType};
use super::recording_state::{DeviceType as RecordingDeviceType, StreamFault, StreamHealthEvent};

// Import transcription modules
use super::transcription::{
    self,
    reset_speech_detected_flag,
    EngineKind,
    TranscriptionProgress,
    TranscriptionTask,
};
use super::recording_saver::{TranscriptSegment, TranscriptStore};

// Re-export TranscriptUpdate for backward compatibility
pub use super::transcription::TranscriptUpdate;

// ============================================================================
// GLOBAL STATE
// ============================================================================

// Simple recording state tracking
static IS_RECORDING: AtomicBool = AtomicBool::new(false);

/// The stop guard: true from the moment one `stop_recording` call wins it until
/// that call returns. Concurrent stops (UI + tray, double clicks) lose the
/// compare-exchange and return `STOP_IN_PROGRESS_ERROR` without side effects,
/// and starts are refused while it is held.
///
/// `IS_RECORDING` stays true through the stop tail — the frontend polls it to
/// keep the stop UI up — so the mic-disconnect fallback checks this flag too,
/// otherwise a fallback queued before Stop retries against a taken manager and
/// surfaces a spurious "Microphone fallback failed" toast.
static STOP_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Returned by a stop that lost the guard to one already running.
pub const STOP_IN_PROGRESS_ERROR: &str = "STOP_IN_PROGRESS";

/// Recording is live and not being torn down — the only state in which the
/// mic-disconnect fallback should run or report.
fn recording_live() -> bool {
    IS_RECORDING.load(Ordering::SeqCst) && !STOP_IN_PROGRESS.load(Ordering::SeqCst)
}

/// Recording is live AND the global manager is still the session `s` belongs to.
/// Used by the mic-disconnect fallback to refuse acting on a *later* recording
/// after a Stop/Start swapped the manager out from under an in-flight task.
///
/// NOTE: this locks `RECORDING_MANAGER`. Never call it while already holding
/// that lock (e.g. inside a `RECORDING_MANAGER.lock()` scope) — the std Mutex
/// is non-reentrant and it would self-deadlock. All current callers invoke it
/// outside any held lock; keep it that way.
fn session_live(s: &Arc<super::RecordingState>) -> bool {
    recording_live()
        && RECORDING_MANAGER
            .lock()
            .unwrap()
            .as_ref()
            .map_or(false, |m| Arc::ptr_eq(m.get_state(), s))
}

/// Holds `STOP_IN_PROGRESS` for one stop and clears it on Drop — including
/// during unwind — so a panic anywhere in the stop tail can't leave the flag
/// stuck true, which would block every later start and stop.
///
/// This unwind-clears behaviour depends on `panic = "unwind"` (the default).
/// If a release profile ever sets `panic = "abort"`, Drop won't run on panic
/// and the stuck-flag failure mode returns — add a start-time reset then.
struct StopGuard;
impl StopGuard {
    /// Win the stop guard, or `None` if another stop holds it.
    fn try_acquire() -> Option<Self> {
        STOP_IN_PROGRESS
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .ok()
            .map(|_| StopGuard)
    }
}
impl Drop for StopGuard {
    fn drop(&mut self) {
        STOP_IN_PROGRESS.store(false, Ordering::SeqCst);
    }
}

/// Why a new recording cannot start right now, if it cannot.
fn start_blocker() -> Option<&'static str> {
    if STOP_IN_PROGRESS.load(Ordering::SeqCst) {
        Some("The previous recording is still stopping. Try again in a moment.")
    } else if IS_RECORDING.load(Ordering::SeqCst) {
        Some("Recording already in progress")
    } else {
        None
    }
}

/// Who asked for a stop; reported in the `recording-stopping` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopSource {
    Ui,
    Tray,
}

impl StopSource {
    fn as_str(self) -> &'static str {
        match self {
            StopSource::Ui => "ui",
            StopSource::Tray => "tray",
        }
    }
}

/// The transcription model is in use (or about to be): a recording is live, is
/// stopping, or a stopped recording's transcription is still finishing in the
/// background. Model loads, unloads and batch jobs must wait for it.
pub fn transcription_engine_in_use() -> bool {
    IS_RECORDING.load(Ordering::SeqCst)
        || STOP_IN_PROGRESS.load(Ordering::SeqCst)
        || LINGERING_DRAIN
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|drain| !drain.finisher.is_finished())
}

/// User-facing error for operations refused by `transcription_engine_in_use`.
pub fn engine_in_use_error(action: &str) -> String {
    format!(
        "Cannot {} while a recording is in progress or its transcription is still finishing. Try again after the recording has been saved.",
        action
    )
}

/// Shared start-path finalize. Both start commands MUST call this so a new
/// start path can't silently ship with a per-session flag left unreset (e.g.
/// the mic-recovery budget already exhausted).
fn finalize_recording_start() {
    info!("🔍 Setting IS_RECORDING to true and resetting SPEECH_DETECTED_EMITTED");
    IS_RECORDING.store(true, Ordering::SeqCst);
    MIC_FALLBACK_FAILED_ATTEMPTS.store(0, Ordering::SeqCst); // fresh mic-recovery budget per session
    reset_speech_detected_flag(); // reset speech-detected emit latch for the new session
}

// Global recording manager and transcription task to keep them alive during recording
static RECORDING_MANAGER: Mutex<Option<RecordingManager>> = Mutex::new(None);
static TRANSCRIPTION_TASK: Mutex<Option<TranscriptionTask>> = Mutex::new(None);

/// Accounting for the most recent recording's transcription. Unlike
/// `TRANSCRIPTION_TASK` it survives stop, so status polls during and after the
/// stop drain report real numbers.
static TRANSCRIPTION_PROGRESS: Mutex<Option<Arc<TranscriptionProgress>>> = Mutex::new(None);

/// The current recording's `transcript-update` listener and the store it writes to.
static TRANSCRIPT_LISTENER: Mutex<Option<TranscriptListener>> = Mutex::new(None);

/// A registered transcript listener. It holds a strong reference to its store:
/// the store's debounced writer only holds a weak one, so whoever removes the
/// listener must flush the store (`close_transcript_listener`) or the segments
/// since the last debounced write are lost.
struct TranscriptListener {
    id: tauri::EventId,
    store: Arc<TranscriptStore>,
}

/// A stopped recording whose transcription outlived the stop drain timeout.
/// `finisher` waits for it, then closes its transcript listener and unloads
/// the model; a new recording abandons it instead (see `abandon_lingering_drain`).
struct LingeringDrain {
    finisher: JoinHandle<()>,
    transcription: AbortHandle,
    cancel: CancellationToken,
    listener: Option<TranscriptListener>,
}

static LINGERING_DRAIN: Mutex<Option<LingeringDrain>> = Mutex::new(None);

/// Bound on saving a stopped recording: finalizing encodes all of its audio, so
/// the bound grows with its length (encoding must run at least 4x real time).
fn save_timeout_for(recorded_seconds: u64) -> std::time::Duration {
    const BASE: std::time::Duration = std::time::Duration::from_secs(300);
    BASE + std::time::Duration::from_secs(recorded_seconds / 4)
}

/// When the last successful stop finished, and how many meetings the frontend
/// had saved by then; its own save of that meeting comes after.
#[derive(Debug, Clone, Copy)]
pub struct CompletedStop {
    pub at: std::time::Instant,
    pub saved_meetings: u64,
}

static LAST_COMPLETED_STOP: Mutex<Option<CompletedStop>> = Mutex::new(None);

/// The last stop that completed, for tray Quit to wait on its frontend save.
pub fn last_completed_stop() -> Option<CompletedStop> {
    *LAST_COMPLETED_STOP.lock().unwrap_or_else(|e| e.into_inner())
}

/// Longest the stop waits for queued chunks to be transcribed before saving.
const TRANSCRIPTION_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

const TRANSCRIPTION_RUNTIME_START_ERROR_CODE: &str =
    "TRANSCRIPTION_RUNTIME_INITIALIZATION_FAILED";
const TRANSCRIPTION_RUNTIME_USER_MESSAGE: &str = "Speech recognition could not initialize. Restart Meetily. If the problem continues, repair or reinstall the app.";

// ============================================================================
// PUBLIC TYPES
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct RecordingArgs {
    pub save_path: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct TranscriptionStatus {
    pub chunks_in_queue: usize,
    pub is_processing: bool,
    pub last_activity_ms: u64,
}

fn map_recording_start_error<R: Runtime>(
    app: &AppHandle<R>,
    error: RecordingStartError,
) -> String {
    crate::tray::update_tray_menu(app);

    match error {
        RecordingStartError::TranscriptionRuntime(source) => {
            error!("Failed to initialize speech recognition: {source:#}");
            let error = RecordingStartError::TranscriptionRuntime(source);
            if let Err(emit_error) = app.emit("transcription-error", serde_json::json!({
                "error": error.to_string(),
                "userMessage": TRANSCRIPTION_RUNTIME_USER_MESSAGE,
                "actionable": false,
                "phase": "startup"
            })) {
                error!("Failed to emit transcription runtime startup error: {emit_error}");
            }
            TRANSCRIPTION_RUNTIME_START_ERROR_CODE.to_string()
        }
        RecordingStartError::Other(error) => format!("Failed to start recording: {error}"),
    }
}

/// Shared tail of both start commands, run once `RecordingManager::start_recording`
/// has succeeded: refuse a recording that would save no audio, then install the
/// manager globally, flip recording live and start transcription.
async fn activate_recording<R: Runtime>(
    app: &AppHandle<R>,
    mut manager: RecordingManager,
    transcription_receiver: tokio::sync::mpsc::UnboundedReceiver<super::AudioChunk>,
    auto_save: bool,
) -> Result<(), String> {
    let saver_session = super::recording_saver::take_started_session();

    // With auto-save on, a missing meeting folder means the whole meeting's
    // audio would be silently discarded: fail the start instead.
    if auto_save {
        if let Some(reason) = saver_session.as_ref().and_then(|s| s.folder_error.clone()) {
            if let Err(e) = manager.stop_streams_and_force_flush().await {
                warn!("Failed to stop streams after meeting folder error: {}", e);
            }
            crate::tray::update_tray_menu(app);
            let recordings_folder = super::recording_preferences::get_default_recordings_folder();
            return Err(format!(
                "Recording cannot start: the meeting folder could not be created in {} ({}). Check that the recordings folder exists and is writable, or choose another one in Settings.",
                recordings_folder.display(),
                reason
            ));
        }
    }

    // Take the device event receiver BEFORE storing manager globally.
    // A background task will process device events (hot-swap) without frontend polling.
    let device_event_receiver = manager.take_device_event_receiver();
    let session = manager.get_state().clone();

    // Store the manager globally to keep it alive
    {
        let mut global_manager = RECORDING_MANAGER.lock().unwrap();
        *global_manager = Some(manager);
    }

    // Spawn background device event processor (mic-disconnect fallback).
    if let Some(receiver) = device_event_receiver {
        spawn_device_event_processor(app.clone(), receiver, session);
    }

    // Flip recording live + reset per-session flags (speech-detected latch,
    // mic-recovery budget). Shared with the other start path — see helper.
    finalize_recording_start();

    let task = transcription::start_transcription_task(app.clone(), transcription_receiver);
    let session_id = task.session_id;
    *TRANSCRIPTION_PROGRESS.lock().unwrap() = Some(task.progress.clone());
    *TRANSCRIPTION_TASK.lock().unwrap() = Some(task);

    if let Some(saver_session) = saver_session {
        let app_for_save_errors = app.clone();
        saver_session.save_errors.set_sink(move |message| {
            if let Err(e) = app_for_save_errors.emit(
                "recording-save-error",
                serde_json::json!({ "message": message }),
            ) {
                error!("Failed to emit recording-save-error: {}", e);
            }
        });
        register_transcript_listener(app, saver_session.transcripts, session_id);
    }

    Ok(())
}

/// Save every `transcript-update` of transcription session `session_id` into
/// this recording's transcript store, for transcripts.json and page-reload
/// sync. The listener owns the store, so it keeps working through the stop
/// drain (after the manager has been taken) and never touches
/// `RECORDING_MANAGER` on the emitting worker thread.
fn register_transcript_listener<R: Runtime>(
    app: &AppHandle<R>,
    store: Arc<TranscriptStore>,
    session_id: u64,
) {
    use tauri::Listener;
    let listener_store = Arc::clone(&store);
    let id = app.listen("transcript-update", move |event: tauri::Event| {
        if let Some(segment) = segment_for_session(event.payload(), session_id) {
            listener_store.upsert(segment);
        }
    });
    *TRANSCRIPT_LISTENER.lock().unwrap() = Some(TranscriptListener { id, store });
    info!("✅ Transcript-update event listener registered for history persistence");
}

/// The transcript segment in a `transcript-update` payload, if it belongs to
/// `session_id`. Output of an earlier recording's transcription is dropped.
fn segment_for_session(payload: &str, session_id: u64) -> Option<TranscriptSegment> {
    let update = match serde_json::from_str::<TranscriptUpdate>(payload) {
        Ok(update) => update,
        Err(e) => {
            warn!("Ignoring malformed transcript-update payload: {}", e);
            return None;
        }
    };
    if update.session_id != session_id {
        warn!(
            "Ignoring transcript-update from transcription session {} (recording's session is {})",
            update.session_id, session_id
        );
        return None;
    }
    Some(TranscriptSegment {
        id: format!("seg_{}", update.sequence_id),
        text: update.text,
        audio_start_time: update.audio_start_time,
        audio_end_time: update.audio_end_time,
        duration: update.duration,
        display_time: update.timestamp, // Use wall-clock timestamp for display
        confidence: update.confidence,
        sequence_id: update.sequence_id,
    })
}

/// Remove a transcript listener and write everything its store holds to
/// transcripts.json. `unlisten` drops the listener closure, which may hold the
/// last strong reference to the store.
async fn close_transcript_listener(listener: TranscriptListener, unlisten: impl FnOnce(tauri::EventId)) {
    unlisten(listener.id);
    let store = listener.store;
    match tokio::task::spawn_blocking(move || store.write_now()).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => warn!("Failed to write final transcripts.json: {}", e),
        Err(e) => warn!("Final transcripts.json write task failed: {}", e),
    }
}

/// A new recording is starting while the previous one's transcription is still
/// finishing in the background. Its meeting was already saved, and its output
/// would otherwise land in the new recording, so stop it: cancel the worker
/// (aborting the task alone would detach it), and close its listener, writing
/// what it has transcribed so far to its own transcripts.json. The model stays
/// loaded for the new recording.
async fn abandon_lingering_drain<R: Runtime>(app: &AppHandle<R>) {
    use tauri::Listener;
    let Some(drain) = LINGERING_DRAIN.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    if !drain.finisher.is_finished() {
        warn!("Starting a new recording: abandoning the previous recording's unfinished transcription");
    }
    drain.cancel.cancel();
    drain.transcription.abort();
    drain.finisher.abort();
    if let Some(listener) = drain.listener {
        close_transcript_listener(listener, |id| app.unlisten(id)).await;
    }
}

// ============================================================================
// DEVICE RESOLUTION
// ============================================================================

/// Resolve the microphone to record with: requested device (if it actually
/// enumerates) → system default → none (system-audio-only recording).
///
/// The device picker has no "no microphone" option: choosing "Default
/// Microphone" sends `None`, so `None` here means "use the system default",
/// NOT "record without a mic". A specifically-requested mic that isn't in
/// cpal's current enumeration (a stale saved device, or a Continuity
/// "iPhone Microphone" that isn't available right now) is downgraded to the
/// system default — the same `default_input_device()` helper the
/// mid-recording disconnect path uses — so start never hard-fails with
/// "Device not found".
///
/// Emits at most one event per call:
/// - `mic-device-switched` — a specific mic was requested but unavailable,
///   and we fell back to the default (reuses the existing frontend listener).
/// - `mic-unavailable` — no usable mic at all; recording proceeds with
///   system audio only. If system audio is also unavailable, start_streams'
///   own guard reports it.
/// Resolving `None` to the default is the user's actual choice, so it's silent.
///
/// ponytail: sync pre-flight substitution (matches Pro), not catch-and-retry —
/// stream.rs keeps its hard-fail as the last line of defense. cpal calls
/// block briefly either way.
fn resolve_mic_or_default<R: Runtime>(
    app: &AppHandle<R>,
    requested_name: Option<&str>,
) -> Option<Arc<super::AudioDevice>> {
    #[cfg(not(target_os = "linux"))]
    use cpal::traits::{DeviceTrait, HostTrait};

    let requested_specific = requested_name.is_some();

    if let Some(name) = requested_name {
        match parse_audio_device(name) {
            Ok(device) => {
                // Linux: check ALSA name hints; cpal enumeration opens every PCM.
                #[cfg(target_os = "linux")]
                let exists = super::devices::platform::resolve_capture_pcm(&device).is_ok();
                #[cfg(not(target_os = "linux"))]
                let exists = cpal::default_host()
                    .input_devices()
                    .map(|mut it| it.any(|d| d.name().map(|n| n == device.name).unwrap_or(false)))
                    .unwrap_or(false);
                if exists {
                    info!("✅ Using requested microphone: '{}'", device.name);
                    return Some(Arc::new(device));
                }
                warn!(
                    "⚠️ Requested mic '{}' not enumerated — falling back to system default",
                    device.name
                );
            }
            Err(e) => {
                warn!(
                    "⚠️ Requested mic '{}' not available: {} — falling back to system default",
                    name, e
                );
            }
        }
    }

    match default_input_device() {
        Ok(device) => {
            info!("✅ Using default microphone: '{}'", device.name);
            if requested_specific {
                // Tell the user their selected mic wasn't available and which
                // mic is actually recording. Reuses the mic-device-switched
                // listener the disconnect path wires up.
                let _ = app.emit(
                    "mic-device-switched",
                    serde_json::json!({ "device_name": device.name }),
                );
            }
            Some(Arc::new(device))
        }
        Err(e) => {
            warn!("❌ No microphone available: {} — recording system audio only", e);
            let _ = app.emit("mic-unavailable", serde_json::json!({}));
            None
        }
    }
}

/// System-audio analog of `resolve_mic_or_default`: `Some(name)` -> parse it,
/// falling back to the default output if unparseable; `None` ("Default System
/// Audio" in the UI) -> default output. Returns `None` only when no output
/// device exists — system audio is optional, mic-only recording proceeds.
///
/// ponytail: no cpal enumeration check (unlike the mic helper) — Linux system
/// devices are Pulse/ALSA monitor *inputs* tagged Output, so output_devices()
/// would false-negative them. stream.rs still hard-fails on a missing device.
///
/// When nothing resolves (e.g. Linux with no monitor source configured) the
/// user is told with `system-audio-unavailable` that participants' audio will
/// not be recorded.
fn resolve_system_or_default<R: Runtime>(
    app: &AppHandle<R>,
    requested_name: Option<&str>,
) -> Option<Arc<super::AudioDevice>> {
    if let Some(name) = requested_name {
        match parse_audio_device(name) {
            Ok(device) => {
                info!("✅ Using requested system audio: '{}'", device.name);
                return Some(Arc::new(device));
            }
            Err(e) => warn!(
                "⚠️ Requested system audio '{}' not available: {} — falling back to system default",
                name, e
            ),
        }
    }

    match default_output_device() {
        Ok(device) => {
            info!("✅ Using default system audio: '{}'", device.name);
            Some(Arc::new(device))
        }
        Err(e) => {
            warn!("⚠️ No system audio available: {} — recording will continue with microphone only", e);
            emit_stream_health_event(
                app,
                StreamHealthEvent::SystemAudioUnavailable {
                    device_name: None,
                    reason: format!("No system audio device found: {}", e),
                },
            );
            None
        }
    }
}

/// Wake idle audio hardware before checking microphone callbacks, and finish
/// validation before creating any recording resources.
#[cfg(target_os = "macos")]
async fn prepare_audio_for_recording(
    system_device: Option<&super::AudioDevice>,
) -> Result<(), String> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let wake_name = system_device
        .map(|s| s.name.clone())
        .or_else(|| {
            cpal::default_host()
                .default_output_device()
                .and_then(|d| d.name().ok())
        });
    if let Some(name) = wake_name {
        if let Err(e) = super::recording_manager::wake_audio_connection(&name).await {
            warn!("[AUDIO_WAKE] Wake failed: {} — proceeding anyway", e);
        }
    }

    if let Err(e) = super::devices::verify_microphone_access().await {
        error!("Microphone access verification failed: {}", e);
        return Err(format!("Microphone access required: {}", e));
    }
    Ok(())
}

// ============================================================================
// RECORDING COMMANDS
// ============================================================================

/// Start recording with default devices
pub async fn start_recording<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    start_recording_with_meeting_name(app, None).await
}

/// Start recording with default devices and optional meeting name
pub async fn start_recording_with_meeting_name<R: Runtime>(
    app: AppHandle<R>,
    meeting_name: Option<String>,
) -> Result<(), String> {
    info!(
        "Starting recording with default devices, meeting: {:?}",
        meeting_name
    );

    let engine_lifecycle_guard = super::common::acquire_engine_lifecycle_lock().await;

    info!("🔍 IS_RECORDING state check: {}", IS_RECORDING.load(Ordering::SeqCst));
    if let Some(reason) = start_blocker() {
        return Err(reason.to_string());
    }

    if let Err(error) = crate::ensure_onnx_runtime_available() {
        return Err(map_recording_start_error(
            &app,
            RecordingStartError::TranscriptionRuntime(error),
        ));
    }

    // Validate that transcription models are available before starting recording
    info!("🔍 Validating transcription model availability before starting recording...");
    if let Err(validation_error) = transcription::validate_transcription_model_ready(&app).await {
        error!("Model validation failed: {}", validation_error);

        // Emit error event for frontend - actionable: false to show toast instead of modal
        // (download progress is already shown in top-right toast)
        let _ = app.emit("transcription-error", serde_json::json!({
            "error": validation_error,
            "userMessage": format!("Recording cannot start: {}", validation_error),
            "actionable": false,
            "phase": "startup"
        }));

        return Err(validation_error);
    }
    info!("✅ Transcription model validation passed");

    // Only now that this start can proceed: stop a previous recording's background transcription.
    abandon_lingering_drain(&app).await;

    // Notify frontend that startup has begun (surfaces STARTING state)
    app.emit("recording-starting", serde_json::json!({
        "message": "Recording initialization started"
    })).map_err(|e| e.to_string())?;

    // Load recording preferences to get auto_save AND device preferences
    let (auto_save, preferred_mic_name, preferred_system_name) =
        match super::recording_preferences::load_recording_preferences(&app).await {
            Ok(prefs) => {
                info!("📋 Loaded recording preferences: auto_save={}, preferred_mic={:?}, preferred_system={:?}",
                      prefs.auto_save, prefs.preferred_mic_device, prefs.preferred_system_device);
                (prefs.auto_save, prefs.preferred_mic_device, prefs.preferred_system_device)
            }
            Err(e) => {
                warn!("Failed to load recording preferences, using defaults: {}", e);
                (true, None, None)
            }
        };

    #[cfg(not(target_os = "macos"))]
    let microphone_device = resolve_mic_or_default(&app, preferred_mic_name.as_deref());

    let system_device = resolve_system_or_default(&app, preferred_system_name.as_deref());

    #[cfg(target_os = "macos")]
    prepare_audio_for_recording(system_device.as_deref()).await?;

    #[cfg(target_os = "macos")]
    let microphone_device = resolve_mic_or_default(&app, preferred_mic_name.as_deref());

    // Async-first approach - no more blocking operations!
    info!("🚀 Starting async recording initialization");

    // Create new recording manager only after startup validation succeeds
    let mut manager = RecordingManager::new();

    // Always ensure a meeting name is set so incremental saver initializes
    let effective_meeting_name = meeting_name.clone().unwrap_or_else(|| {
        // Example: Meeting 2025-10-03_08-25-23
        let now = chrono::Local::now();
        format!(
            "Meeting {}",
            now.format("%Y-%m-%d_%H-%M-%S")
        )
    });
    manager.set_meeting_name(Some(effective_meeting_name));

    // Report stream health changes (degraded/recovered/unavailable/fatal) to the UI
    let app_for_health = app.clone();
    manager.set_health_callback(move |event| emit_stream_health_event(&app_for_health, event));

    // Start recording with resolved devices (replaces start_recording_with_defaults_and_auto_save call)
    let transcription_receiver = manager
        .start_recording(microphone_device, system_device, auto_save)
        .await
        .map_err(|error| map_recording_start_error(&app, error))?;

    activate_recording(&app, manager, transcription_receiver, auto_save).await?;
    drop(engine_lifecycle_guard);

    // Emit success event
    app.emit("recording-started", serde_json::json!({
        "message": "Recording started successfully with parallel processing",
        "devices": ["Default Microphone", "Default System Audio"],
        "workers": transcription::TRANSCRIPTION_WORKERS
    })).map_err(|e| e.to_string())?;

    // Update tray menu to reflect recording state
    crate::tray::update_tray_menu(&app);

    info!("✅ Recording started successfully with async-first approach");

    Ok(())
}

/// Start recording with specific devices
pub async fn start_recording_with_devices<R: Runtime>(
    app: AppHandle<R>,
    mic_device_name: Option<String>,
    system_device_name: Option<String>,
) -> Result<(), String> {
    start_recording_with_devices_and_meeting(app, mic_device_name, system_device_name, None).await
}

/// Start recording with specific devices and optional meeting name
pub async fn start_recording_with_devices_and_meeting<R: Runtime>(
    app: AppHandle<R>,
    mic_device_name: Option<String>,
    system_device_name: Option<String>,
    meeting_name: Option<String>,
) -> Result<(), String> {
    info!(
        "Starting recording with specific devices: mic={:?}, system={:?}, meeting={:?}",
        mic_device_name, system_device_name, meeting_name
    );

    let engine_lifecycle_guard = super::common::acquire_engine_lifecycle_lock().await;

    info!("🔍 IS_RECORDING state check: {}", IS_RECORDING.load(Ordering::SeqCst));
    if let Some(reason) = start_blocker() {
        return Err(reason.to_string());
    }

    if let Err(error) = crate::ensure_onnx_runtime_available() {
        return Err(map_recording_start_error(
            &app,
            RecordingStartError::TranscriptionRuntime(error),
        ));
    }

    // Validate that transcription models are available before starting recording
    info!("🔍 Validating transcription model availability before starting recording...");
    if let Err(validation_error) = transcription::validate_transcription_model_ready(&app).await {
        error!("Model validation failed: {}", validation_error);

        // Emit error event for frontend - actionable: false to show toast instead of modal
        // (download progress is already shown in top-right toast)
        let _ = app.emit("transcription-error", serde_json::json!({
            "error": validation_error,
            "userMessage": format!("Recording cannot start: {}", validation_error),
            "actionable": false,
            "phase": "startup"
        }));

        return Err(validation_error);
    }
    info!("✅ Transcription model validation passed");

    // Only now that this start can proceed: stop a previous recording's background transcription.
    abandon_lingering_drain(&app).await;

    // Notify frontend that startup has begun (surfaces STARTING state)
    app.emit("recording-starting", serde_json::json!({
        "message": "Recording initialization started"
    })).map_err(|e| e.to_string())?;

    #[cfg(not(target_os = "macos"))]
    let mic_device = resolve_mic_or_default(&app, mic_device_name.as_deref());

    let system_device = resolve_system_or_default(&app, system_device_name.as_deref());

    #[cfg(target_os = "macos")]
    prepare_audio_for_recording(system_device.as_deref()).await?;

    #[cfg(target_os = "macos")]
    let mic_device = resolve_mic_or_default(&app, mic_device_name.as_deref());

    // Async-first approach for custom devices - no more blocking operations!
    info!("🚀 Starting async recording initialization with custom devices");

    // Create new recording manager
    let mut manager = RecordingManager::new();

    // Load recording preferences to check auto_save setting
    let auto_save = match super::recording_preferences::load_recording_preferences(&app).await {
        Ok(prefs) => {
            info!("📋 Loaded recording preferences: auto_save={}", prefs.auto_save);
            prefs.auto_save
        }
        Err(e) => {
            warn!("Failed to load recording preferences, defaulting to auto_save=true: {}", e);
            true // Default to saving if preferences can't be loaded
        }
    };

    // Always ensure a meeting name is set so incremental saver initializes
    let effective_meeting_name = meeting_name.clone().unwrap_or_else(|| {
        let now = chrono::Local::now();
        format!(
            "Meeting {}",
            now.format("%Y-%m-%d_%H-%M-%S")
        )
    });
    manager.set_meeting_name(Some(effective_meeting_name));

    // Report stream health changes (degraded/recovered/unavailable/fatal) to the UI
    let app_for_health = app.clone();
    manager.set_health_callback(move |event| emit_stream_health_event(&app_for_health, event));

    // Start recording with specified devices and auto_save setting
    let transcription_receiver = manager
        .start_recording(mic_device, system_device, auto_save)
        .await
        .map_err(|error| map_recording_start_error(&app, error))?;

    activate_recording(&app, manager, transcription_receiver, auto_save).await?;
    drop(engine_lifecycle_guard);

    // Emit success event
    app.emit("recording-started", serde_json::json!({
        "message": "Recording started with custom devices and parallel processing",
        "devices": [
            mic_device_name.unwrap_or_else(|| "Default Microphone".to_string()),
            system_device_name.unwrap_or_else(|| "Default System Audio".to_string())
        ],
        "workers": transcription::TRANSCRIPTION_WORKERS
    })).map_err(|e| e.to_string())?;

    // Update tray menu to reflect recording state
    crate::tray::update_tray_menu(&app);

    info!("✅ Recording started with custom devices using async-first approach");

    Ok(())
}

/// Stop the recording: stop capture, let transcription finish, save, then report.
///
/// Only one stop runs at a time. A call made while another stop holds the guard
/// returns `Err(STOP_IN_PROGRESS_ERROR)` and does nothing else. The winning call
/// emits `recording-stopping`; if it then fails, it emits `recording-stop-failed`
/// and leaves the recording active so it can be stopped again.
pub async fn stop_recording<R: Runtime>(
    app: AppHandle<R>,
    _args: RecordingArgs,
    source: StopSource,
) -> Result<(), String> {
    info!("🛑 Stop requested from {}", source.as_str());

    // Check if recording is active
    if !IS_RECORDING.load(Ordering::SeqCst) {
        info!("Recording was not active");
        return Ok(());
    }

    let Some(_stop_guard) = StopGuard::try_acquire() else {
        info!("Stop from {} ignored: another stop is already in progress", source.as_str());
        return Err(STOP_IN_PROGRESS_ERROR.to_string());
    };
    // A stop that completed between the check above and winning the guard.
    if !IS_RECORDING.load(Ordering::SeqCst) {
        info!("Recording was stopped by a concurrent stop");
        return Ok(());
    }

    if let Err(e) = app.emit("recording-stopping", serde_json::json!({ "source": source.as_str() })) {
        warn!("Failed to emit recording-stopping: {}", e);
    }
    crate::tray::set_tray_state(&app, crate::tray::RecordingState::Stopping);

    let result = stop_recording_tail(&app).await;
    if let Err(ref message) = result {
        error!("❌ Stop failed: {}", message);
        if let Err(e) = app.emit("recording-stop-failed", serde_json::json!({ "message": message })) {
            warn!("Failed to emit recording-stop-failed: {}", e);
        }
        crate::tray::update_tray_menu(&app);
    }
    result
}

/// Everything after the stop guard is won. Runs with `STOP_IN_PROGRESS` held.
async fn stop_recording_tail<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    // Emit shutdown progress to frontend
    let _ = app.emit(
        "recording-shutdown-progress",
        serde_json::json!({
            "stage": "stopping_audio",
            "message": "Stopping audio capture...",
            "progress": 20
        }),
    );

    // Step 1: Stop audio capture immediately (no more new chunks)
    let taken_manager = RECORDING_MANAGER.lock().unwrap().take();
    let manager_for_cleanup = match taken_manager {
        Some(mut manager) => {
            // Use FORCE FLUSH to immediately process all accumulated audio - eliminates 30s delay!
            info!("🚀 Using FORCE FLUSH to eliminate pipeline accumulation delays");
            if let Err(e) = manager.stop_streams_and_force_flush().await {
                // Put the manager back: the recording is still active and can be stopped again.
                *RECORDING_MANAGER.lock().unwrap() = Some(manager);
                return Err(format!("Failed to stop audio streams: {}", e));
            }
            info!("✅ Audio streams stopped successfully - no more chunks will be created");
            Some(manager)
        }
        None => {
            warn!("No recording manager found to stop");
            None
        }
    };

    // Step 2: Let transcription finish the chunks already queued
    drain_transcription(app).await;

    // Step 3.5: Track meeting ended analytics with privacy-safe metadata
    // Extract all data from manager BEFORE any async operations to avoid Send issues
    let analytics_data = if let Some(ref manager) = manager_for_cleanup {
        let state = manager.get_state();
        let stats = state.get_stats();

        Some((
            manager.get_recording_duration(),
            manager.get_active_recording_duration().unwrap_or(0.0),
            manager.get_total_pause_duration(),
            manager.get_transcript_segments().len() as u64,
            state.has_fatal_error(),
            state.get_microphone_device().map(|d| d.name.clone()),
            state.get_system_device().map(|d| d.name.clone()),
            stats.chunks_processed,
        ))
    } else {
        None
    };

    // Now perform async analytics tracking without holding manager reference
    if let Some((total_duration, active_duration, pause_duration, transcript_segments_count, had_fatal_error, mic_device_name, sys_device_name, chunks_processed)) = analytics_data {
        info!("📊 Collecting analytics for meeting end");

        // Helper function to classify device type from device name (privacy-safe)
        fn classify_device_type(device_name: &str) -> &'static str {
            let name_lower = device_name.to_lowercase();
            // Check for Bluetooth keywords
            if name_lower.contains("bluetooth")
                || name_lower.contains("airpods")
                || name_lower.contains("beats")
                || name_lower.contains("headphones")
                || name_lower.contains("bt ")
                || name_lower.contains("wireless") {
                "Bluetooth"
            } else {
                "Wired"
            }
        }

        // Get transcription model info (already loaded above for model unload)
        let transcription_config = match crate::api::api::api_get_transcript_config(
            app.clone(),
            app.clone().state(),
            None,
        )
        .await
        {
            Ok(Some(config)) => Some((config.provider, config.model)),
            _ => None,
        };

        let (transcription_provider, transcription_model) = transcription_config
            .unwrap_or_else(|| ("unknown".to_string(), "unknown".to_string()));

        // Get summary model info from API
        let summary_config = match crate::api::api::api_get_model_config(
            app.clone(),
            app.clone().state(),
            None,
        )
        .await
        {
            Ok(Some(config)) => Some((config.provider, config.model)),
            _ => None,
        };

        let (summary_provider, summary_model) = summary_config
            .unwrap_or_else(|| ("unknown".to_string(), "unknown".to_string()));

        // Classify device types (privacy-safe)
        let microphone_device_type = mic_device_name
            .as_ref()
            .map(|name| classify_device_type(name))
            .unwrap_or("Unknown");

        let system_audio_device_type = sys_device_name
            .as_ref()
            .map(|name| classify_device_type(name))
            .unwrap_or("Unknown");

        // Track meeting ended event with privacy-safe data
        match crate::analytics::commands::track_meeting_ended(
            transcription_provider.clone(),
            transcription_model.clone(),
            summary_provider.clone(),
            summary_model.clone(),
            total_duration,
            active_duration,
            pause_duration,
            microphone_device_type.to_string(),
            system_audio_device_type.to_string(),
            chunks_processed,
            transcript_segments_count,
            had_fatal_error,
        )
        .await
        {
            Ok(_) => info!("✅ Analytics tracked successfully for meeting end"),
            Err(e) => warn!("⚠️ Failed to track analytics: {}", e),
        }
    }

    // Step 4: Finalize recording state and cleanup resources safely
    let _ = app.emit(
        "recording-shutdown-progress",
        serde_json::json!({
            "stage": "finalizing",
            "message": "Finalizing recording and cleaning up resources...",
            "progress": 90
        }),
    );

    // Perform final cleanup with the manager if available
    let (meeting_folder, meeting_name) = if let Some(mut manager) = manager_for_cleanup {
        info!("🧹 Performing final cleanup and saving recording data");

        // Extract meeting info BEFORE async operations
        let meeting_folder = manager.get_meeting_folder();
        let meeting_name = manager.get_meeting_name();

        // Checkpoints are 30 s each; the final encode takes time proportional to them.
        let (checkpoints, _) = manager.get_recording_stats();
        let save_timeout = save_timeout_for(checkpoints as u64 * 30);
        match tokio::time::timeout(save_timeout, manager.save_recording_only(app)).await {
            Ok(Ok(_)) => {
                info!("✅ Recording data saved successfully during cleanup");
            }
            Ok(Err(e)) => {
                warn!(
                    "⚠️ Error during recording cleanup (transcripts preserved): {}",
                    e
                );
                // Don't fail shutdown - transcripts are already preserved
            }
            Err(_) => {
                warn!("⏱️ Save timeout ({}s) reached, continuing shutdown", save_timeout.as_secs());
                let location = meeting_folder
                    .as_ref()
                    .map(|folder| folder.join(".checkpoints").display().to_string())
                    .unwrap_or_else(|| "the meeting folder's .checkpoints folder".to_string());
                let _ = app.emit(
                    "recording-save-error",
                    serde_json::json!({
                        "message": format!(
                            "Encoding the meeting audio took too long and was stopped, so the meeting has no audio file. The recorded audio is kept as WAV files in {}.",
                            location
                        )
                    }),
                );
            }
        }

        (meeting_folder, meeting_name)
    } else {
        info!("ℹ️ No recording manager available for cleanup");
        (None, None)
    };

    // Set recording flag to false. STOP_IN_PROGRESS is released when the caller's guard drops.
    info!("🔍 Setting IS_RECORDING to false");
    *LAST_COMPLETED_STOP.lock().unwrap_or_else(|e| e.into_inner()) = Some(CompletedStop {
        at: std::time::Instant::now(),
        saved_meetings: crate::database::repositories::transcript::saved_meeting_count(),
    });
    IS_RECORDING.store(false, Ordering::SeqCst);

    // Step 4.5: Prepare metadata for frontend (NO database save)
    // NOTE: We do NOT save to database here. The frontend will save after all transcripts are displayed.
    // This ensures the user sees all transcripts streaming in before the database save happens.
    let (folder_path_str, meeting_name_str) = match (&meeting_folder, &meeting_name) {
        (Some(path), Some(name)) => (
            Some(path.to_string_lossy().to_string()),
            Some(name.clone()),
        ),
        _ => (None, None),
    };

    info!("📤 Preparing recording metadata for frontend save");
    info!("   folder_path: {:?}", folder_path_str);
    info!("   meeting_name: {:?}", meeting_name_str);

    // Step 5: Complete shutdown
    let _ = app.emit(
        "recording-shutdown-progress",
        serde_json::json!({
            "stage": "complete",
            "message": "Recording stopped successfully",
            "progress": 100
        }),
    );

    // Emit final stop event with folder_path and meeting_name for frontend to save
    if let Err(e) = app.emit(
        "recording-stopped",
        serde_json::json!({
            "message": "Recording stopped - frontend will save after all transcripts received",
            "folder_path": folder_path_str,
            "meeting_name": meeting_name_str
        }),
    ) {
        error!("Failed to emit recording-stopped: {}", e);
    }

    // Update tray menu to reflect stopped state
    crate::tray::update_tray_menu(app);

    info!("🎉 Recording stopped");
    Ok(())
}

/// Wait (bounded) for the transcription task to finish the queued chunks, then
/// remove the transcript listener and unload the model.
///
/// The model is never unloaded under a running task. If the drain times out,
/// the task keeps running in the background (a finisher unloads the model when
/// it ends) and `transcript-chunk-loss-detected` reports the chunks that will
/// not be in the meeting the frontend is about to save.
async fn drain_transcription<R: Runtime>(app: &AppHandle<R>) {
    use tauri::Listener;

    let _ = app.emit(
        "recording-shutdown-progress",
        serde_json::json!({
            "stage": "processing_transcripts",
            "message": "Processing remaining transcript chunks...",
            "progress": 40
        }),
    );

    let listener = TRANSCRIPT_LISTENER.lock().unwrap().take();
    let Some(TranscriptionTask { handle: mut task_handle, progress, cancel, .. }) =
        TRANSCRIPTION_TASK.lock().unwrap().take()
    else {
        info!("ℹ️ No transcription task found to wait for");
        if let Some(listener) = listener {
            close_transcript_listener(listener, |id| app.unlisten(id)).await;
        }
        return;
    };

    info!("⏳ Waiting for queued transcription chunks ({}s max)", TRANSCRIPTION_DRAIN_TIMEOUT.as_secs());
    let progress_app = app.clone();
    let monitored = progress.clone();
    let progress_task = tokio::spawn(async move {
        let started = std::time::Instant::now();
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            let elapsed = started.elapsed().as_secs();
            let pending = monitored.snapshot().pending();
            let _ = progress_app.emit(
                "recording-shutdown-progress",
                serde_json::json!({
                    "stage": "processing_transcripts",
                    "message": format!("Processing transcripts... ({} left, {}s elapsed)", pending, elapsed),
                    "progress": 40,
                    "detailed": true,
                    "elapsed_seconds": elapsed,
                    "chunks_remaining": pending
                }),
            );
        }
    });

    let drained = tokio::time::timeout(TRANSCRIPTION_DRAIN_TIMEOUT, &mut task_handle).await;
    progress_task.abort();

    match drained {
        Ok(join_result) => {
            if let Err(e) = join_result {
                warn!("⚠️ Transcription task ended with error: {:?}", e);
            }
            // Every transcript-update has been emitted (listeners run inside emit).
            if let Some(listener) = listener {
                close_transcript_listener(listener, |id| app.unlisten(id)).await;
            }
            let snapshot = progress.snapshot();
            info!(
                "✅ Transcription finished: {}/{} chunks transcribed",
                snapshot.completed, snapshot.queued
            );
            if snapshot.not_transcribed() > 0 {
                emit_chunk_loss(
                    app,
                    &snapshot,
                    format!(
                        "{} of {} speech segments could not be transcribed.",
                        snapshot.not_transcribed(),
                        snapshot.queued
                    ),
                );
            }

            let _ = app.emit(
                "recording-shutdown-progress",
                serde_json::json!({
                    "stage": "unloading_model",
                    "message": "Unloading speech recognition model...",
                    "progress": 70
                }),
            );
            unload_transcription_engine(app, progress.engine_kind()).await;
        }
        Err(_) => {
            let snapshot = progress.snapshot();
            warn!(
                "⏱️ Transcription still running after {}s ({} chunks left); saving now and finishing in the background",
                TRANSCRIPTION_DRAIN_TIMEOUT.as_secs(),
                snapshot.pending()
            );
            emit_chunk_loss(
                app,
                &snapshot,
                format!(
                    "Transcription could not keep up: {} of {} speech segments were not transcribed before the meeting was saved. They are still being transcribed in the background and will be added to transcripts.json in the meeting folder, unless a new recording starts first.",
                    snapshot.not_transcribed(),
                    snapshot.queued
                ),
            );
            finish_transcription_in_background(app, task_handle, progress, cancel, listener);
        }
    }
}

fn emit_chunk_loss<R: Runtime>(
    app: &AppHandle<R>,
    snapshot: &transcription::ProgressSnapshot,
    message: String,
) {
    warn!("{}", message);
    if let Err(e) = app.emit(
        "transcript-chunk-loss-detected",
        serde_json::json!({
            "chunks_queued": snapshot.queued,
            "chunks_completed": snapshot.completed,
            "chunks_lost": snapshot.not_transcribed(),
            "message": message
        }),
    ) {
        error!("Failed to emit transcript-chunk-loss-detected: {}", e);
    }
}

/// Let a transcription task that outlived the stop drain run to completion,
/// then close its listener (writing the late segments to transcripts.json) and
/// unload the model unless a new recording is using it by then. A new
/// recording abandons it via `abandon_lingering_drain`.
fn finish_transcription_in_background<R: Runtime>(
    app: &AppHandle<R>,
    task_handle: JoinHandle<()>,
    progress: Arc<TranscriptionProgress>,
    cancel: CancellationToken,
    listener: Option<TranscriptListener>,
) {
    use tauri::Listener;

    let transcription = task_handle.abort_handle();
    let app = app.clone();
    // Hold the slot while spawning so the finisher cannot look for its entry before it exists.
    let mut slot = LINGERING_DRAIN.lock().unwrap_or_else(|e| e.into_inner());
    let finisher = tokio::spawn(async move {
        if let Err(e) = task_handle.await {
            warn!("Background transcription ended with error: {:?}", e);
        }
        // The stop that spawned this is still saving and holds the stop guard;
        // judge "is a recording using the model" only once it has finished.
        wait_for_stop_to_finish().await;
        // Serialized with starts, which abandon this drain under the same lock.
        let _engine_lifecycle_guard = super::common::acquire_engine_lifecycle_lock().await;
        let drain = LINGERING_DRAIN.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(listener) = drain.and_then(|d| d.listener) {
            close_transcript_listener(listener, |id| app.unlisten(id)).await;
        }

        let snapshot = progress.snapshot();
        info!(
            "Background transcription finished: {}/{} chunks transcribed",
            snapshot.completed, snapshot.queued
        );
        if transcription_engine_idle() {
            unload_transcription_engine(&app, progress.engine_kind()).await;
        } else {
            info!("Keeping the transcription model loaded: a recording is using it");
        }
    });
    *slot = Some(LingeringDrain {
        finisher,
        transcription,
        cancel,
        listener,
    });
}

/// Wait until no stop is running.
async fn wait_for_stop_to_finish() {
    while STOP_IN_PROGRESS.load(Ordering::SeqCst) {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

/// No recording is live or stopping, so nothing is transcribing with the model.
fn transcription_engine_idle() -> bool {
    !IS_RECORDING.load(Ordering::SeqCst) && !STOP_IN_PROGRESS.load(Ordering::SeqCst)
}

/// Unload the model the stopped recording used. Falls back to the saved
/// transcript config when the engine never started.
async fn unload_transcription_engine<R: Runtime>(app: &AppHandle<R>, engine_kind: Option<EngineKind>) {
    let use_parakeet = match engine_kind {
        Some(EngineKind::Parakeet) => true,
        Some(EngineKind::Whisper) => false,
        Some(EngineKind::Other) | None => {
            let provider = match tokio::time::timeout(
                tokio::time::Duration::from_secs(30), // 30 seconds max for DB operation
                crate::api::api::api_get_transcript_config(app.clone(), app.clone().state(), None),
            )
            .await
            {
                Ok(Ok(Some(config))) => Some(config.provider),
                Ok(Ok(None)) => None,
                Ok(Err(e)) => {
                    warn!("⚠️ Failed to get transcript config: {:?}", e);
                    None
                }
                Err(_) => {
                    warn!("⏱️ Transcript config timeout (30s), continuing shutdown");
                    None
                }
            };
            provider.as_deref() == Some("parakeet")
        }
    };

    if use_parakeet {
        info!("🦜 Unloading Parakeet model...");
        let engine_clone = crate::parakeet_engine::commands::PARAKEET_ENGINE
            .lock()
            .unwrap()
            .as_ref()
            .cloned();
        match engine_clone {
            Some(engine) if engine.unload_model().await => info!("✅ Parakeet model unloaded"),
            Some(_) => warn!("⚠️ Parakeet model was not loaded"),
            None => warn!("⚠️ No Parakeet engine found to unload model"),
        }
    } else {
        info!("🎤 Unloading Whisper model...");
        let engine_clone = crate::whisper_engine::commands::WHISPER_ENGINE
            .lock()
            .unwrap()
            .as_ref()
            .cloned();
        match engine_clone {
            Some(engine) if engine.unload_model().await => info!("✅ Whisper model unloaded"),
            Some(_) => warn!("⚠️ Whisper model was not loaded"),
            None => warn!("⚠️ No Whisper engine found to unload model"),
        }
    }
}

/// Check if recording is active
pub async fn is_recording() -> bool {
    IS_RECORDING.load(Ordering::SeqCst)
}

/// A stop is running (the recording is still active until it finishes).
pub fn is_stopping() -> bool {
    STOP_IN_PROGRESS.load(Ordering::SeqCst)
}

/// Transcription queue status for the most recent recording (live or stopping).
pub fn get_transcription_status() -> TranscriptionStatus {
    let progress = TRANSCRIPTION_PROGRESS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    match progress {
        Some(progress) => TranscriptionStatus {
            chunks_in_queue: progress.snapshot().pending() as usize,
            is_processing: !progress.is_finished(),
            last_activity_ms: progress.idle_for().as_millis() as u64,
        },
        None => TranscriptionStatus {
            chunks_in_queue: 0,
            is_processing: false,
            last_activity_ms: 0,
        },
    }
}

/// Pause the current recording
#[tauri::command]
pub async fn pause_recording<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    info!("Pausing recording");

    // Check if currently recording
    if !IS_RECORDING.load(Ordering::SeqCst) {
        return Err("No recording is currently active".to_string());
    }

    // Access the recording manager and pause it
    let manager_guard = RECORDING_MANAGER.lock().unwrap();
    if let Some(manager) = manager_guard.as_ref() {
        manager.pause_recording().map_err(|e| e.to_string())?;

        // Emit pause event to frontend
        app.emit(
            "recording-paused",
            serde_json::json!({
                "message": "Recording paused"
            }),
        )
        .map_err(|e| e.to_string())?;

        // Update tray menu to reflect paused state
        crate::tray::update_tray_menu(&app);

        info!("Recording paused successfully");
        Ok(())
    } else {
        Err("No recording manager found".to_string())
    }
}

/// Resume the current recording
#[tauri::command]
pub async fn resume_recording<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    info!("Resuming recording");

    // Check if currently recording
    if !IS_RECORDING.load(Ordering::SeqCst) {
        return Err("No recording is currently active".to_string());
    }

    // Access the recording manager and resume it
    let manager_guard = RECORDING_MANAGER.lock().unwrap();
    if let Some(manager) = manager_guard.as_ref() {
        manager.resume_recording().map_err(|e| e.to_string())?;

        // Emit resume event to frontend
        app.emit(
            "recording-resumed",
            serde_json::json!({
                "message": "Recording resumed"
            }),
        )
        .map_err(|e| e.to_string())?;

        // Update tray menu to reflect resumed state
        crate::tray::update_tray_menu(&app);

        info!("Recording resumed successfully");
        Ok(())
    } else {
        Err("No recording manager found".to_string())
    }
}

/// Check if recording is currently paused
#[tauri::command]
pub async fn is_recording_paused() -> bool {
    let manager_guard = RECORDING_MANAGER.lock().unwrap();
    if let Some(manager) = manager_guard.as_ref() {
        manager.is_paused()
    } else {
        false
    }
}

/// Get detailed recording state
#[tauri::command]
pub async fn get_recording_state() -> serde_json::Value {
    let is_recording = IS_RECORDING.load(Ordering::SeqCst);
    let manager_guard = RECORDING_MANAGER.lock().unwrap();

    if let Some(manager) = manager_guard.as_ref() {
        serde_json::json!({
            "is_recording": is_recording,
            "is_paused": manager.is_paused(),
            "is_active": manager.is_active(),
            "recording_duration": manager.get_recording_duration(),
            "active_duration": manager.get_active_recording_duration(),
            "total_pause_duration": manager.get_total_pause_duration(),
            "current_pause_duration": manager.get_current_pause_duration()
        })
    } else {
        serde_json::json!({
            "is_recording": is_recording,
            "is_paused": false,
            "is_active": false,
            "recording_duration": null,
            "active_duration": null,
            "total_pause_duration": 0.0,
            "current_pause_duration": null
        })
    }
}

/// Get the meeting folder path for the current recording
/// Returns the path if a meeting name was set and folder structure initialized
#[tauri::command]
pub async fn get_meeting_folder_path() -> Result<Option<String>, String> {
    let manager_guard = RECORDING_MANAGER.lock().unwrap();
    if let Some(manager) = manager_guard.as_ref() {
        Ok(manager.get_meeting_folder().map(|p| p.to_string_lossy().to_string()))
    } else {
        Ok(None)
    }
}

/// Get accumulated transcript segments from current recording session
/// Used for syncing frontend state after page reload during active recording
#[tauri::command]
pub async fn get_transcript_history() -> Result<Vec<crate::audio::recording_saver::TranscriptSegment>, String> {
    let manager_guard = RECORDING_MANAGER.lock().unwrap();

    if let Some(manager) = manager_guard.as_ref() {
        Ok(manager.get_transcript_segments())
    } else {
        Ok(Vec::new()) // No recording active, return empty
    }
}

/// Get meeting name from current recording session
/// Used for syncing frontend state after page reload during active recording
#[tauri::command]
pub async fn get_recording_meeting_name() -> Result<Option<String>, String> {
    let manager_guard = RECORDING_MANAGER.lock().unwrap();

    if let Some(manager) = manager_guard.as_ref() {
        Ok(manager.get_meeting_name())
    } else {
        Ok(None)
    }
}

// ============================================================================
// DEVICE MONITORING COMMANDS (AirPods/Bluetooth disconnect/reconnect support)
// ============================================================================

/// Get information about the active audio output device
/// Used to warn users about Bluetooth playback issues
#[tauri::command]
pub async fn get_active_audio_output() -> Result<super::playback_monitor::AudioOutputInfo, String> {
    super::playback_monitor::get_active_audio_output()
        .await
        .map_err(|e| format!("Failed to get audio output info: {}", e))
}


// ============================================================================
// MIC HOT-SWAP (disconnect recovery)
// ============================================================================

// Guard against concurrent mic hot-swap tasks. Only used by the disconnect
// fallback path (trigger_mic_fallback_to_default) — the "chase the new
// default" auto-swap has been removed.
static MIC_SWAP_IN_PROGRESS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// Bounded retry budget for the disconnect fallback (P1 #2). Counts COMPLETED
// failed attempts; MIC_SWAP_IN_PROGRESS still prevents overlapping swaps.
static MIC_FALLBACK_FAILED_ATTEMPTS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
const MAX_MIC_FALLBACK_ATTEMPTS: u32 = 3;

/// Perform mic hot-swap using phased locking — never holds RECORDING_MANAGER during I/O
/// except the brief mic-stream stop in Phase 1.
/// If CPAL hangs during stream creation, only this task blocks; stop flow stays unblocked.
async fn perform_mic_hot_swap_task<R: Runtime>(
    new_device_name: String,
    session: &Arc<super::RecordingState>,
    app: AppHandle<R>,
) -> Result<(), String> {
    info!("[HOT_SWAP] Starting mic hot-swap to '{}'", new_device_name);

    match do_mic_swap(&new_device_name, session).await {
        Ok(()) => {
            info!("[HOT_SWAP] Mic switched to '{}'", new_device_name);
            let _ = app.emit("mic-device-switched", serde_json::json!({
                "device_name": new_device_name
            }));
            Ok(())
        }
        Err(e) => {
            if !session_live(session) {
                return Err(e);
            }
            warn!("[HOT_SWAP] First attempt failed: {} — retrying in 500ms", e);
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            match do_mic_swap(&new_device_name, session).await {
                Ok(()) => {
                    info!("[HOT_SWAP] Mic switched to '{}' on retry", new_device_name);
                    let _ = app.emit("mic-device-switched", serde_json::json!({
                        "device_name": new_device_name
                    }));
                    Ok(())
                }
                Err(e) => {
                    error!("[HOT_SWAP] Mic swap failed after retry: {}", e);
                    if session_live(session) {
                        let _ = app.emit("mic-swap-failed", serde_json::json!({
                            "error": e,
                            "device_name": new_device_name
                        }));
                    }
                    Err(e)
                }
            }
        }
    }
}

/// Phased mic swap to `device_name` — lock is never held during async I/O.
async fn do_mic_swap(device_name: &str, session: &Arc<super::RecordingState>) -> Result<(), String> {
    // Build the AudioDevice directly from the name — the caller
    // (trigger_mic_fallback_to_default) already resolved it via
    // default_input_device(). Skipping list_audio_devices() here avoids a
    // full cpal enumeration on the exact BT-transition hot path where it's
    // known to hang 100+ s (see H2 in PR-175 review). The real device
    // validation happens inside AudioStream::create → get_device_and_config.
    let device = Arc::new(super::AudioDevice::new(
        device_name.to_string(),
        super::DeviceType::Input,
    ));
    do_stream_swap(RecordingDeviceType::Microphone, device, session).await
}

/// Replace one capture stream of `session` with a fresh stream on `device`.
/// Used by the mic hot-swap and by dead-stream rebuilds. The other stream
/// keeps running; RECORDING_MANAGER is only held for the take and install.
async fn do_stream_swap(
    device_type: RecordingDeviceType,
    device: Arc<super::AudioDevice>,
    session: &Arc<super::RecordingState>,
) -> Result<(), String> {
    // Phase 1: Lock briefly — verify identity, take old stream OUT (no teardown under lock)
    let old_stream = {
        let mut guard = RECORDING_MANAGER.lock().unwrap();
        let manager = guard.as_mut().ok_or_else(|| "Recording manager not available".to_string())?;
        if !manager.is_recording() {
            return Err(format!("Recording stopped — aborting {:?} stream swap", device_type));
        }
        if !Arc::ptr_eq(manager.get_state(), session) {
            return Err("Session changed before stream swap — aborting".to_string());
        }
        manager.take_stream_for_rebuild(device_type)
    }; // lock released

    // Tear down the old stream OUTSIDE the lock and off the runtime — cpal
    // stop()/drop on a disconnected device can stall (CoreAudio HAL lock) or
    // panic (ALSA worker join); under RECORDING_MANAGER that would freeze
    // stop_recording (deep-review #2). A teardown failure must not abort the swap.
    if let Some(old) = old_stream {
        old.stop_off_runtime().await;
    }

    // Phase 2: Async I/O WITHOUT lock — may be slow, that's OK
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    info!("[HOT_SWAP] Creating new {:?} stream for '{}' (lock released)", device_type, device.name);
    let new_stream = super::stream::AudioStream::create(device.clone(), session.clone(), device_type, None)
        .await
        .map_err(|e| format!("Failed to create {:?} stream: {}", device_type, e))?;

    // Phase 3: Lock briefly — install ONLY if still the same session
    let install_result = {
        let mut guard = RECORDING_MANAGER.lock().unwrap();
        match guard.as_mut() {
            Some(manager) if Arc::ptr_eq(manager.get_state(), session) => {
                manager.install_rebuilt_stream(device_type, new_stream, device.clone());
                info!("[HOT_SWAP] {:?} stream on '{}' installed", device_type, device.name);
                Ok(())
            }
            Some(_) => Err((new_stream, "Session changed during stream swap — discarding stale stream")),
            None => Err((new_stream, "Recording manager gone during stream swap")),
        }
    }; // lock released

    install_result.map_err(|(stale, reason)| {
        // Dropping a cpal stream joins its worker; do it off the runtime.
        tokio::spawn(stale.stop_off_runtime());
        reason.to_string()
    })
}

// ============================================================================
// STREAM SUPERVISION (dead/stalled stream rebuild)
// ============================================================================

mod stream_supervisor;
use stream_supervisor::{RebuildRequest, SupervisorHost};

/// Guards against overlapping system-stream rebuilds. Mic rebuilds share
/// MIC_SWAP_IN_PROGRESS with the disconnect fallback so they never overlap.
static SYSTEM_REBUILD_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Map a stream health change onto its frontend event.
fn emit_stream_health_event<R: Runtime>(app: &AppHandle<R>, event: StreamHealthEvent) {
    let (name, payload) = stream_supervisor::frontend_event(event);
    if let Err(e) = app.emit(name, payload) {
        warn!("Failed to emit {}: {}", name, e);
    }
}

/// The running app as seen by the stream supervisor of one session.
struct AppSupervisorHost<R: Runtime> {
    app: AppHandle<R>,
    session: Arc<super::RecordingState>,
}

impl<R: Runtime> Clone for AppSupervisorHost<R> {
    fn clone(&self) -> Self {
        Self { app: self.app.clone(), session: self.session.clone() }
    }
}

impl<R: Runtime> SupervisorHost for AppSupervisorHost<R> {
    fn session_live(&self) -> bool {
        session_live(&self.session)
    }

    fn swap_stream(
        &self,
        device_type: RecordingDeviceType,
        device: Arc<super::AudioDevice>,
    ) -> impl std::future::Future<Output = Result<(), String>> + Send {
        let session = self.session.clone();
        async move { do_stream_swap(device_type, device, &session).await }
    }

    fn default_input_name(&self) -> Option<String> {
        default_input_device().ok().map(|d| d.name)
    }

    fn emit_mic_switched(&self, device_name: &str) {
        let _ = self.app.emit("mic-device-switched", serde_json::json!({ "device_name": device_name }));
    }

    fn emit_mic_recovery_exhausted(&self, device_name: &str) {
        let _ = self.app.emit("mic-recovery-exhausted", serde_json::json!({ "device_name": device_name }));
    }

    fn rebuild_flag(&self, device_type: RecordingDeviceType) -> &'static AtomicBool {
        match device_type {
            RecordingDeviceType::Microphone => &MIC_SWAP_IN_PROGRESS,
            RecordingDeviceType::System => &SYSTEM_REBUILD_IN_PROGRESS,
        }
    }

    fn sleep(&self, duration: std::time::Duration) -> impl std::future::Future<Output = ()> + Send {
        tokio::time::sleep(duration)
    }
}

/// Start a rebuild for `request` if the supervisor's policy says one is due.
fn start_rebuild_if_due<R: Runtime>(host: &AppSupervisorHost<R>, request: RebuildRequest) {
    let Some(ticket) = stream_supervisor::claim_rebuild(host, &host.session, &request, std::time::Instant::now())
    else {
        return;
    };
    let host = host.clone();
    tokio::spawn(async move {
        stream_supervisor::rebuild_stream(&host, &host.session, request, ticket).await;
    });
}

/// Background supervisor for one recording session.
///
/// Handles device monitor events, stream faults (a stream marked dead by its
/// error rate) and a watchdog tick (stall detection, recovery reports).
///
/// The ONLY mid-recording mic switches allowed are the fallback from a dead
/// device to the system default — triggered by the device monitor's
/// DeviceDisconnected event, or when rebuilding a dead mic stream on its own
/// device fails. Any other device event is explicitly ignored — recording
/// stays on whatever device was picked at start time until the meeting ends.
///
/// Rationale: auto-swapping to a freshly-connected BT device during recording
/// triggers a reliable hang inside cpal's stream creation on macOS. Locking
/// the device at start eliminates that hang and also makes the recording
/// session predictable.
///
/// The task stops when the device event channel closes (the session's
/// manager, which owns the device monitor, is dropped after stop).
fn spawn_device_event_processor<R: Runtime>(
    app: AppHandle<R>,
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<DeviceEvent>,
    session: Arc<super::RecordingState>,
) {
    let (fault_sender, mut fault_receiver) = tokio::sync::mpsc::unbounded_channel::<StreamFault>();
    session.set_fault_sender(fault_sender);
    let host = AppSupervisorHost { app: app.clone(), session: session.clone() };

    tokio::spawn(async move {
        info!("[DEVICE_EVENTS] Background event processor started");
        let mut watchdog = tokio::time::interval(stream_supervisor::WATCHDOG_INTERVAL);
        watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                event = receiver.recv() => {
                    let Some(event) = event else { break };
                    handle_device_event(&app, &session, event);
                }
                Some(fault) = fault_receiver.recv() => {
                    if session_live(&session) {
                        start_rebuild_if_due(&host, fault.into());
                    }
                }
                _ = watchdog.tick() => {
                    if session_live(&session) {
                        for request in stream_supervisor::watchdog_tick(&session, std::time::Instant::now()) {
                            start_rebuild_if_due(&host, request);
                        }
                    }
                }
            }
        }
        info!("[DEVICE_EVENTS] Background event processor stopped (channel closed)");
    });
}

fn handle_device_event<R: Runtime>(
    app: &AppHandle<R>,
    session: &Arc<super::RecordingState>,
    event: DeviceEvent,
) {
    // Skip if recording has stopped
    if !recording_live() {
        info!("[DEVICE_EVENTS] Recording stopped — ignoring event: {:?}", event);
        return;
    }

    match event {
        DeviceEvent::DeviceDisconnected { ref device_name, ref device_type } => {
            info!("[DEVICE_EVENTS] Device disconnected: '{}' ({:?})", device_name, device_type);
            // When the active microphone dies, fall back to the system default
            // input. Triggered after the device monitor's polling threshold fires.
            if matches!(device_type, DeviceMonitorType::Microphone) {
                let name = device_name.clone();
                let app_clone = app.clone();
                let session = session.clone();
                tokio::spawn(async move {
                    trigger_mic_fallback_to_default(app_clone, name, session).await;
                });
            }
        }
        DeviceEvent::DeviceReconnected { ref device_name, ref device_type } => {
            // Per product decision: once we have fallen back to the
            // built-in mic we stay there for the rest of the meeting.
            // This is intentional — just log and do nothing.
            info!("[DEVICE_EVENTS] Device reconnected: '{}' ({:?}) — staying on current mic (fallback is sticky)", device_name, device_type);
        }
        DeviceEvent::DeviceListChanged => {
            debug!("[DEVICE_EVENTS] Device list changed");
        }
    }
}

/// Disconnect fallback: swap the active mic to the system default input
/// device. Triggered from the background device event processor after the
/// device monitor's polling threshold (3 × 2s) fires `DeviceDisconnected`
/// for the active microphone.
///
/// `disconnected_name` is the device that just died. We keep it to detect
/// the edge case where macOS hasn't yet updated the system default input
/// away from the dead device — we wait and retry in that case rather than
/// swapping back to the same broken device.
///
/// This function takes the MIC_SWAP_IN_PROGRESS guard itself; the caller
/// must NOT already hold it. If a swap is somehow already running this
/// returns immediately.
async fn trigger_mic_fallback_to_default<R: Runtime>(
    app: AppHandle<R>,
    disconnected_name: String,
    session: Arc<super::RecordingState>,
) {
    if !session_live(&session) {
        info!(
            "[MIC_FALLBACK] Not recording — skipping fallback for '{}'",
            disconnected_name
        );
        return;
    }

    if MIC_FALLBACK_FAILED_ATTEMPTS.load(Ordering::SeqCst) >= MAX_MIC_FALLBACK_ATTEMPTS {
        warn!(
            "[MIC_FALLBACK] {} failed attempts reached — giving up on '{}' (terminal event already announced)",
            MAX_MIC_FALLBACK_ATTEMPTS, disconnected_name
        );
        return;
    }

    if MIC_SWAP_IN_PROGRESS
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        info!(
            "[MIC_FALLBACK] Swap already in progress — skipping fallback for '{}'",
            disconnected_name
        );
        return;
    }

    // Guard that clears MIC_SWAP_IN_PROGRESS on any return path below so a
    // panic or early return can't leave the flag stuck.
    struct SwapGuard;
    impl Drop for SwapGuard {
        fn drop(&mut self) {
            MIC_SWAP_IN_PROGRESS.store(false, Ordering::SeqCst);
        }
    }
    let _guard = SwapGuard;

    info!(
        "[MIC_FALLBACK] Starting fallback from disconnected device '{}'",
        disconnected_name
    );

    // Let macOS finish swapping the system default input away from the dead
    // device. 150ms is enough in practice for the built-in mic to become the
    // default when an explicitly-selected BT device disconnects.
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    // A Stop-A/Start-B during the sleep swapped our session out. Bail silently —
    // emitting or spending the recovery budget here would fire against B with
    // A's device. Covers the default_input_device() error branch below.
    if !session_live(&session) {
        info!("[MIC_FALLBACK] Session no longer live after wait — aborting fallback for '{}'", disconnected_name);
        return;
    }

    // Query the current system default input. If it still reports the
    // disconnected device, back off once more and re-query — this handles
    // the edge case where the OS hasn't propagated the change yet.
    let fallback_name = match default_input_device() {
        Ok(dev) => dev.name,
        Err(e) => {
            error!("[MIC_FALLBACK] Failed to query default input device: {}", e);
            let _ = app.emit(
                "mic-swap-failed",
                serde_json::json!({
                    "error": format!("Failed to query default input: {}", e),
                    "device_name": disconnected_name,
                }),
            );
            let n = MIC_FALLBACK_FAILED_ATTEMPTS.fetch_add(1, Ordering::SeqCst) + 1;
            if n == MAX_MIC_FALLBACK_ATTEMPTS {
                let _ = app.emit(
                    "mic-recovery-exhausted",
                    serde_json::json!({ "device_name": disconnected_name }),
                );
            }
            return;
        }
    };

    let fallback_name = if fallback_name == disconnected_name {
        warn!(
            "[MIC_FALLBACK] Default input still reports disconnected device '{}' — retrying after 300ms",
            disconnected_name
        );
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        if !session_live(&session) {
            info!("[MIC_FALLBACK] Session no longer live after retry wait — aborting fallback for '{}'", disconnected_name);
            return;
        }
        match default_input_device() {
            Ok(dev) if dev.name != disconnected_name => dev.name,
            Ok(dev) => {
                error!(
                    "[MIC_FALLBACK] Default input still '{}' after retry — aborting fallback",
                    dev.name
                );
                let _ = app.emit(
                    "mic-swap-failed",
                    serde_json::json!({
                        "error": "System default input still reports disconnected device after retry",
                        "device_name": disconnected_name,
                    }),
                );
                let n = MIC_FALLBACK_FAILED_ATTEMPTS.fetch_add(1, Ordering::SeqCst) + 1;
                if n == MAX_MIC_FALLBACK_ATTEMPTS {
                    let _ = app.emit(
                        "mic-recovery-exhausted",
                        serde_json::json!({ "device_name": disconnected_name }),
                    );
                }
                return;
            }
            Err(e) => {
                error!("[MIC_FALLBACK] Failed to re-query default input device: {}", e);
                let _ = app.emit(
                    "mic-swap-failed",
                    serde_json::json!({
                        "error": format!("Failed to re-query default input: {}", e),
                        "device_name": disconnected_name,
                    }),
                );
                let n = MIC_FALLBACK_FAILED_ATTEMPTS.fetch_add(1, Ordering::SeqCst) + 1;
                if n == MAX_MIC_FALLBACK_ATTEMPTS {
                    let _ = app.emit(
                        "mic-recovery-exhausted",
                        serde_json::json!({ "device_name": disconnected_name }),
                    );
                }
                return;
            }
        }
    } else {
        fallback_name
    };

    info!(
        "[MIC_FALLBACK] Falling back '{}' → '{}'",
        disconnected_name, fallback_name
    );

    // macOS Core Audio pre-wake for the hot-swap path — before we call the
    // rebuild path (which internally calls `AudioDeviceStart` on the new
    // mic), play 150ms of digital silence through the current system
    // output device to force the Core Audio hardware unit out of its idle
    // power state. Without this, `AudioDeviceStart` can return `noErr` but
    // the IO proc will not fire for 10-30 seconds until some other audio
    // nudges the hardware awake — the "backend idle until you play YouTube"
    // symptom from earlier testing.
    //
    // `wake_audio_connection_for_swap` has a built-in fallback: if the
    // current system device name doesn't enumerate (e.g. the BT output just
    // disappeared), it plays through `default_output_device()` instead,
    // which on macOS will now be the built-in speakers — exactly the
    // hardware unit we want to wake for the fallback mic.
    //
    // Non-fatal: on error we log and proceed to the swap anyway. A failed
    // wake is strictly better than no wake.
    #[cfg(target_os = "macos")]
    {
        // Read from the captured session directly (no manager lock) — this
        // stays correct even if the global manager has since been swapped by
        // a Stop/Start of a different session.
        let sys_device_name = session.get_system_device().map(|d| d.name.clone());
        if let Some(name) = sys_device_name {
            match super::recording_manager::wake_audio_connection_for_swap(&name).await {
                Ok(()) => info!("[MIC_FALLBACK] Pre-swap audio wake completed"),
                Err(e) => warn!(
                    "[MIC_FALLBACK] Pre-swap audio wake failed: {} — proceeding anyway",
                    e
                ),
            }
        } else {
            log::debug!("[MIC_FALLBACK] No system device recorded — skipping pre-swap wake");
        }
    }

    // Stop may have started during the sleeps above — bail before touching
    // the (possibly already taken) manager.
    if !session_live(&session) {
        info!("[MIC_FALLBACK] Recording stopping — aborting fallback for '{}'", disconnected_name);
        return;
    }

    // perform_mic_hot_swap_task performs its own retry-once logic on failure
    // and emits the mic-device-switched / mic-swap-failed events, so we can
    // just delegate here. It does NOT touch MIC_SWAP_IN_PROGRESS internally.
    match perform_mic_hot_swap_task(fallback_name.clone(), &session, app.clone()).await {
        Ok(()) => {
            info!(
                "[MIC_FALLBACK] Fallback complete: now recording via '{}'",
                fallback_name
            );
            MIC_FALLBACK_FAILED_ATTEMPTS.store(0, Ordering::SeqCst);
        }
        Err(e) => {
            error!("[MIC_FALLBACK] Fallback swap failed: {}", e);
            if !session_live(&session) {
                return;
            }
            let n = MIC_FALLBACK_FAILED_ATTEMPTS.fetch_add(1, Ordering::SeqCst) + 1;
            if n == MAX_MIC_FALLBACK_ATTEMPTS {
                let _ = app.emit(
                    "mic-recovery-exhausted",
                    serde_json::json!({ "device_name": disconnected_name }),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that set the global stop guard.
    static STOP_FLAG_TESTS: Mutex<()> = Mutex::new(());

    #[test]
    fn only_one_stop_holds_the_guard_and_starts_wait_for_it() {
        let _serial = STOP_FLAG_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        assert!(start_blocker().is_none());

        let contenders: Vec<_> = (0..8)
            .map(|_| std::thread::spawn(StopGuard::try_acquire))
            .collect();
        let winners: Vec<StopGuard> = contenders
            .into_iter()
            .filter_map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(winners.len(), 1, "exactly one concurrent stop wins");
        assert!(StopGuard::try_acquire().is_none(), "a later stop loses while one runs");
        assert!(start_blocker().unwrap().contains("still stopping"));
        assert!(transcription_engine_in_use());

        drop(winners);
        assert!(start_blocker().is_none(), "the guard is released when its stop returns");
        assert!(StopGuard::try_acquire().is_some());
    }

    #[tokio::test]
    async fn background_finisher_waits_for_its_own_stop_before_deciding_to_unload() {
        let _serial = STOP_FLAG_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let stop = StopGuard::try_acquire().expect("no other stop in this test");
        assert!(!transcription_engine_idle(), "the finishing stop still owns the engine");

        let waiter = tokio::spawn(wait_for_stop_to_finish());
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        assert!(!waiter.is_finished(), "must not decide while its own stop is saving");

        drop(stop);
        tokio::time::timeout(std::time::Duration::from_secs(2), waiter)
            .await
            .expect("stops waiting once the stop finished")
            .unwrap();
        assert!(transcription_engine_idle(), "then the model is free to unload");
    }

    #[test]
    fn save_timeout_grows_with_the_recording_it_encodes() {
        assert_eq!(save_timeout_for(0).as_secs(), 300);
        assert_eq!(save_timeout_for(2 * 3600).as_secs(), 300 + 1800);
    }

    fn update_payload(session_id: u64, sequence_id: u64) -> String {
        serde_json::json!({
            "text": "late words",
            "timestamp": "10:00:00",
            "source": "Audio",
            "sequence_id": sequence_id,
            "chunk_start_time": 1.0,
            "is_partial": false,
            "confidence": 0.9,
            "audio_start_time": 1.0,
            "audio_end_time": 2.0,
            "duration": 1.0,
            "session_id": session_id
        })
        .to_string()
    }

    #[test]
    fn a_recording_ignores_transcripts_from_another_transcription_session() {
        assert!(segment_for_session(&update_payload(7, 1), 7).is_some());
        assert!(segment_for_session(&update_payload(6, 2), 7).is_none());
        assert!(segment_for_session("not json", 7).is_none());
    }

    #[tokio::test]
    async fn a_segment_received_just_before_the_listener_closes_reaches_transcripts_json() {
        let folder = tempfile::tempdir().unwrap();
        let store = TranscriptStore::for_test(folder.path().to_path_buf());
        let listener = TranscriptListener { id: 7, store: Arc::clone(&store) };
        store.upsert(segment_for_session(&update_payload(1, 1), 1).unwrap());
        // The last transcript lands inside the debounce window, just before unlisten.
        store.upsert(segment_for_session(&update_payload(1, 2), 1).unwrap());
        drop(store); // the listener's reference is now the only strong one

        let mut unlistened = None;
        close_transcript_listener(listener, |id| unlistened = Some(id)).await;

        assert_eq!(unlistened, Some(7));
        let written: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(folder.path().join("transcripts.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(written["segments"].as_array().unwrap().len(), 2);
    }
}
