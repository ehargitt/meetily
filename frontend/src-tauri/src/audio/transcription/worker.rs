// audio/transcription/worker.rs
//
// Transcription worker and chunk processing logic.

use super::engine::TranscriptionEngine;
use super::provider::TranscriptionError;
use crate::audio::AudioChunk;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Runtime};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Serial processing keeps transcripts in chronological order.
pub const TRANSCRIPTION_WORKERS: usize = 1;

// Sequence counter for transcript updates
static SEQUENCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Identifies one recording's transcription task; stamped on its transcript updates.
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

/// Issue the next transcription session id. Called before the task that
/// stamps it is spawned, so `last_issued_session_id` already covers it by the
/// time any of its updates are emitted.
pub(crate) fn allocate_session_id() -> u64 {
    NEXT_SESSION_ID.fetch_add(1, Ordering::SeqCst)
}

/// The highest transcription session id issued in this app process, if any.
/// A reloaded webview uses it to reject updates from earlier sessions.
pub fn last_issued_session_id() -> Option<u64> {
    match NEXT_SESSION_ID.load(Ordering::SeqCst) {
        1 => None,
        next => Some(next - 1),
    }
}

// Speech detection flag - reset per recording session
static SPEECH_DETECTED_EMITTED: AtomicBool = AtomicBool::new(false);

/// Reset the speech detected flag for a new recording session
pub fn reset_speech_detected_flag() {
    SPEECH_DETECTED_EMITTED.store(false, Ordering::SeqCst);
    info!("🔍 SPEECH_DETECTED_EMITTED reset to: {}", SPEECH_DETECTED_EMITTED.load(Ordering::SeqCst));
}

/// Returns true if the transcript text is non-trivial and should be emitted.
/// Filters empty/whitespace-only text; no confidence gating is applied.
fn should_emit_transcript(text: &str) -> bool {
    !text.trim().is_empty()
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranscriptUpdate {
    pub text: String,
    pub timestamp: String, // Wall-clock time for reference (e.g., "14:30:05")
    pub source: String,
    pub sequence_id: u64,
    pub chunk_start_time: f64, // Legacy field, kept for compatibility
    pub is_partial: bool,
    pub confidence: f32,
    // NEW: Recording-relative timestamps for playback sync
    pub audio_start_time: f64, // Seconds from recording start (e.g., 125.3)
    pub audio_end_time: f64,   // Seconds from recording start (e.g., 128.6)
    pub duration: f64,          // Segment duration in seconds (e.g., 3.3)
    /// The transcription session (one per recording) that produced this update,
    /// so a listener can ignore a previous recording's late output.
    #[serde(default)]
    pub session_id: u64,
}

// NOTE: get_transcript_history and get_recording_meeting_name functions
// have been moved to recording_commands.rs where they have access to RECORDING_MANAGER

/// Where the transcription task reports to. `AppHandle` in the app; a recorder in tests.
pub trait TranscriptionEvents: Clone + Send + Sync + 'static {
    fn emit_json(&self, event: &str, payload: serde_json::Value);
}

impl<R: Runtime> TranscriptionEvents for AppHandle<R> {
    fn emit_json(&self, event: &str, payload: serde_json::Value) {
        if let Err(e) = self.emit(event, payload) {
            error!("Failed to emit {}: {}", event, e);
        }
    }
}

/// Which engine a transcription task ended up using, so stop unloads that one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Whisper,
    Parakeet,
    Other,
}

/// Point-in-time chunk accounting for one recording's transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressSnapshot {
    /// Speech chunks received from the audio pipeline.
    pub queued: u64,
    /// Chunks actually transcribed (including ones with no speech in them).
    pub completed: u64,
    /// Chunks the engine failed on.
    pub failed: u64,
    /// Chunks never transcribed: model unavailable or worker gone.
    pub skipped: u64,
}

impl ProgressSnapshot {
    /// Chunks received but not yet handled.
    pub fn pending(&self) -> u64 {
        self.queued
            .saturating_sub(self.completed + self.failed + self.skipped)
    }

    /// Chunks with no transcript: failed, skipped, or still pending.
    pub fn not_transcribed(&self) -> u64 {
        self.queued.saturating_sub(self.completed)
    }
}

/// Live counters for one recording's transcription task, shared with
/// `get_transcription_status` and the stop drain.
#[derive(Debug)]
pub struct TranscriptionProgress {
    queued: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    skipped: AtomicU64,
    busy: AtomicBool,
    finished: AtomicBool,
    last_activity: Mutex<Instant>,
    engine_kind: OnceLock<EngineKind>,
}

impl TranscriptionProgress {
    pub(crate) fn new() -> Self {
        Self {
            queued: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            busy: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            last_activity: Mutex::new(Instant::now()),
            engine_kind: OnceLock::new(),
        }
    }

    pub fn snapshot(&self) -> ProgressSnapshot {
        ProgressSnapshot {
            queued: self.queued.load(Ordering::SeqCst),
            completed: self.completed.load(Ordering::SeqCst),
            failed: self.failed.load(Ordering::SeqCst),
            skipped: self.skipped.load(Ordering::SeqCst),
        }
    }

    /// True once the task has ended (normally, by panic, or by abort).
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    /// True while a chunk is being transcribed.
    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    /// Time since a chunk was last queued or handled.
    pub fn idle_for(&self) -> Duration {
        self.last_activity.lock().unwrap_or_else(|e| e.into_inner()).elapsed()
    }

    pub fn engine_kind(&self) -> Option<EngineKind> {
        self.engine_kind.get().copied()
    }

    fn touch(&self) {
        *self.last_activity.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
    }

    fn record(&self, outcome: ChunkOutcome) {
        let counter = match outcome {
            ChunkOutcome::Transcribed => &self.completed,
            ChunkOutcome::Failed => &self.failed,
            ChunkOutcome::Skipped => &self.skipped,
        };
        counter.fetch_add(1, Ordering::SeqCst);
        self.busy.store(false, Ordering::SeqCst);
        self.touch();
    }
}

/// Marks the progress finished when the task's future is dropped, however it ends.
struct FinishedOnDrop(Arc<TranscriptionProgress>);

impl Drop for FinishedOnDrop {
    fn drop(&mut self) {
        self.0.busy.store(false, Ordering::SeqCst);
        self.0.finished.store(true, Ordering::SeqCst);
    }
}

/// A running transcription task and its live accounting.
pub struct TranscriptionTask {
    pub handle: JoinHandle<()>,
    pub progress: Arc<TranscriptionProgress>,
    /// Stops the dispatcher and the worker, including results of an in-flight
    /// chunk. Aborting `handle` alone would detach the worker, not stop it.
    pub cancel: CancellationToken,
    pub session_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkOutcome {
    Transcribed,
    Failed,
    Skipped,
}

/// Session-level reasons transcription stops working while audio keeps recording.
/// Each is reported to the frontend once per recording, never per chunk.
#[derive(Debug, Clone, Copy)]
enum ActiveFailure {
    EngineInit,
    ModelUnavailable,
    WorkerStopped,
}

#[derive(Default)]
struct ActiveFailureLatch {
    engine_init: AtomicBool,
    model_unavailable: AtomicBool,
    worker_stopped: AtomicBool,
}

impl ActiveFailureLatch {
    fn report<E: TranscriptionEvents>(&self, events: &E, failure: ActiveFailure, detail: &str) {
        let (latch, user_message, actionable) = match failure {
            ActiveFailure::EngineInit => (
                &self.engine_init,
                "Live transcription could not start. Audio is still being recorded; check your transcription model settings.",
                true,
            ),
            ActiveFailure::ModelUnavailable => (
                &self.model_unavailable,
                "The speech recognition model is no longer loaded, so live transcription stopped. Audio is still being recorded.",
                false,
            ),
            ActiveFailure::WorkerStopped => (
                &self.worker_stopped,
                "Live transcription stopped unexpectedly. Audio is still being recorded.",
                false,
            ),
        };
        if latch.swap(true, Ordering::SeqCst) {
            return;
        }
        error!("Transcription stopped for this recording ({:?}): {}", failure, detail);
        events.emit_json(
            "transcription-error",
            serde_json::json!({
                "error": detail,
                "userMessage": user_message,
                "actionable": actionable,
                "phase": "active"
            }),
        );
    }
}

/// Start the transcription task for one recording.
///
/// Every chunk the pipeline sends is accounted for in the returned progress:
/// transcribed, failed, or skipped. If the engine cannot start or the worker
/// dies, the frontend is told once and remaining chunks are drained and counted
/// as skipped, so the pipeline's sends never fail and loss is reported honestly.
pub fn start_transcription_task<R: Runtime>(
    app: AppHandle<R>,
    transcription_receiver: mpsc::UnboundedReceiver<AudioChunk>,
) -> TranscriptionTask {
    let progress = Arc::new(TranscriptionProgress::new());
    let task_progress = Arc::clone(&progress);
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let session_id = allocate_session_id();

    let handle = tokio::spawn(async move {
        let _finished = FinishedOnDrop(Arc::clone(&task_progress));
        let failures = Arc::new(ActiveFailureLatch::default());
        info!("🚀 Starting transcription task");

        // Initialize transcription engine (Whisper or Parakeet based on config)
        let engine = match super::engine::get_or_init_transcription_engine(&app).await {
            Ok(engine) => Some(engine),
            Err(e) => {
                failures.report(&app, ActiveFailure::EngineInit, &e);
                None
            }
        };

        let session = WorkerSession { failures, cancel: task_cancel, session_id };
        run_transcription(engine, transcription_receiver, app, task_progress, session).await;
    });

    TranscriptionTask { handle, progress, cancel, session_id }
}

/// Per-recording state shared by the dispatcher and the worker.
#[derive(Clone)]
struct WorkerSession {
    failures: Arc<ActiveFailureLatch>,
    cancel: CancellationToken,
    session_id: u64,
}

/// Dispatch chunks from the pipeline to the worker until the pipeline closes the channel.
async fn run_transcription<E: TranscriptionEvents>(
    engine: Option<TranscriptionEngine>,
    mut receiver: mpsc::UnboundedReceiver<AudioChunk>,
    events: E,
    progress: Arc<TranscriptionProgress>,
    session: WorkerSession,
) {
    let failures = Arc::clone(&session.failures);
    let mut worker = engine.map(|engine| {
        let kind = match &engine {
            TranscriptionEngine::Whisper(_) => EngineKind::Whisper,
            TranscriptionEngine::Parakeet(_) => EngineKind::Parakeet,
            TranscriptionEngine::Provider(_) => EngineKind::Other,
        };
        let _ = progress.engine_kind.set(kind);
        let (work_sender, work_receiver) = mpsc::unbounded_channel::<AudioChunk>();
        let handle = tokio::spawn(run_worker(
            engine,
            work_receiver,
            events.clone(),
            Arc::clone(&progress),
            session.clone(),
        ));
        (work_sender, handle)
    });

    loop {
        let chunk = tokio::select! {
            biased;
            _ = session.cancel.cancelled() => {
                info!("Transcription session {} cancelled", session.session_id);
                return;
            }
            chunk = receiver.recv() => chunk,
        };
        let Some(chunk) = chunk else { break };
        progress.queued.fetch_add(1, Ordering::SeqCst);
        progress.touch();

        let delivered = match &worker {
            Some((work_sender, handle)) if !handle.is_finished() => work_sender.send(chunk).is_ok(),
            _ => false,
        };
        if !delivered {
            if worker.take().is_some() {
                failures.report(&events, ActiveFailure::WorkerStopped, "Transcription worker exited");
            }
            progress.skipped.fetch_add(1, Ordering::SeqCst);
        }
    }

    // Input finished: let the worker drain its queue, then account for anything it never handled.
    if let Some((work_sender, handle)) = worker {
        drop(work_sender);
        if session.cancel.is_cancelled() {
            return;
        }
        if let Err(e) = handle.await {
            failures.report(&events, ActiveFailure::WorkerStopped, &format!("Transcription worker failed: {}", e));
        }
    }
    let unhandled = progress.snapshot().pending();
    if unhandled > 0 {
        progress.skipped.fetch_add(unhandled, Ordering::SeqCst);
    }

    let snapshot = progress.snapshot();
    events.emit_json(
        "transcription-queue-complete",
        serde_json::json!({
            "total_chunks": snapshot.queued,
            "message": format!("{} chunks queued for processing", snapshot.queued)
        }),
    );
    info!(
        "Transcription task finished: {} queued, {} transcribed, {} failed, {} skipped",
        snapshot.queued, snapshot.completed, snapshot.failed, snapshot.skipped
    );
}

async fn run_worker<E: TranscriptionEvents>(
    engine: TranscriptionEngine,
    mut work_receiver: mpsc::UnboundedReceiver<AudioChunk>,
    events: E,
    progress: Arc<TranscriptionProgress>,
    session: WorkerSession,
) {
    let engine_name = engine.provider_name().to_string();
    if engine.is_model_loaded().await {
        let current_model = engine
            .get_current_model()
            .await
            .unwrap_or_else(|| "unknown".to_string());
        info!("✅ Worker: {} model '{}' is loaded and ready", engine_name, current_model);
    } else {
        warn!("⚠️ Worker: {} model not loaded - chunks will be skipped", engine_name);
    }

    loop {
        let chunk = tokio::select! {
            biased;
            _ = session.cancel.cancelled() => break,
            chunk = work_receiver.recv() => chunk,
        };
        let Some(chunk) = chunk else { break };
        progress.busy.store(true, Ordering::SeqCst);
        let outcome = process_chunk(&engine, chunk, &events, &session).await;
        if session.cancel.is_cancelled() {
            break;
        }
        progress.record(outcome);

        let snapshot = progress.snapshot();
        let handled = snapshot.queued - snapshot.pending();
        let progress_percentage = if snapshot.queued > 0 {
            (handled as f64 / snapshot.queued as f64 * 100.0) as u32
        } else {
            100
        };
        events.emit_json(
            "transcription-progress",
            serde_json::json!({
                "worker_id": 0,
                "chunks_completed": snapshot.completed,
                "chunks_processed": handled,
                "chunks_queued": snapshot.queued,
                "progress_percentage": progress_percentage,
                "message": format!("Transcribing... ({}/{})", handled, snapshot.queued)
            }),
        );
    }

    info!("👷 Transcription worker completed");
}

async fn process_chunk<E: TranscriptionEvents>(
    engine: &TranscriptionEngine,
    chunk: AudioChunk,
    events: &E,
    session: &WorkerSession,
) -> ChunkOutcome {
    let failures = &session.failures;
    if !engine.is_model_loaded().await {
        failures.report(events, ActiveFailure::ModelUnavailable, "No transcription model is loaded");
        return ChunkOutcome::Skipped;
    }

    let chunk_id = chunk.chunk_id;
    let chunk_timestamp = chunk.timestamp;
    let chunk_duration = chunk.data.len() as f64 / chunk.sample_rate as f64;

    let result = transcribe_chunk_with_provider(engine, chunk).await;
    // A cancelled session's in-flight result belongs to a meeting that was already saved.
    if session.cancel.is_cancelled() {
        return ChunkOutcome::Skipped;
    }
    let (transcript, confidence_opt, is_partial) = match result {
        Ok(result) => result,
        // Expected for very short chunks: there was nothing to transcribe.
        Err(e @ TranscriptionError::AudioTooShort { .. }) => {
            info!("Chunk {}: {}", chunk_id, e);
            return ChunkOutcome::Transcribed;
        }
        Err(TranscriptionError::ModelNotLoaded) => {
            failures.report(events, ActiveFailure::ModelUnavailable, "Model unloaded during transcription");
            return ChunkOutcome::Skipped;
        }
        Err(e) => {
            warn!("Chunk {}: transcription failed: {}", chunk_id, e);
            events.emit_json("transcription-warning", serde_json::Value::String(e.to_string()));
            return ChunkOutcome::Failed;
        }
    };

    if !should_emit_transcript(&transcript) {
        return ChunkOutcome::Transcribed;
    }

    let confidence_str = match confidence_opt {
        Some(c) => format!("{:.2}", c),
        None => "N/A".to_string(),
    };
    info!("✅ Transcribed chunk {}: {} (confidence: {}, partial: {})",
          chunk_id, transcript, confidence_str, is_partial);

    // Emit speech-detected event for frontend UX (only on first detection per session)
    if !SPEECH_DETECTED_EMITTED.swap(true, Ordering::SeqCst) {
        events.emit_json(
            "speech-detected",
            serde_json::json!({ "message": "Speech activity detected" }),
        );
        info!("🎤 First speech detected - emitted speech-detected event");
    }

    // The recording_commands listener saves each update to the meeting's transcript store.
    let update = TranscriptUpdate {
        text: transcript,
        timestamp: format_current_timestamp(), // Wall-clock for reference
        source: "Audio".to_string(),
        sequence_id: SEQUENCE_COUNTER.fetch_add(1, Ordering::SeqCst),
        chunk_start_time: chunk_timestamp, // Legacy compatibility
        is_partial,
        confidence: confidence_opt.unwrap_or(0.85), // Default for providers without confidence
        audio_start_time: chunk_timestamp, // Already in seconds from recording start
        audio_end_time: chunk_timestamp + chunk_duration,
        duration: chunk_duration,
        session_id: session.session_id,
    };
    match serde_json::to_value(&update) {
        Ok(payload) => events.emit_json("transcript-update", payload),
        Err(e) => error!("Failed to serialize transcript update: {}", e),
    }

    ChunkOutcome::Transcribed
}

/// Transcribe audio chunk using the appropriate provider (Whisper, Parakeet, or trait-based)
/// Returns: (text, confidence Option, is_partial)
async fn transcribe_chunk_with_provider(
    engine: &TranscriptionEngine,
    chunk: AudioChunk,
) -> std::result::Result<(String, Option<f32>, bool), TranscriptionError> {
    // Convert to 16kHz mono for transcription
    let transcription_data = if chunk.sample_rate != 16000 {
        crate::audio::audio_processing::resample_audio(&chunk.data, chunk.sample_rate, 16000)
    } else {
        chunk.data
    };

    // Skip VAD processing here since the pipeline already extracted speech using VAD
    let speech_samples = transcription_data;

    // Check for empty samples - improved error handling
    if speech_samples.is_empty() {
        warn!(
            "Audio chunk {} is empty, skipping transcription",
            chunk.chunk_id
        );
        return Err(TranscriptionError::AudioTooShort {
            samples: 0,
            minimum: 1600, // 100ms at 16kHz
        });
    }

    // Calculate energy for logging/monitoring only
    let energy: f32 =
        speech_samples.iter().map(|&x| x * x).sum::<f32>() / speech_samples.len() as f32;
    info!(
        "Processing speech audio chunk {} with {} samples (energy: {:.6})",
        chunk.chunk_id,
        speech_samples.len(),
        energy
    );

    // Transcribe using the appropriate engine
    let result = match engine {
        TranscriptionEngine::Whisper(whisper_engine) => {
            // Get language preference from global state
            let language = crate::get_language_preference_internal();
            whisper_engine
                .transcribe_audio_with_confidence(speech_samples, language)
                .await
                .map(|(text, confidence, is_partial)| (text.trim().to_string(), Some(confidence), is_partial))
                .map_err(|e| TranscriptionError::EngineFailed(e.to_string()))
        }
        TranscriptionEngine::Parakeet(parakeet_engine) => parakeet_engine
            .transcribe_audio(speech_samples)
            .await
            // Parakeet doesn't provide confidence or partial results
            .map(|text| (text.trim().to_string(), None, false))
            .map_err(|e| TranscriptionError::EngineFailed(e.to_string())),
        TranscriptionEngine::Provider(provider) => {
            let language = crate::get_language_preference_internal();
            provider
                .transcribe(speech_samples, language)
                .await
                .map(|result| (result.text.trim().to_string(), result.confidence, result.is_partial))
        }
    };

    match result {
        Ok(transcript) => {
            if !transcript.0.is_empty() {
                info!(
                    "{} transcription complete for chunk {}: '{}'",
                    engine.provider_name(),
                    chunk.chunk_id,
                    transcript.0
                );
            }
            Ok(transcript)
        }
        Err(e) => {
            error!(
                "{} transcription failed for chunk {}: {}",
                engine.provider_name(),
                chunk.chunk_id,
                e
            );
            // An engine error caused by an unload is a session problem, not a bad chunk.
            if !engine.is_model_loaded().await {
                return Err(TranscriptionError::ModelNotLoaded);
            }
            Err(e)
        }
    }
}

/// Format current timestamp (wall-clock time)
fn format_current_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();

    let hours = (now.as_secs() / 3600) % 24;
    let minutes = (now.as_secs() / 60) % 60;
    let seconds = now.as_secs() % 60;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Format recording-relative time as [MM:SS]
#[allow(dead_code)]
fn format_recording_time(seconds: f64) -> String {
    let total_seconds = seconds.floor() as u64;
    let minutes = total_seconds / 60;
    let secs = total_seconds % 60;

    format!("[{:02}:{:02}]", minutes, secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::recording_state::DeviceType;
    use crate::audio::transcription::provider::{TranscriptResult, TranscriptionProvider};
    use async_trait::async_trait;

    #[test]
    fn the_last_issued_session_id_covers_every_allocated_session() {
        let first = allocate_session_id();
        let second = allocate_session_id();
        assert!(second > first);
        // Other tests may allocate concurrently, so the latest id is at least ours.
        assert!(last_issued_session_id().is_some_and(|last| last >= second));
    }

    #[test]
    fn keeps_short_acknowledgements() {
        assert!(should_emit_transcript("Yes"));
        assert!(should_emit_transcript("ok"));
    }

    #[test]
    fn drops_empty_and_whitespace_only() {
        assert!(!should_emit_transcript(""));
        assert!(!should_emit_transcript("   "));
    }

    #[derive(Clone, Default)]
    struct RecordedEvents(Arc<Mutex<Vec<(String, serde_json::Value)>>>);

    impl TranscriptionEvents for RecordedEvents {
        fn emit_json(&self, event: &str, payload: serde_json::Value) {
            self.0.lock().unwrap().push((event.to_string(), payload));
        }
    }

    impl RecordedEvents {
        fn named(&self, event: &str) -> Vec<serde_json::Value> {
            self.0
                .lock()
                .unwrap()
                .iter()
                .filter(|(name, _)| name == event)
                .map(|(_, payload)| payload.clone())
                .collect()
        }
    }

    /// Chunk behaviour is encoded in its first sample.
    const SPEECH: f32 = 0.1;
    const ENGINE_ERROR: f32 = 0.2;
    const PANIC: f32 = 0.3;

    /// Fake engine: transcribes, fails or panics per chunk, and "unloads" its
    /// model after `unload_after` successful calls.
    struct ScriptedProvider {
        calls: AtomicU64,
        unload_after: u64,
        delay: Duration,
    }

    fn test_session() -> WorkerSession {
        WorkerSession {
            failures: Arc::new(ActiveFailureLatch::default()),
            cancel: CancellationToken::new(),
            session_id: 42,
        }
    }

    #[async_trait]
    impl TranscriptionProvider for ScriptedProvider {
        async fn transcribe(
            &self,
            audio: Vec<f32>,
            _language: Option<String>,
        ) -> Result<TranscriptResult, TranscriptionError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            match audio[0] {
                x if x == ENGINE_ERROR => Err(TranscriptionError::EngineFailed("decoder error".into())),
                x if x == PANIC => panic!("engine crashed"),
                _ => Ok(TranscriptResult {
                    text: "hello".into(),
                    confidence: Some(0.9),
                    is_partial: false,
                }),
            }
        }

        async fn is_model_loaded(&self) -> bool {
            self.calls.load(Ordering::SeqCst) < self.unload_after
        }

        async fn get_current_model(&self) -> Option<String> {
            Some("scripted".into())
        }

        fn provider_name(&self) -> &'static str {
            "Scripted"
        }
    }

    fn chunk(marker: f32, id: u64) -> AudioChunk {
        AudioChunk {
            data: vec![marker; 1600],
            sample_rate: 16000,
            timestamp: id as f64,
            chunk_id: id,
            device_type: DeviceType::Microphone,
        }
    }

    async fn run_script(markers: &[f32], unload_after: u64) -> (ProgressSnapshot, RecordedEvents) {
        let engine = TranscriptionEngine::Provider(Arc::new(ScriptedProvider {
            calls: AtomicU64::new(0),
            unload_after,
            delay: Duration::ZERO,
        }));
        let (sender, receiver) = mpsc::unbounded_channel();
        for (id, marker) in markers.iter().enumerate() {
            sender.send(chunk(*marker, id as u64)).unwrap();
        }
        drop(sender);

        let events = RecordedEvents::default();
        let progress = Arc::new(TranscriptionProgress::new());
        tokio::time::timeout(
            Duration::from_secs(5),
            run_transcription(
                Some(engine),
                receiver,
                events.clone(),
                Arc::clone(&progress),
                test_session(),
            ),
        )
        .await
        .expect("the transcription task must finish once its input closes");
        (progress.snapshot(), events)
    }

    #[tokio::test]
    async fn only_transcribed_chunks_count_as_completed() {
        // One engine error, two transcribed, then the model is gone for three more.
        let (snapshot, events) =
            run_script(&[ENGINE_ERROR, SPEECH, SPEECH, SPEECH, SPEECH, SPEECH], 3).await;

        assert_eq!(
            snapshot,
            ProgressSnapshot { queued: 6, completed: 2, failed: 1, skipped: 3 }
        );
        assert_eq!(snapshot.not_transcribed(), 4);
        assert_eq!(snapshot.pending(), 0);
        assert_eq!(events.named("transcript-update").len(), 2);
        assert_eq!(events.named("transcription-warning").len(), 1, "per-chunk failure is a warning");

        let errors = events.named("transcription-error");
        assert_eq!(errors.len(), 1, "model loss is reported once, not per chunk");
        assert_eq!(errors[0]["phase"], "active");
    }

    #[tokio::test]
    async fn a_dead_worker_is_reported_once_and_its_chunks_count_as_lost() {
        let (snapshot, events) =
            run_script(&[SPEECH, PANIC, SPEECH, SPEECH, SPEECH], u64::MAX).await;

        assert_eq!(snapshot.queued, 5);
        assert_eq!(snapshot.completed, 1);
        assert_eq!(snapshot.not_transcribed(), 4);
        assert_eq!(snapshot.pending(), 0, "nothing is left looking in-progress");

        let errors = events.named("transcription-error");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0]["phase"], "active");
    }

    #[tokio::test]
    async fn chunks_after_worker_death_are_skipped_not_queued_into_it() {
        let engine = TranscriptionEngine::Provider(Arc::new(ScriptedProvider {
            calls: AtomicU64::new(0),
            unload_after: u64::MAX,
            delay: Duration::ZERO,
        }));
        let (sender, receiver) = mpsc::unbounded_channel();
        let events = RecordedEvents::default();
        let progress = Arc::new(TranscriptionProgress::new());
        let task = tokio::spawn(run_transcription(
            Some(engine),
            receiver,
            events.clone(),
            Arc::clone(&progress),
            test_session(),
        ));

        sender.send(chunk(PANIC, 0)).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await; // the worker panics on it
        sender.send(chunk(SPEECH, 1)).unwrap();
        sender.send(chunk(SPEECH, 2)).unwrap();
        drop(sender);
        tokio::time::timeout(Duration::from_secs(5), task).await.unwrap().unwrap();

        let snapshot = progress.snapshot();
        assert_eq!(snapshot.queued, 3);
        assert_eq!(snapshot.completed, 0);
        assert_eq!(snapshot.skipped, 3);
        assert_eq!(events.named("transcription-error").len(), 1);
    }

    #[tokio::test]
    async fn cancelling_a_session_stops_its_worker_and_its_output() {
        let engine = TranscriptionEngine::Provider(Arc::new(ScriptedProvider {
            calls: AtomicU64::new(0),
            unload_after: u64::MAX,
            delay: Duration::from_millis(50),
        }));
        let (sender, receiver) = mpsc::unbounded_channel();
        for id in 0..20 {
            sender.send(chunk(SPEECH, id)).unwrap();
        }
        let events = RecordedEvents::default();
        let session = test_session();
        let cancel = session.cancel.clone();
        let task = tokio::spawn(run_transcription(
            Some(engine),
            receiver,
            events.clone(),
            Arc::new(TranscriptionProgress::new()),
            session,
        ));

        while events.named("transcript-update").len() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        cancel.cancel();
        task.abort(); // what abandoning a lingering drain does to the outer task
        let at_cancel = events.named("transcript-update").len();
        tokio::time::sleep(Duration::from_millis(400)).await;

        assert!(at_cancel < 20);
        assert_eq!(
            events.named("transcript-update").len(),
            at_cancel,
            "a cancelled session must not emit more transcripts"
        );
        assert!(events
            .named("transcript-update")
            .iter()
            .all(|update| update["session_id"] == 42));
        drop(sender);
    }
}
