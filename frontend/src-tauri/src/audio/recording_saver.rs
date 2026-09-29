use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use anyhow::Result;
use log::{info, warn, error};
use tauri::{AppHandle, Runtime, Emitter};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};
use std::path::PathBuf;

use super::recording_state::AudioChunk;
use super::audio_processing::create_meeting_folder;
use super::incremental_saver::IncrementalAudioSaver;

/// Coalesce transcripts.json rewrites: a burst of segments costs one write.
const TRANSCRIPT_FLUSH_DEBOUNCE: Duration = Duration::from_secs(2);
/// At most one `recording-save-error` per this interval while checkpoints keep failing.
const SAVE_ERROR_REPORT_INTERVAL: Duration = Duration::from_secs(30);
/// Bound on waiting for the accumulator to drain after the pipeline stopped.
const ACCUMULATOR_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Structured transcript segment for JSON export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub id: String,
    pub text: String,
    pub audio_start_time: f64, // Seconds from recording start
    pub audio_end_time: f64,   // Seconds from recording start
    pub duration: f64,          // Segment duration in seconds
    pub display_time: String,   // Formatted time for display like "[02:15]"
    pub confidence: f32,
    pub sequence_id: u64,
}

/// Meeting metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingMetadata {
    pub version: String,
    pub meeting_id: Option<String>,
    pub meeting_name: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
    pub duration_seconds: Option<f64>,
    pub devices: DeviceInfo,
    pub audio_file: String,
    pub transcript_file: String,
    pub sample_rate: u32,
    pub status: String,  // "recording", "completed", "error"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub microphone: Option<String>,
    pub system_audio: Option<String>,
}

/// One recording's transcript segments and their transcripts.json file.
///
/// Segments arrive on the transcription worker's thread (Tauri runs Rust
/// listeners inside `emit`), so `upsert` only updates memory and nudges a
/// background writer; the file is rewritten at most once per debounce window
/// instead of once per segment.
pub struct TranscriptStore {
    segments: Mutex<Vec<TranscriptSegment>>,
    folder: Mutex<Option<PathBuf>>,
    flush_signal: Mutex<Option<mpsc::Sender<()>>>,
    /// Serializes file writes: the debounced writer and the final flush share a temp file.
    write_lock: Mutex<()>,
}

impl TranscriptStore {
    fn new() -> Self {
        Self {
            segments: Mutex::new(Vec::new()),
            folder: Mutex::new(None),
            flush_signal: Mutex::new(None),
            write_lock: Mutex::new(()),
        }
    }

    /// Add or update a segment (upsert by sequence_id) and schedule a file write.
    pub fn upsert(&self, segment: TranscriptSegment) {
        {
            let mut segments = self.segments.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(existing) = segments.iter_mut().find(|s| s.sequence_id == segment.sequence_id) {
                *existing = segment;
            } else {
                segments.push(segment);
            }
        }

        if let Some(signal) = self.flush_signal.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            // Full means a write is already pending, which will include this segment.
            let _ = signal.try_send(());
        }
    }

    pub fn segments(&self) -> Vec<TranscriptSegment> {
        self.segments.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Point the store at the meeting folder and start its debounced writer.
    /// Must be called from within the tokio runtime.
    fn attach_folder(self: &Arc<Self>, folder: PathBuf, debounce: Duration) {
        *self.folder.lock().unwrap_or_else(|e| e.into_inner()) = Some(folder);
        let (sender, receiver) = mpsc::channel(1);
        *self.flush_signal.lock().unwrap_or_else(|e| e.into_inner()) = Some(sender);
        tokio::spawn(run_transcript_writer(Arc::downgrade(self), receiver, debounce));
    }

    /// A store writing to `folder`, with a debounce long enough that only explicit flushes write.
    #[cfg(test)]
    pub(crate) fn for_test(folder: PathBuf) -> Arc<Self> {
        let store = Arc::new(Self::new());
        store.attach_folder(folder, Duration::from_secs(3600));
        store
    }

    /// Write transcripts.json now (atomic write with temp file). No-op without a folder.
    pub fn write_now(&self) -> Result<()> {
        let Some(folder) = self.folder.lock().unwrap_or_else(|e| e.into_inner()).clone() else {
            return Ok(());
        };
        let _write_guard = self.write_lock.lock().unwrap_or_else(|e| e.into_inner());
        let segments = self.segments();

        let transcript_path = folder.join("transcripts.json");
        let temp_path = folder.join(".transcripts.json.tmp");

        let json = serde_json::json!({
            "version": "1.0",
            "segments": segments,
            "last_updated": chrono::Utc::now().to_rfc3339(),
            "total_segments": segments.len()
        });

        let json_string = serde_json::to_string_pretty(&json)
            .map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))?;

        std::fs::write(&temp_path, &json_string)
            .map_err(|e| anyhow::anyhow!("Failed to write {}: {}", temp_path.display(), e))?;

        std::fs::rename(&temp_path, &transcript_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to rename {} to {}: {}",
                temp_path.display(),
                transcript_path.display(),
                e
            )
        })?;

        info!("Wrote transcripts.json with {} segments", segments.len());
        Ok(())
    }
}

/// Debounced transcripts.json writer; ends when its store is dropped.
async fn run_transcript_writer(store: Weak<TranscriptStore>, mut signal: mpsc::Receiver<()>, debounce: Duration) {
    while signal.recv().await.is_some() {
        tokio::time::sleep(debounce).await;
        let Some(store) = store.upgrade() else { break };
        match tokio::task::spawn_blocking(move || store.write_now()).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => warn!("Failed to write incremental transcript update: {}", e),
            Err(e) => warn!("Transcript writer task failed: {}", e),
        }
    }
}

/// Reports audio that could not be written to disk while recording.
///
/// The saver has no AppHandle; the recording command installs a sink that
/// emits `recording-save-error`. Reports are rate-limited so a persistently
/// failing disk produces one toast per interval, not one per chunk.
pub struct SaveErrorReporter {
    sink: Mutex<Option<Box<dyn Fn(&str) + Send + Sync>>>,
    last_reported: Mutex<Option<Instant>>,
}

impl SaveErrorReporter {
    fn new() -> Self {
        Self {
            sink: Mutex::new(None),
            last_reported: Mutex::new(None),
        }
    }

    pub fn set_sink(&self, sink: impl Fn(&str) + Send + Sync + 'static) {
        *self.sink.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(sink));
    }

    pub fn report(&self, message: &str) {
        self.report_at(message, Instant::now());
    }

    /// Returns whether the report reached the sink (false when rate-limited).
    fn report_at(&self, message: &str, now: Instant) -> bool {
        {
            let mut last = self.last_reported.lock().unwrap_or_else(|e| e.into_inner());
            if last.is_some_and(|at| now.duration_since(at) < SAVE_ERROR_REPORT_INTERVAL) {
                return false;
            }
            *last = Some(now);
        }
        error!("Recording save error: {}", message);
        if let Some(sink) = self.sink.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            sink(message);
        }
        true
    }
}

/// Per-recording handles the recording command needs but `RecordingManager`
/// does not expose. Published by `start_accumulation` and taken once by the
/// start command right after `RecordingManager::start_recording` returns
/// (both run under the engine lifecycle lock, so sessions cannot interleave).
pub struct SaverSession {
    pub transcripts: Arc<TranscriptStore>,
    pub save_errors: Arc<SaveErrorReporter>,
    /// Why the meeting folder could not be created, if it could not.
    pub folder_error: Option<String>,
}

static STARTED_SESSION: Mutex<Option<SaverSession>> = Mutex::new(None);

/// Take the handles published by the most recent `start_accumulation`.
pub fn take_started_session() -> Option<SaverSession> {
    STARTED_SESSION.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// New recording saver using incremental saving strategy
pub struct RecordingSaver {
    incremental_saver: Option<Arc<AsyncMutex<IncrementalAudioSaver>>>,
    meeting_folder: Option<PathBuf>,
    meeting_name: Option<String>,
    metadata: Option<MeetingMetadata>,
    transcripts: Arc<TranscriptStore>,
    save_errors: Arc<SaveErrorReporter>,
    accumulator: Option<JoinHandle<()>>,
}

impl RecordingSaver {
    pub fn new() -> Self {
        Self {
            incremental_saver: None,
            meeting_folder: None,
            meeting_name: None,
            metadata: None,
            transcripts: Arc::new(TranscriptStore::new()),
            save_errors: Arc::new(SaveErrorReporter::new()),
            accumulator: None,
        }
    }

    /// Set the meeting name for this recording session
    pub fn set_meeting_name(&mut self, name: Option<String>) {
        self.meeting_name = name;
    }

    /// Set device information in metadata
    pub fn set_device_info(&mut self, mic_name: Option<String>, sys_name: Option<String>) {
        if let Some(ref mut metadata) = self.metadata {
            metadata.devices.microphone = mic_name;
            metadata.devices.system_audio = sys_name;

            // Write updated metadata to disk if folder exists
            if let Some(folder) = &self.meeting_folder {
                let metadata_clone = metadata.clone();
                if let Err(e) = self.write_metadata(folder, &metadata_clone) {
                    warn!("Failed to update metadata with device info: {}", e);
                }
            }
        }
    }

    /// Add or update a structured transcript segment (upserts based on sequence_id)
    /// transcripts.json is rewritten in the background, debounced.
    pub fn add_transcript_segment(&self, segment: TranscriptSegment) {
        self.transcripts.upsert(segment);
    }

    /// Legacy method for backward compatibility - converts text to basic segment
    pub fn add_transcript_chunk(&self, text: String) {
        let segment = TranscriptSegment {
            id: format!("seg_{}", chrono::Utc::now().timestamp_millis()),
            text,
            audio_start_time: 0.0,
            audio_end_time: 0.0,
            duration: 0.0,
            display_time: "[00:00]".to_string(),
            confidence: 1.0,
            sequence_id: 0,
        };
        self.add_transcript_segment(segment);
    }

    /// Start accumulation with optional incremental saving
    ///
    /// # Arguments
    /// * `auto_save` - If true, creates checkpoints and enables saving. If false, audio chunks are discarded.
    pub fn start_accumulation(
        &mut self,
        auto_save: bool,
        mut receiver: mpsc::UnboundedReceiver<AudioChunk>,
    ) {
        if auto_save {
            info!("Initializing incremental audio saver for recording (auto-save ENABLED)");
        } else {
            info!("Starting recording without audio saving (auto-save DISABLED - transcripts only)");
        }

        // Create the meeting folder; with auto_save also .checkpoints/ and the incremental saver.
        // A failure is published in the SaverSession: the start command refuses to
        // record without a folder when auto-save is on.
        let mut folder_error = None;
        if let Some(name) = self.meeting_name.clone() {
            match self.initialize_meeting_folder(&name, auto_save) {
                Ok(()) => info!("Successfully initialized meeting folder (checkpoints: {})", auto_save),
                Err(e) => {
                    error!("Failed to initialize meeting folder: {}", e);
                    folder_error = Some(e.to_string());
                }
            }
        }

        let incremental_saver_arc = self.incremental_saver.clone();
        let save_errors = self.save_errors.clone();
        let save_audio = auto_save;

        // Runs until the pipeline drops its sender, so every chunk mixed before
        // stop is written.
        self.accumulator = Some(tokio::spawn(async move {
            info!("Recording saver accumulation task started (save_audio: {})", save_audio);

            while let Some(chunk) = receiver.recv().await {
                // auto_save off: audio is discarded (transcription already has its own copy)
                if !save_audio {
                    continue;
                }
                if let Some(saver_arc) = &incremental_saver_arc {
                    // add_chunk writes and fsyncs a checkpoint every 30 s: keep that off the runtime.
                    let saver = Arc::clone(saver_arc);
                    let added = tokio::task::spawn_blocking(move || saver.blocking_lock().add_chunk(chunk)).await;
                    match added {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => save_errors.report(&format!("Audio could not be saved to disk: {}", e)),
                        Err(e) => save_errors.report(&format!("Audio could not be saved to disk: {}", e)),
                    }
                }
            }

            info!("Recording saver accumulation task ended");
        }));

        *STARTED_SESSION.lock().unwrap_or_else(|e| e.into_inner()) = Some(SaverSession {
            transcripts: self.transcripts.clone(),
            save_errors: self.save_errors.clone(),
            folder_error,
        });
    }

    /// Initialize meeting folder structure and metadata
    ///
    /// # Arguments
    /// * `meeting_name` - Name of the meeting
    /// * `create_checkpoints` - Whether to create .checkpoints/ directory and IncrementalAudioSaver
    fn initialize_meeting_folder(&mut self, meeting_name: &str, create_checkpoints: bool) -> Result<()> {
        // Load preferences to get base recordings folder
        let base_folder = super::recording_preferences::get_default_recordings_folder();

        // Create meeting folder structure (with or without .checkpoints/ subdirectory)
        let meeting_folder = create_meeting_folder(&base_folder, meeting_name, create_checkpoints)?;

        // Only initialize incremental saver if checkpoints are needed (auto_save is true)
        if create_checkpoints {
            let incremental_saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000)?;
            self.incremental_saver = Some(Arc::new(AsyncMutex::new(incremental_saver)));
            info!("✅ Incremental audio saver initialized for meeting: {}", meeting_name);
        } else {
            info!("⚠️  Skipped incremental audio saver (auto-save disabled)");
        }

        // Create initial metadata
        let metadata = MeetingMetadata {
            version: "1.0".to_string(),
            meeting_id: None,  // Will be set by backend
            meeting_name: Some(meeting_name.to_string()),
            created_at: chrono::Utc::now().to_rfc3339(),
            completed_at: None,
            duration_seconds: None,
            devices: DeviceInfo {
                microphone: None,  // Could be enhanced to store actual device names
                system_audio: None,
            },
            audio_file: if create_checkpoints { "audio.mp4".to_string() } else { "".to_string() },
            transcript_file: "transcripts.json".to_string(),
            sample_rate: 48000,
            status: "recording".to_string(),
        };

        // Write initial metadata.json
        self.write_metadata(&meeting_folder, &metadata)?;

        self.transcripts.attach_folder(meeting_folder.clone(), TRANSCRIPT_FLUSH_DEBOUNCE);
        self.meeting_folder = Some(meeting_folder);
        self.metadata = Some(metadata);

        Ok(())
    }

    /// Write metadata.json to disk (atomic write with temp file)
    fn write_metadata(&self, folder: &PathBuf, metadata: &MeetingMetadata) -> Result<()> {
        let metadata_path = folder.join("metadata.json");
        let temp_path = folder.join(".metadata.json.tmp");

        let json_string = serde_json::to_string_pretty(metadata)?;
        std::fs::write(&temp_path, json_string)?;
        std::fs::rename(&temp_path, &metadata_path)?;  // Atomic

        Ok(())
    }

    // in frontend/src-tauri/src/audio/recording_saver.rs
    pub fn get_stats(&self) -> (usize, u32) {
        if let Some(ref saver) = self.incremental_saver {
            if let Ok(guard) = saver.try_lock() {
                (guard.get_checkpoint_count() as usize, 48000)
            } else {
                (0, 48000)
            }
        } else {
            (0, 48000)
        }
    }

    /// Wait for the accumulation task to write every chunk the pipeline sent.
    /// The pipeline has stopped (dropping its sender) before this is called.
    async fn drain_accumulator(&mut self) {
        let Some(mut accumulator) = self.accumulator.take() else { return };
        match tokio::time::timeout(ACCUMULATOR_DRAIN_TIMEOUT, &mut accumulator).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => error!("Recording saver accumulation task failed: {}", e),
            Err(_) => {
                error!(
                    "Recording saver accumulation task still running after {}s; finalizing without later chunks",
                    ACCUMULATOR_DRAIN_TIMEOUT.as_secs()
                );
                accumulator.abort();
            }
        }
    }

    /// Stop and save using incremental saving approach
    ///
    /// # Arguments
    /// * `app` - Tauri app handle for emitting events
    /// * `recording_duration` - Actual recording duration in seconds (from RecordingState)
    pub async fn stop_and_save<R: Runtime>(
        &mut self,
        app: &AppHandle<R>,
        recording_duration: Option<f64>
    ) -> Result<Option<String>, String> {
        info!("Stopping recording saver");

        self.drain_accumulator().await;

        // Check if incremental saver exists (indicates auto_save was enabled)
        let should_save_audio = self.incremental_saver.is_some();

        if !should_save_audio {
            info!("⚠️  No audio saver initialized (auto-save was disabled) - skipping audio finalization");
            if let Err(e) = self.transcripts.write_now() {
                warn!("Failed to write final transcripts: {}", e);
            }
            return Ok(None);
        }

        // Finalize incremental saver (encode checkpoints into final audio.mp4)
        let final_audio_path = if let Some(saver_arc) = &self.incremental_saver {
            let mut saver = saver_arc.lock().await;
            match saver.finalize().await {
                Ok(path) => {
                    info!("✅ Successfully finalized audio: {}", path.display());
                    path
                }
                Err(e) => {
                    error!("❌ Failed to finalize incremental saver: {}", e);
                    let message = finalize_failure_message(
                        &e.to_string(),
                        saver.get_checkpoint_count(),
                        saver.has_unsaved_audio(),
                        self.meeting_folder.as_deref(),
                    );
                    if let Err(emit_error) = app.emit("recording-save-error", serde_json::json!({ "message": message })) {
                        warn!("Failed to emit recording-save-error: {}", emit_error);
                    }
                    if let Err(e) = self.transcripts.write_now() {
                        warn!("Failed to write final transcripts: {}", e);
                    }
                    return Err(format!("Failed to finalize audio: {}", e));
                }
            }
        } else {
            error!("No incremental saver initialized - cannot save recording");
            return Err("No incremental saver initialized".to_string());
        };

        // Save final transcripts.json with validation
        if let Some(folder) = &self.meeting_folder {
            if let Err(e) = self.transcripts.write_now() {
                error!("❌ Failed to write final transcripts: {}", e);
                return Err(format!("Failed to save transcripts: {}", e));
            }

            // Verify transcripts were written correctly
            let transcript_path = folder.join("transcripts.json");
            if !transcript_path.exists() {
                error!("❌ Transcript file was not created at: {}", transcript_path.display());
                return Err("Transcript file verification failed".to_string());
            }
            info!("✅ Transcripts saved and verified at: {}", transcript_path.display());
        }

        // Update metadata to completed status with actual recording duration
        if let (Some(folder), Some(mut metadata)) = (&self.meeting_folder, self.metadata.clone()) {
            metadata.status = "completed".to_string();
            metadata.completed_at = Some(chrono::Utc::now().to_rfc3339());

            // Use actual recording duration from RecordingState (more accurate than transcript segments)
            // Falls back to last transcript segment if duration not provided
            metadata.duration_seconds = recording_duration
                .or_else(|| self.transcripts.segments().last().map(|seg| seg.audio_end_time));

            if let Err(e) = self.write_metadata(folder, &metadata) {
                error!("❌ Failed to update metadata to completed: {}", e);
                return Err(format!("Failed to update metadata: {}", e));
            }

            info!("✅ Metadata updated with duration: {:?}s", metadata.duration_seconds);
        }

        // Emit save event with audio and transcript paths
        let save_event = serde_json::json!({
            "audio_file": final_audio_path.to_string_lossy(),
            "transcript_file": self.meeting_folder.as_ref()
                .map(|f| f.join("transcripts.json").to_string_lossy().to_string()),
            "meeting_name": self.meeting_name,
            "meeting_folder": self.meeting_folder.as_ref()
                .map(|f| f.to_string_lossy().to_string())
        });

        if let Err(e) = app.emit("recording-saved", &save_event) {
            warn!("Failed to emit recording-saved event: {}", e);
        }

        // Segments are not cleared: transcription may still be finishing in the
        // background and its late segments are written to the same file.

        Ok(Some(final_audio_path.to_string_lossy().to_string()))
    }

    /// Get the meeting folder path (for passing to backend)
    pub fn get_meeting_folder(&self) -> Option<&PathBuf> {
        self.meeting_folder.as_ref()
    }

    /// Get accumulated transcript segments (for reload sync)
    pub fn get_transcript_segments(&self) -> Vec<TranscriptSegment> {
        self.transcripts.segments()
    }

    /// Get meeting name (for reload sync)
    pub fn get_meeting_name(&self) -> Option<String> {
        self.meeting_name.clone()
    }
}

/// User message for a failed finalize. It points at the WAV checkpoints only
/// when some were written.
fn finalize_failure_message(
    error: &str,
    checkpoints_written: u32,
    has_unsaved_audio: bool,
    meeting_folder: Option<&std::path::Path>,
) -> String {
    if checkpoints_written == 0 {
        return if has_unsaved_audio {
            format!("The meeting audio could not be written to disk ({}), so the meeting has no audio file.", error)
        } else {
            format!("No audio was captured for this meeting, so it has no audio file ({}).", error)
        };
    }
    let checkpoints = meeting_folder
        .map(|folder| folder.join(".checkpoints").display().to_string())
        .unwrap_or_else(|| "the meeting folder's .checkpoints folder".to_string());
    format!(
        "The meeting audio could not be finalized ({}), so the meeting has no audio file. The recorded audio is kept as WAV files in {}.",
        error, checkpoints
    )
}

impl Default for RecordingSaver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::recording_state::DeviceType;

    fn segment(sequence_id: u64, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            id: format!("seg_{}", sequence_id),
            text: text.to_string(),
            audio_start_time: sequence_id as f64,
            audio_end_time: sequence_id as f64 + 1.0,
            duration: 1.0,
            display_time: "00:00:00".to_string(),
            confidence: 0.9,
            sequence_id,
        }
    }

    fn written_segment_count(folder: &std::path::Path) -> usize {
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("transcripts.json")).unwrap()).unwrap();
        json["segments"].as_array().unwrap().len()
    }

    #[tokio::test]
    async fn accumulator_writes_every_chunk_sent_before_the_channel_closes() {
        let temp = tempfile::tempdir().unwrap();
        let folder = temp.path().join("meeting");
        std::fs::create_dir_all(folder.join(".checkpoints")).unwrap();

        let mut saver = RecordingSaver::new();
        saver.incremental_saver = Some(Arc::new(AsyncMutex::new(
            IncrementalAudioSaver::new(folder.clone(), 48000).unwrap(),
        )));
        let (sender, receiver) = mpsc::unbounded_channel();
        let incremental = saver.incremental_saver.clone().unwrap();
        // No meeting name: skip folder creation (it uses the user's recordings folder).
        saver.start_accumulation(true, receiver);

        for i in 0..70 {
            sender
                .send(AudioChunk {
                    data: vec![0.2; 24000],
                    sample_rate: 48000,
                    timestamp: i as f64 * 0.5,
                    chunk_id: i,
                    device_type: DeviceType::Microphone,
                })
                .unwrap();
        }
        drop(sender); // the pipeline stopping

        saver.drain_accumulator().await;

        // 35 s sent: one 30 s checkpoint on disk, the last 5 s buffered for finalize.
        let guard = incremental.lock().await;
        assert_eq!(guard.get_checkpoint_count(), 1);
    }

    #[tokio::test]
    async fn transcript_writes_are_debounced_and_late_segments_still_reach_the_file() {
        let debounce = Duration::from_millis(50);
        let temp = tempfile::tempdir().unwrap();
        let store = Arc::new(TranscriptStore::new());
        store.attach_folder(temp.path().to_path_buf(), debounce);

        for i in 0..50 {
            store.upsert(segment(i, "hello"));
        }
        assert!(!temp.path().join("transcripts.json").exists(), "no write per segment");
        wait_for_segment_count(temp.path(), 50).await;

        // A segment transcribed after the final flush is still written.
        store.write_now().unwrap();
        store.upsert(segment(50, "late"));
        wait_for_segment_count(temp.path(), 51).await;
    }

    async fn wait_for_segment_count(folder: &std::path::Path, expected: usize) {
        for _ in 0..200 {
            if folder.join("transcripts.json").exists() && written_segment_count(folder) == expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("transcripts.json never reached {expected} segments");
    }

    #[test]
    fn finalize_failure_mentions_checkpoints_only_when_some_exist() {
        let folder = std::path::Path::new("/meetings/Standup");
        let none = finalize_failure_message("No audio checkpoints to merge", 0, false, Some(folder));
        assert!(none.starts_with("No audio was captured"));
        assert!(!none.contains(".checkpoints"));

        let unwritten = finalize_failure_message("disk full", 0, true, Some(folder));
        assert!(unwritten.contains("could not be written to disk"));
        assert!(!unwritten.contains(".checkpoints"));

        let some = finalize_failure_message("FFmpeg failed", 3, true, Some(folder));
        assert!(some.contains("/meetings/Standup/.checkpoints"));
    }

    #[test]
    fn save_errors_are_reported_at_most_once_per_interval() {
        let reporter = SaveErrorReporter::new();
        let delivered = Arc::new(Mutex::new(Vec::new()));
        let sink_delivered = delivered.clone();
        reporter.set_sink(move |message| sink_delivered.lock().unwrap().push(message.to_string()));

        let start = Instant::now();
        assert!(reporter.report_at("disk full", start));
        assert!(!reporter.report_at("disk full", start + Duration::from_secs(5)));
        assert!(reporter.report_at("disk full", start + SAVE_ERROR_REPORT_INTERVAL));

        assert_eq!(delivered.lock().unwrap().len(), 2);
    }
}
