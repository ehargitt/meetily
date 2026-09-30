use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use anyhow::Result;
use log::{error, warn};

use super::devices::AudioDevice;
use super::buffer_pool::AudioBufferPool;

mod stream_health;
pub use stream_health::{
    ErrorOutcome, StreamFault, StreamHealth, StreamHealthEvent, StreamStatus, MAX_SHORT_LIVED_REBUILDS,
    RECOVERED_AFTER_AUDIO,
};

/// Non-stream audio errors (e.g. a failed hand-off to the pipeline) can repeat
/// on every buffer; log them at most this often.
const PIPELINE_ERROR_LOG_INTERVAL: Duration = Duration::from_secs(5);

type HealthCallback = Box<dyn Fn(StreamHealthEvent) + Send + Sync>;

/// Why a captured chunk could not be handed to the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ChunkSendError {
    /// Capture started before the pipeline, or stop already detached it.
    #[error("audio pipeline not ready")]
    NotReady,
    /// The pipeline task has exited while capture is running.
    #[error("audio pipeline has stopped")]
    PipelineClosed,
}

/// Device type for audio chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Microphone,
    System,
}

/// Audio chunk with metadata for processing
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub data: Vec<f32>,
    pub sample_rate: u32,
    pub timestamp: f64,
    pub chunk_id: u64,
    pub device_type: DeviceType,
}

/// Processed audio chunk (post-VAD) for recording
#[derive(Debug, Clone)]
pub struct ProcessedAudioChunk {
    pub data: Vec<f32>,
    pub sample_rate: u32,
    pub timestamp: f64,
    pub device_type: DeviceType,
}

/// Comprehensive error types for audio system
#[derive(Debug, Clone)]
pub enum AudioError {
    DeviceDisconnected,
    StreamFailed,
    ProcessingFailed,
    TranscriptionFailed,
    ChannelClosed,
    InitializationFailed,
    ConfigurationError,
    PermissionDenied,
    BufferOverflow,
    SampleRateUnsupported,
}

impl AudioError {
    /// Check if error is recoverable (can attempt reconnection)
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Device disconnect is now recoverable - we can attempt reconnection
            AudioError::DeviceDisconnected => true,
            AudioError::StreamFailed => true,
            AudioError::ProcessingFailed => true,
            AudioError::TranscriptionFailed => true,
            AudioError::ChannelClosed => false,
            AudioError::InitializationFailed => false,
            AudioError::ConfigurationError => false,
            AudioError::PermissionDenied => false,
            AudioError::BufferOverflow => true,
            AudioError::SampleRateUnsupported => false,
        }
    }

    /// Get user-friendly error message
    pub fn user_message(&self) -> &'static str {
        match self {
            AudioError::DeviceDisconnected => "Audio device was disconnected",
            AudioError::StreamFailed => "Audio stream encountered an error",
            AudioError::ProcessingFailed => "Audio processing failed",
            AudioError::TranscriptionFailed => "Speech transcription failed",
            AudioError::ChannelClosed => "Audio channel was closed unexpectedly",
            AudioError::InitializationFailed => "Failed to initialize audio system",
            AudioError::ConfigurationError => "Audio configuration error",
            AudioError::PermissionDenied => "Microphone permission denied",
            AudioError::BufferOverflow => "Audio buffer overflow",
            AudioError::SampleRateUnsupported => "Audio sample rate not supported",
        }
    }
}

/// Recording statistics
#[derive(Debug, Default)]
pub struct RecordingStats {
    pub chunks_processed: u64,
    pub total_duration: f64,
    pub last_activity: Option<Instant>,
}

/// Unified state management for audio recording
pub struct RecordingState {
    // Core recording state
    is_recording: AtomicBool,
    is_paused: AtomicBool,

    // Audio devices
    microphone_device: Mutex<Option<Arc<AudioDevice>>>,
    system_device: Mutex<Option<Arc<AudioDevice>>>,

    // Audio pipeline
    audio_sender: Mutex<Option<mpsc::UnboundedSender<AudioChunk>>>,

    // Memory optimization
    buffer_pool: AudioBufferPool,

    // Error handling
    error_count: AtomicU32,
    recoverable_error_count: AtomicU32,
    last_error: Mutex<Option<AudioError>>,
    last_error_log: Mutex<Option<Instant>>,

    // Per-stream health; faults go to the session supervisor, which rebuilds
    // dead streams and reports health changes through `health_callback`.
    microphone_health: StreamHealth,
    system_health: StreamHealth,
    fault_sender: Mutex<Option<mpsc::UnboundedSender<StreamFault>>>,
    health_callback: Mutex<Option<HealthCallback>>,
    capture_failed: AtomicBool,

    // Statistics
    stats: Mutex<RecordingStats>,

    // Recording start time for accurate timestamps
    recording_start: Mutex<Option<Instant>>,
    // Pause time tracking
    pause_clock: Mutex<PauseClock>,
}

/// Pause bookkeeping kept under one lock. Capture timestamps are computed from
/// it on the audio threads while pause/resume run on another; with the two
/// fields under separate locks a reader could see the pause already ended but
/// not yet added to the total, and time jumped ahead by the whole pause.
#[derive(Debug, Default, Clone, Copy)]
struct PauseClock {
    /// When the current pause began; `None` while not paused.
    pause_start: Option<Instant>,
    /// Sum of all finished pauses.
    total: Duration,
}

impl PauseClock {
    fn pause(&mut self, now: Instant) {
        self.pause_start.get_or_insert(now);
    }

    /// End the current pause; returns its length.
    fn resume(&mut self, now: Instant) -> Option<Duration> {
        let paused_for = now.saturating_duration_since(self.pause_start.take()?);
        self.total += paused_for;
        Some(paused_for)
    }

    /// Time spent paused up to `now`, including a pause still in progress.
    fn paused_until(&self, now: Instant) -> Duration {
        self.total
            + self
                .pause_start
                .map_or(Duration::ZERO, |start| now.saturating_duration_since(start))
    }
}

impl RecordingState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            is_recording: AtomicBool::new(false),
            is_paused: AtomicBool::new(false),
            microphone_device: Mutex::new(None),
            system_device: Mutex::new(None),
            audio_sender: Mutex::new(None),
            buffer_pool: AudioBufferPool::new(16, 48000), // Pool of 16 buffers with 48kHz samples capacity
            error_count: AtomicU32::new(0),
            recoverable_error_count: AtomicU32::new(0),
            last_error: Mutex::new(None),
            last_error_log: Mutex::new(None),
            microphone_health: StreamHealth::new(Instant::now()),
            system_health: StreamHealth::new(Instant::now()),
            fault_sender: Mutex::new(None),
            health_callback: Mutex::new(None),
            capture_failed: AtomicBool::new(false),
            stats: Mutex::new(RecordingStats::default()),
            recording_start: Mutex::new(None),
            pause_clock: Mutex::new(PauseClock::default()),
        })
    }

    // Recording control
    pub fn start_recording(&self) -> Result<()> {
        self.is_recording.store(true, Ordering::SeqCst);
        *self.recording_start.lock().unwrap() = Some(Instant::now());
        self.error_count.store(0, Ordering::SeqCst);
        self.recoverable_error_count.store(0, Ordering::SeqCst);
        *self.last_error.lock().unwrap() = None;
        Ok(())
    }

    pub fn stop_recording(&self) {
        self.is_recording.store(false, Ordering::SeqCst);
        self.is_paused.store(false, Ordering::SeqCst);
        // Clear pause tracking when stopping
        self.pause_clock.lock().unwrap().pause_start = None;
        // CRITICAL: Clear audio sender to close the pipeline channel
        // This ensures the pipeline loop exits properly after processing all chunks
        *self.audio_sender.lock().unwrap() = None;
        // CRITICAL: Clear device references to release microphone/speaker
        // Without this, Arc<AudioDevice> references persist and keep the mic active
        *self.microphone_device.lock().unwrap() = None;
        *self.system_device.lock().unwrap() = None;
        log::info!("Recording stopped, device references cleared");
    }

    pub fn pause_recording(&self) -> Result<()> {
        if !self.is_recording() {
            return Err(anyhow::anyhow!("Cannot pause when not recording"));
        }
        if self.is_paused() {
            return Err(anyhow::anyhow!("Recording is already paused"));
        }

        self.pause_clock.lock().unwrap().pause(Instant::now());
        self.is_paused.store(true, Ordering::SeqCst);
        log::info!("Recording paused");
        Ok(())
    }

    pub fn resume_recording(&self) -> Result<()> {
        if !self.is_recording() {
            return Err(anyhow::anyhow!("Cannot resume when not recording"));
        }
        if !self.is_paused() {
            return Err(anyhow::anyhow!("Recording is not paused"));
        }

        // Ends the pause and adds it to the total in one step (see PauseClock).
        let paused_for = self.pause_clock.lock().unwrap().resume(Instant::now());
        if let Some(pause_duration) = paused_for {
            log::info!("Recording resumed after pause of {:.2}s", pause_duration.as_secs_f64());
        }

        self.is_paused.store(false, Ordering::SeqCst);
        Ok(())
    }

    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }

    pub fn is_paused(&self) -> bool {
        self.is_paused.load(Ordering::SeqCst)
    }

    pub fn is_active(&self) -> bool {
        self.is_recording() && !self.is_paused()
    }

    // Device management
    pub fn set_microphone_device(&self, device: Arc<AudioDevice>) {
        *self.microphone_device.lock().unwrap() = Some(device);
    }

    pub fn set_system_device(&self, device: Arc<AudioDevice>) {
        *self.system_device.lock().unwrap() = Some(device);
    }

    pub fn get_microphone_device(&self) -> Option<Arc<AudioDevice>> {
        self.microphone_device.lock().unwrap().clone()
    }

    pub fn get_system_device(&self) -> Option<Arc<AudioDevice>> {
        self.system_device.lock().unwrap().clone()
    }

    // Audio pipeline management
    pub fn set_audio_sender(&self, sender: mpsc::UnboundedSender<AudioChunk>) {
        *self.audio_sender.lock().unwrap() = Some(sender);
    }

    pub fn send_audio_chunk(&self, chunk: AudioChunk) -> std::result::Result<(), ChunkSendError> {
        // Don't send audio chunks when paused
        if self.is_paused() {
            return Ok(()); // Silently discard chunks while paused
        }

        if let Some(sender) = self.audio_sender.lock().unwrap().as_ref() {
            sender.send(chunk).map_err(|_| ChunkSendError::PipelineClosed)?;

            // Update statistics
            let mut stats = self.stats.lock().unwrap();
            stats.chunks_processed += 1;
            stats.last_activity = Some(Instant::now());
            Ok(())
        } else {
            Err(ChunkSendError::NotReady)
        }
    }

    /// The pipeline task is gone while capture is still running: nothing more
    /// can be saved or transcribed, so report the session as failed (once).
    pub fn report_pipeline_closed(&self) {
        self.report_error(AudioError::ChannelClosed);
        self.report_capture_failed(
            "Audio processing stopped unexpectedly. The recording up to this point is being saved."
                .to_string(),
        );
    }

    // Error handling

    /// Receives stream health changes (degraded, recovered, system audio
    /// unavailable, capture failed) for the frontend.
    pub fn set_health_callback<F>(&self, callback: F)
    where
        F: Fn(StreamHealthEvent) + Send + Sync + 'static,
    {
        *self.health_callback.lock().unwrap() = Some(Box::new(callback));
    }

    pub fn emit_health_event(&self, event: StreamHealthEvent) {
        if let Some(callback) = self.health_callback.lock().unwrap().as_ref() {
            callback(event);
        }
    }

    /// Where stream faults (dead streams) are delivered for rebuilding.
    pub fn set_fault_sender(&self, sender: mpsc::UnboundedSender<StreamFault>) {
        *self.fault_sender.lock().unwrap() = Some(sender);
    }

    pub fn stream_health(&self, device_type: DeviceType) -> &StreamHealth {
        match device_type {
            DeviceType::Microphone => &self.microphone_health,
            DeviceType::System => &self.system_health,
        }
    }

    /// Hot path: a capture callback delivered audio.
    pub fn note_stream_callback(&self, device_type: DeviceType) {
        self.stream_health(device_type).on_callback(Instant::now());
    }

    /// A capture stream reported an error. Never stops the session: a stream
    /// with persistent errors is marked dead and handed to the supervisor.
    pub fn report_stream_error(
        &self,
        device_type: DeviceType,
        device_name: &str,
        error: AudioError,
        detail: &str,
    ) -> ErrorOutcome {
        let disconnected = matches!(error, AudioError::DeviceDisconnected);
        let outcome = self.stream_health(device_type).record_error(&error, Instant::now());
        if outcome.already_dead {
            return outcome;
        }

        self.error_count.fetch_add(1, Ordering::SeqCst);
        if error.is_recoverable() {
            self.recoverable_error_count.fetch_add(1, Ordering::SeqCst);
        }
        *self.last_error.lock().unwrap() = Some(error.clone());

        if let Some(suppressed) = outcome.log_now {
            warn!(
                "Audio stream error on {:?} '{}': {:?} ({}); {} similar errors suppressed",
                device_type, device_name, error, detail, suppressed
            );
        }

        if outcome.became_dead {
            error!("{:?} stream '{}' is dead ({:?}); requesting rebuild", device_type, device_name, error);
            self.send_fault(StreamFault {
                device_type,
                reason: format!("{} ({})", error.user_message(), detail),
                disconnected,
            });
        }
        outcome
    }

    fn send_fault(&self, fault: StreamFault) {
        let sent = self
            .fault_sender
            .lock()
            .unwrap()
            .as_ref()
            .map_or(false, |sender| sender.send(fault.clone()).is_ok());
        if !sent {
            warn!("No stream supervisor to rebuild {:?}: {}", fault.device_type, fault.reason);
        }
    }

    /// Mark running streams that have stopped delivering callbacks as dead.
    /// Returns the stalled streams; the caller rebuilds them.
    ///
    /// Linux only: the stall is cpal's ALSA overrun bug. On macOS a Bluetooth
    /// input can legitimately deliver nothing for a while after start (see the
    /// audio wake in recording_manager.rs), and a ScreenCaptureKit system stream
    /// is not known to deliver buffers through silence; a Core Audio system
    /// stream that ends reports itself instead (stream.rs).
    pub fn detect_stalls(&self, now: Instant) -> Vec<DeviceType> {
        let active = self.is_active();
        Self::stall_checked_streams()
            .iter()
            .copied()
            .filter(|device_type| self.stream_health(*device_type).check_stall(now, active))
            .collect()
    }

    /// The streams `detect_stalls` watches on this platform.
    fn stall_checked_streams() -> &'static [DeviceType] {
        if cfg!(target_os = "linux") {
            &[DeviceType::Microphone, DeviceType::System]
        } else {
            &[]
        }
    }

    /// True when no stream can deliver audio: each is absent or failed for good.
    pub fn all_streams_down(&self) -> bool {
        [DeviceType::Microphone, DeviceType::System].into_iter().all(|device_type| {
            matches!(
                self.stream_health(device_type).status(),
                StreamStatus::Absent | StreamStatus::Failed
            )
        })
    }

    /// System audio could not be started (no device resolved, or its stream
    /// failed to open): tell the user and leave the stream `Failed`, so the
    /// supervisor retries it like one that failed mid-session.
    pub fn report_system_audio_unavailable(&self, device_name: Option<String>, reason: String) {
        warn!("System audio unavailable ({:?}): {} — recording microphone only", device_name, reason);
        let health = self.stream_health(DeviceType::System);
        health.mark_failed(Instant::now());
        if health.claim_failure_announcement() {
            self.emit_health_event(StreamHealthEvent::SystemAudioUnavailable { device_name, reason });
        }
    }

    /// Report that the session can no longer capture any audio. Emitted once;
    /// the session keeps recording state so the normal stop/save still runs.
    pub fn report_capture_failed(&self, message: String) {
        if self.capture_failed.swap(true, Ordering::SeqCst) {
            return;
        }
        error!("Recording can no longer capture audio: {}", message);
        self.emit_health_event(StreamHealthEvent::CaptureFailed { message });
    }

    /// Record a non-stream audio error (e.g. the pipeline hand-off failed).
    /// Logged with a rate limit; it never stops the session.
    pub fn report_error(&self, error: AudioError) {
        let count = self.error_count.fetch_add(1, Ordering::SeqCst) + 1;
        if error.is_recoverable() {
            self.recoverable_error_count.fetch_add(1, Ordering::SeqCst);
        }
        *self.last_error.lock().unwrap() = Some(error.clone());

        let mut last_log = self.last_error_log.lock().unwrap();
        let now = Instant::now();
        if last_log.map_or(true, |t| now.duration_since(t) >= PIPELINE_ERROR_LOG_INTERVAL) {
            *last_log = Some(now);
            warn!("Audio error: {:?} ({} audio errors this session)", error, count);
        }
    }

    pub fn get_error_count(&self) -> u32 {
        self.error_count.load(Ordering::SeqCst)
    }

    pub fn get_recoverable_error_count(&self) -> u32 {
        self.recoverable_error_count.load(Ordering::SeqCst)
    }

    pub fn get_last_error(&self) -> Option<AudioError> {
        self.last_error.lock().unwrap().clone()
    }

    pub fn has_fatal_error(&self) -> bool {
        if self.capture_failed.load(Ordering::SeqCst) {
            return true;
        }
        if let Some(error) = &*self.last_error.lock().unwrap() {
            !error.is_recoverable() && self.error_count.load(Ordering::SeqCst) > 0
        } else {
            false
        }
    }

    // Statistics
    pub fn get_stats(&self) -> RecordingStats {
        self.stats.lock().unwrap().clone()
    }

    pub fn get_recording_duration(&self) -> Option<f64> {
        self.recording_start
            .lock()
            .unwrap()
            .map(|start| start.elapsed().as_secs_f64())
    }

    pub fn get_active_recording_duration(&self) -> Option<f64> {
        let start = (*self.recording_start.lock().unwrap())?;
        let now = Instant::now();
        let paused = self.pause_clock.lock().unwrap().paused_until(now);
        Some(now.saturating_duration_since(start).saturating_sub(paused).as_secs_f64())
    }

    pub fn get_total_pause_duration(&self) -> f64 {
        self.pause_clock.lock().unwrap().total.as_secs_f64()
    }

    pub fn get_current_pause_duration(&self) -> Option<f64> {
        self.pause_clock
            .lock()
            .unwrap()
            .pause_start
            .map(|start| start.elapsed().as_secs_f64())
    }

    // Memory management
    pub fn get_buffer_pool(&self) -> AudioBufferPool {
        self.buffer_pool.clone()
    }

    // Cleanup
    pub fn cleanup(&self) {
        self.stop_recording();
        *self.microphone_device.lock().unwrap() = None;
        *self.system_device.lock().unwrap() = None;
        *self.audio_sender.lock().unwrap() = None;
        *self.last_error.lock().unwrap() = None;
        *self.health_callback.lock().unwrap() = None;
        *self.fault_sender.lock().unwrap() = None;
        *self.stats.lock().unwrap() = RecordingStats::default();
        *self.recording_start.lock().unwrap() = None;
        *self.pause_clock.lock().unwrap() = PauseClock::default();
        self.error_count.store(0, Ordering::SeqCst);
        self.recoverable_error_count.store(0, Ordering::SeqCst);

        // Clear buffer pool to free memory
        self.buffer_pool.clear();
    }
}

impl Default for RecordingState {
    fn default() -> Self {
        Self {
            is_recording: AtomicBool::new(false),
            is_paused: AtomicBool::new(false),
            microphone_device: Mutex::new(None),
            system_device: Mutex::new(None),
            audio_sender: Mutex::new(None),
            buffer_pool: AudioBufferPool::new(16, 48000), // Pool of 16 buffers with 48kHz samples capacity
            error_count: AtomicU32::new(0),
            recoverable_error_count: AtomicU32::new(0),
            last_error: Mutex::new(None),
            last_error_log: Mutex::new(None),
            microphone_health: StreamHealth::new(Instant::now()),
            system_health: StreamHealth::new(Instant::now()),
            fault_sender: Mutex::new(None),
            health_callback: Mutex::new(None),
            capture_failed: AtomicBool::new(false),
            stats: Mutex::new(RecordingStats::default()),
            recording_start: Mutex::new(None),
            pause_clock: Mutex::new(PauseClock::default()),
        }
    }
}

// Thread-safe cloning for RecordingStats
impl Clone for RecordingStats {
    fn clone(&self) -> Self {
        Self {
            chunks_processed: self.chunks_processed,
            total_duration: self.total_duration,
            last_activity: self.last_activity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The active clock is continuous across resume: at the instant a pause
    /// ends, the time recorded as paused is the same whether read just before
    /// or just after, so a capture timestamp can never jump by the pause.
    #[test]
    fn resume_moves_the_pause_into_the_total_without_a_jump() {
        let start = Instant::now();
        let mut clock = PauseClock::default();
        clock.pause(start + Duration::from_secs(10));

        let resumed_at = start + Duration::from_secs(70);
        let paused_before = clock.paused_until(resumed_at);
        assert_eq!(clock.resume(resumed_at), Some(Duration::from_secs(60)));
        assert_eq!(clock.paused_until(resumed_at), paused_before);
        assert_eq!(clock.paused_until(resumed_at + Duration::from_secs(5)), Duration::from_secs(60));
        assert_eq!(clock.resume(resumed_at), None, "resuming twice adds nothing");
    }

    #[test]
    fn active_duration_excludes_pauses() {
        let state = RecordingState::new();
        state.start_recording().unwrap();
        state.pause_recording().unwrap();
        std::thread::sleep(Duration::from_millis(120));
        let while_paused = state.get_active_recording_duration().unwrap();
        state.resume_recording().unwrap();
        let after_resume = state.get_active_recording_duration().unwrap();

        assert!(state.get_total_pause_duration() >= 0.12);
        assert!(after_resume - while_paused < 0.05, "resume must not move active time forward by the pause");
    }

    /// A recording session with both streams running, a fault channel and a
    /// health callback that records every event.
    fn live_session() -> (
        Arc<RecordingState>,
        mpsc::UnboundedReceiver<StreamFault>,
        Arc<Mutex<Vec<StreamHealthEvent>>>,
    ) {
        let state = RecordingState::new();
        state.start_recording().unwrap();
        let (faults, fault_rx) = mpsc::unbounded_channel();
        state.set_fault_sender(faults);
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        state.set_health_callback(move |event| sink.lock().unwrap().push(event));
        for device_type in [DeviceType::Microphone, DeviceType::System] {
            state.stream_health(device_type).mark_running(Instant::now(), false);
        }
        (state, fault_rx, events)
    }

    fn drain(rx: &mut mpsc::UnboundedReceiver<StreamFault>) -> Vec<StreamFault> {
        std::iter::from_fn(|| rx.try_recv().ok()).collect()
    }

    #[test]
    fn a_disconnect_keeps_the_session_recording_and_requests_one_rebuild() {
        let (state, mut faults, events) = live_session();
        for _ in 0..50 {
            state.report_stream_error(DeviceType::Microphone, "USB mic", AudioError::DeviceDisconnected, "gone");
        }
        assert!(state.is_recording(), "stream errors must never stop the session");
        let faults = drain(&mut faults);
        assert_eq!(faults.len(), 1, "one rebuild request per dead stream");
        assert_eq!(faults[0].device_type, DeviceType::Microphone);
        assert!(faults[0].disconnected);
        assert!(events.lock().unwrap().is_empty(), "the supervisor, not the capture thread, notifies the UI");
    }

    #[test]
    fn an_error_burst_keeps_the_session_recording() {
        let (state, mut faults, _) = live_session();
        for _ in 0..1_000 {
            state.report_stream_error(DeviceType::System, "monitor", AudioError::StreamFailed, "POLLERR");
        }
        assert!(state.is_recording());
        assert_eq!(state.stream_health(DeviceType::System).status(), StreamStatus::Dead);
        assert_eq!(state.stream_health(DeviceType::Microphone).status(), StreamStatus::Running);
        let faults = drain(&mut faults);
        assert_eq!(faults.len(), 1);
        assert!(!faults[0].disconnected);
    }

    #[test]
    fn non_stream_errors_never_stop_the_session() {
        let (state, _, _) = live_session();
        for _ in 0..100 {
            state.report_error(AudioError::ChannelClosed);
        }
        assert!(state.is_recording());
    }

    #[test]
    fn all_streams_down_only_when_each_is_absent_or_failed() {
        use StreamStatus::*;
        let now = Instant::now();
        let cases = [
            (Running, Running, false),
            (Running, Absent, false),
            (Dead, Absent, false),
            (Failed, Running, false),
            (Failed, Absent, true),
            (Absent, Failed, true),
            (Failed, Failed, true),
            (Absent, Absent, true),
        ];
        for (mic, system, expected) in cases {
            let state = RecordingState::new();
            for (device_type, status) in [(DeviceType::Microphone, mic), (DeviceType::System, system)] {
                let health = state.stream_health(device_type);
                health.mark_running(now, false);
                match status {
                    Running => {}
                    Dead => {
                        health.record_error(&AudioError::DeviceDisconnected, now);
                    }
                    Failed => health.mark_failed(now),
                    Absent => health.mark_absent(),
                }
            }
            assert_eq!(state.all_streams_down(), expected, "mic {mic:?}, system {system:?}");
        }
    }

    #[test]
    fn capture_failure_is_reported_once_and_keeps_recording_state() {
        let (state, _, events) = live_session();
        for _ in 0..5 {
            state.report_capture_failed("no audio".to_string());
        }
        assert!(state.is_recording(), "the frontend runs the normal stop/save");
        assert!(state.has_fatal_error());
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamHealthEvent::CaptureFailed { .. }));
    }

    #[test]
    fn a_closed_pipeline_is_reported_once_as_a_capture_failure() {
        let (state, _, events) = live_session();
        let (sender, receiver) = mpsc::unbounded_channel();
        state.set_audio_sender(sender);
        drop(receiver);
        let chunk = AudioChunk {
            data: vec![0.0; 480],
            sample_rate: 48_000,
            timestamp: 0.0,
            chunk_id: 0,
            device_type: DeviceType::Microphone,
        };
        assert_eq!(state.send_audio_chunk(chunk), Err(ChunkSendError::PipelineClosed));
        for _ in 0..10 {
            state.report_pipeline_closed();
        }
        assert_eq!(events.lock().unwrap().len(), 1);
    }

    #[test]
    fn unavailable_system_audio_is_announced_and_retried() {
        let (state, _, events) = live_session();
        state.report_system_audio_unavailable(None, "no monitor source".to_string());

        let events = events.lock().unwrap();
        assert!(matches!(
            events.as_slice(),
            [StreamHealthEvent::SystemAudioUnavailable { device_name: None, reason }] if reason == "no monitor source"
        ));
        let health = state.stream_health(DeviceType::System);
        assert_eq!(health.status(), StreamStatus::Failed);
        assert!(health.retry_due(Instant::now() + Duration::from_secs(31)), "retried like a mid-session failure");
        assert!(!state.all_streams_down(), "the microphone is still recording");
    }

    #[test]
    fn stalls_are_not_detected_while_paused() {
        let (state, _, _) = live_session();
        let later = Instant::now() + Duration::from_secs(10);
        state.pause_recording().unwrap();
        assert!(state.detect_stalls(later).is_empty());
        state.resume_recording().unwrap();
        let stalled = state.detect_stalls(later);
        if cfg!(target_os = "linux") {
            assert_eq!(stalled, vec![DeviceType::Microphone, DeviceType::System]);
        } else {
            assert!(stalled.is_empty(), "no stall watchdog on this platform");
        }
    }
}
