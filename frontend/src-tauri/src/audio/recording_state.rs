use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use anyhow::Result;
use log::{error, warn};

use super::devices::AudioDevice;
use super::buffer_pool::AudioBufferPool;

mod stream_health;
pub use stream_health::{ErrorOutcome, StreamFault, StreamHealth, StreamHealthEvent, StreamStatus};

/// Non-stream audio errors (e.g. a failed hand-off to the pipeline) can repeat
/// on every buffer; log them at most this often.
const PIPELINE_ERROR_LOG_INTERVAL: Duration = Duration::from_secs(5);

type HealthCallback = Box<dyn Fn(StreamHealthEvent) + Send + Sync>;

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
    pause_start: Mutex<Option<Instant>>,
    total_pause_duration: Mutex<std::time::Duration>,
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
            pause_start: Mutex::new(None),
            total_pause_duration: Mutex::new(std::time::Duration::ZERO),
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
        *self.pause_start.lock().unwrap() = None;
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

        self.is_paused.store(true, Ordering::SeqCst);
        *self.pause_start.lock().unwrap() = Some(Instant::now());
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

        // Calculate pause duration and add to total
        if let Some(pause_start) = self.pause_start.lock().unwrap().take() {
            let pause_duration = pause_start.elapsed();
            *self.total_pause_duration.lock().unwrap() += pause_duration;
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

    pub fn send_audio_chunk(&self, chunk: AudioChunk) -> Result<()> {
        // Don't send audio chunks when paused
        if self.is_paused() {
            return Ok(()); // Silently discard chunks while paused
        }

        if let Some(sender) = self.audio_sender.lock().unwrap().as_ref() {
            sender.send(chunk).map_err(|_| anyhow::anyhow!("Failed to send audio chunk"))?;

            // Update statistics
            let mut stats = self.stats.lock().unwrap();
            stats.chunks_processed += 1;
            stats.last_activity = Some(Instant::now());
            Ok(())
        } else {
            // Return an error when no sender is available (pipeline not ready)
            Err(anyhow::anyhow!("Audio pipeline not ready - no sender available"))
        }
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
    /// audio wake in recording_manager.rs), and rebuilding it there would
    /// change behaviour this fix is not about.
    pub fn detect_stalls(&self, now: Instant) -> Vec<DeviceType> {
        if !cfg!(target_os = "linux") {
            return Vec::new();
        }
        let active = self.is_active();
        [DeviceType::Microphone, DeviceType::System]
            .into_iter()
            .filter(|device_type| self.stream_health(*device_type).check_stall(now, active))
            .collect()
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
        self.recording_start.lock().unwrap().map(|start| {
            let total_duration = start.elapsed().as_secs_f64();
            let pause_duration = self.get_total_pause_duration();
            let current_pause = if self.is_paused() {
                self.pause_start
                    .lock()
                    .unwrap()
                    .map(|p| p.elapsed().as_secs_f64())
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            total_duration - pause_duration - current_pause
        })
    }

    pub fn get_total_pause_duration(&self) -> f64 {
        self.total_pause_duration.lock().unwrap().as_secs_f64()
    }

    pub fn get_current_pause_duration(&self) -> Option<f64> {
        if self.is_paused() {
            self.pause_start
                .lock()
                .unwrap()
                .map(|start| start.elapsed().as_secs_f64())
        } else {
            None
        }
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
        *self.pause_start.lock().unwrap() = None;
        *self.total_pause_duration.lock().unwrap() = std::time::Duration::ZERO;
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
            pause_start: Mutex::new(None),
            total_pause_duration: Mutex::new(std::time::Duration::ZERO),
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