use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Result;
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, Stream, SupportedStreamConfig};
use log::{error, info, warn};
use tokio::sync::mpsc;

use super::devices::{AudioDevice, get_device_and_config};
use super::pipeline::AudioCapture;
use super::recording_state::{RecordingState, DeviceType};
use super::capture::{AudioCaptureBackend, get_current_backend};

#[cfg(target_os = "macos")]
use super::capture::CoreAudioCapture;

/// Stream backend implementation
pub enum StreamBackend {
    /// CPAL-based stream (ScreenCaptureKit or default)
    Cpal(Stream),
    /// Core Audio direct implementation (macOS only)
    #[cfg(target_os = "macos")]
    CoreAudio {
        task: Option<tokio::task::JoinHandle<()>>,
    },
}

// SAFETY: While Stream doesn't implement Send, we ensure it's only accessed
// from the same thread context by using spawn_blocking for operations that cross thread boundaries
unsafe impl Send for StreamBackend {}

/// Simplified audio stream wrapper with multi-backend support
pub struct AudioStream {
    device: Arc<AudioDevice>,
    backend: StreamBackend,
}

// SAFETY: AudioStream contains StreamBackend which we've marked as Send
unsafe impl Send for AudioStream {}

impl AudioStream {
    /// Create a new audio stream for the given device
    pub async fn create(
        device: Arc<AudioDevice>,
        state: Arc<RecordingState>,
        device_type: DeviceType,
        recording_sender: Option<mpsc::UnboundedSender<super::recording_state::AudioChunk>>,
    ) -> Result<Self> {
        // Get current backend from global config
        let backend_type = get_current_backend();
        Self::create_with_backend(device, state, device_type, recording_sender, backend_type).await
    }

    /// Create a new audio stream with explicit backend selection
    pub async fn create_with_backend(
        device: Arc<AudioDevice>,
        state: Arc<RecordingState>,
        device_type: DeviceType,
        recording_sender: Option<mpsc::UnboundedSender<super::recording_state::AudioChunk>>,
        backend_type: AudioCaptureBackend,
    ) -> Result<Self> {
        info!("🎵 Stream: Creating audio stream for device: {} with backend: {:?}, device_type: {:?}",
              device.name, backend_type, device_type);

        // For system audio devices, use the selected backend
        // For microphone devices, always use CPAL
        #[cfg(target_os = "macos")]
        let use_core_audio = device_type == DeviceType::System
            && backend_type == AudioCaptureBackend::CoreAudio;

        #[cfg(not(target_os = "macos"))]
        let use_core_audio = false;

        #[cfg(target_os = "macos")]
        info!("🎵 Stream: use_core_audio = {}, device_type == System: {}, backend == CoreAudio: {}",
              use_core_audio,
              device_type == DeviceType::System,
              backend_type == AudioCaptureBackend::CoreAudio);

        #[cfg(not(target_os = "macos"))]
        info!("🎵 Stream: use_core_audio = {}, device_type == System: {}",
              use_core_audio,
              device_type == DeviceType::System);

        #[cfg(target_os = "macos")]
        if use_core_audio {
            info!("🎵 Stream: Using Core Audio backend (cidre) for system audio");
            return Self::create_core_audio_stream(device, state, device_type, recording_sender).await;
        }

        // Default path: use CPAL
        #[cfg(target_os = "macos")]
        let backend_name = if backend_type == AudioCaptureBackend::ScreenCaptureKit {
            "ScreenCaptureKit"
        } else {
            "CPAL (default)"
        };

        #[cfg(not(target_os = "macos"))]
        let backend_name = "CPAL";

        info!("🎵 Stream: Using CPAL backend ({}) for device: {}", backend_name, device.name);
        Self::create_cpal_stream(device, state, device_type, recording_sender).await
    }

    /// Create a CPAL-based stream (ScreenCaptureKit on macOS)
    async fn create_cpal_stream(
        device: Arc<AudioDevice>,
        state: Arc<RecordingState>,
        device_type: DeviceType,
        recording_sender: Option<mpsc::UnboundedSender<super::recording_state::AudioChunk>>,
    ) -> Result<Self> {
        info!("Creating CPAL stream for device: {}", device.name);

        // Get the underlying cpal device and config
        let (cpal_device, config) = get_device_and_config(&device).await?;

        info!("Audio config - Sample rate: {}, Channels: {}, Format: {:?}",
              config.sample_rate().0, config.channels(), config.sample_format());

        // Create audio capture processor
        let capture = AudioCapture::new(
            device.clone(),
            state.clone(),
            config.sample_rate().0,
            config.channels(),
            device_type,
            recording_sender,
        );

        // Build the appropriate stream based on sample format
        let stream = Self::build_stream(&cpal_device, &config, capture.clone())?;

        // Start the stream
        stream.play()?;
        info!("CPAL stream started for device: {}", device.name);

        Ok(Self {
            device,
            backend: StreamBackend::Cpal(stream),
        })
    }

    /// Create a Core Audio stream (macOS only)
    #[cfg(target_os = "macos")]
    async fn create_core_audio_stream(
        device: Arc<AudioDevice>,
        state: Arc<RecordingState>,
        device_type: DeviceType,
        recording_sender: Option<mpsc::UnboundedSender<super::recording_state::AudioChunk>>,
    ) -> Result<Self> {
        info!("🔊 Stream: Creating Core Audio stream for device: {}", device.name);

        // Create Core Audio capture
        info!("🔊 Stream: Calling CoreAudioCapture::new()...");
        let capture_impl = CoreAudioCapture::new()
            .map_err(|e| {
                error!("❌ Stream: CoreAudioCapture::new() failed: {}", e);
                anyhow::anyhow!("Failed to create Core Audio capture: {}", e)
            })?;

        info!("✅ Stream: CoreAudioCapture created, calling stream()...");
        let core_stream = capture_impl.stream()
            .map_err(|e| {
                error!("❌ Stream: capture_impl.stream() failed: {}", e);
                anyhow::anyhow!("Failed to create Core Audio stream: {}", e)
            })?;

        let sample_rate = core_stream.sample_rate();
        info!("✅ Stream: Core Audio stream created with sample rate: {} Hz", sample_rate);

        // Create audio capture processor for pipeline integration
        // CRITICAL: Core Audio tap is MONO (with_mono_global_tap_excluding_processes)
        let capture = AudioCapture::new(
            device.clone(),
            state.clone(),
            sample_rate,
            1, // Core Audio tap is MONO (not stereo!)
            device_type,
            recording_sender,
        );

        // Spawn task to process Core Audio stream samples
        // The stream needs to be polled continuously to produce samples
        let device_name = device.name.clone();
        info!("🔊 Stream: Spawning tokio task to poll Core Audio stream...");
        let task = tokio::spawn({
            let capture = capture.clone();
            let mut stream = core_stream;

            async move {
                use futures_util::StreamExt;

                let mut buffer = Vec::new();
                let mut frame_count = 0;
                let frames_per_chunk = 1024; // Process in chunks of 1024 samples

                info!("✅ Stream: Core Audio processing task started for {}", device_name);

                let mut _sample_count = 0u64;
                while let Some(sample) = stream.next().await {
                    _sample_count += 1;
                    // if _sample_count % 48000 == 0 {
                    //     info!("📊 Stream: Received {} samples from Core Audio stream", _sample_count);
                    // }

                    buffer.push(sample);
                    frame_count += 1;

                    // Process when we have enough samples
                    if frame_count >= frames_per_chunk {
                        capture.process_audio_data(&buffer);
                        buffer.clear();
                        frame_count = 0;
                    }
                }

                // Process any remaining samples
                if !buffer.is_empty() {
                    capture.process_audio_data(&buffer);
                }

                info!("⚠️ Stream: Core Audio processing task ended for {}", device_name);
            }
        });

        info!("✅ Stream: Core Audio stream fully initialized for device: {}", device.name);

        Ok(Self {
            device: device.clone(),
            backend: StreamBackend::CoreAudio {
                task: Some(task),
            },
        })
    }

    /// Build stream based on sample format
    fn build_stream(
        device: &Device,
        config: &SupportedStreamConfig,
        capture: AudioCapture,
    ) -> Result<Stream> {
        let config_copy = config.clone();

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let capture_clone = capture.clone();
                device.build_input_stream(
                    &config_copy.into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        capture.process_audio_data(data);
                    },
                    move |err| {
                        capture_clone.handle_stream_error(err);
                    },
                    None,
                )?
            }
            cpal::SampleFormat::I16 => {
                let capture_clone = capture.clone();
                device.build_input_stream(
                    &config_copy.into(),
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        let f32_data: Vec<f32> = data.iter()
                            .map(|&sample| sample as f32 / i16::MAX as f32)
                            .collect();
                        capture.process_audio_data(&f32_data);
                    },
                    move |err| {
                        capture_clone.handle_stream_error(err);
                    },
                    None,
                )?
            }
            cpal::SampleFormat::I32 => {
                let capture_clone = capture.clone();
                device.build_input_stream(
                    &config_copy.into(),
                    move |data: &[i32], _: &cpal::InputCallbackInfo| {
                        let f32_data: Vec<f32> = data.iter()
                            .map(|&sample| sample as f32 / i32::MAX as f32)
                            .collect();
                        capture.process_audio_data(&f32_data);
                    },
                    move |err| {
                        capture_clone.handle_stream_error(err);
                    },
                    None,
                )?
            }
            cpal::SampleFormat::I8 => {
                let capture_clone = capture.clone();
                device.build_input_stream(
                    &config_copy.into(),
                    move |data: &[i8], _: &cpal::InputCallbackInfo| {
                        let f32_data: Vec<f32> = data.iter()
                            .map(|&sample| sample as f32 / i8::MAX as f32)
                            .collect();
                        capture.process_audio_data(&f32_data);
                    },
                    move |err| {
                        capture_clone.handle_stream_error(err);
                    },
                    None,
                )?
            }
            _ => {
                return Err(anyhow::anyhow!("Unsupported sample format: {:?}", config.sample_format()));
            }
        };

        Ok(stream)
    }

    /// Get device info
    pub fn device(&self) -> &AudioDevice {
        &self.device
    }

    /// Stop the stream
    pub fn stop(self) -> Result<()> {
        info!("Stopping audio stream for device: {}", self.device.name);

        match self.backend {
            StreamBackend::Cpal(stream) => {
                // CRITICAL: Pause the stream first to stop callbacks immediately
                // This ensures closures stop executing before we drop the stream,
                // allowing Arc references captured in callbacks to be released
                if let Err(e) = stream.pause() {
                    warn!("Failed to pause stream before drop: {}", e);
                }
                info!("Stream paused, now dropping to release callbacks");
                drop(stream);
            }
            #[cfg(target_os = "macos")]
            StreamBackend::CoreAudio { task } => {
                // Abort the processing task and wait briefly for cleanup
                if let Some(task_handle) = task {
                    info!("Aborting Core Audio task...");
                    task_handle.abort();
                    // Give the runtime a moment to clean up the aborted task
                    // This helps ensure Arc references in the closure are dropped
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    info!("Core Audio task aborted");
                }
            }
        }

        // Explicitly drop self.device Arc reference
        drop(self.device);
        info!("Audio stream stopped and device reference dropped");
        Ok(())
    }

    /// Stop the stream on a blocking thread, bounded by `STREAM_STOP_TIMEOUT`.
    ///
    /// Teardown can block (cpal joins its worker thread; a wedged sound server
    /// can hang it) and can panic (cpal's ALSA `Drop` unwraps the worker join, so
    /// a worker that panicked panics again here). Neither may abort a stop or a
    /// rebuild: a panicked worker means the stream was already dead, and a hung
    /// teardown thread is abandoned.
    pub async fn stop_off_runtime(self) {
        let name = self.device.name.clone();
        run_teardown_off_runtime(&name, STREAM_STOP_TIMEOUT, move || self.stop()).await;
    }

    /// Synchronous counterpart of `stop_off_runtime` for `Drop`, which cannot
    /// await: contains a teardown panic instead of letting it escape `drop`.
    fn stop_contained(self) {
        let name = self.device.name.clone();
        run_teardown_contained(&name, move || self.stop());
    }
}

/// How a contained teardown ended.
#[derive(Debug, PartialEq, Eq)]
enum TeardownOutcome {
    Stopped,
    Failed,
    Panicked,
    TimedOut,
}

async fn run_teardown_off_runtime<F>(name: &str, timeout: Duration, teardown: F) -> TeardownOutcome
where
    F: FnOnce() -> Result<()> + Send + 'static,
{
    match tokio::time::timeout(timeout, tokio::task::spawn_blocking(teardown)).await {
        Ok(Ok(Ok(()))) => TeardownOutcome::Stopped,
        Ok(Ok(Err(e))) => {
            warn!("Failed to stop audio stream '{}': {}", name, e);
            TeardownOutcome::Failed
        }
        Ok(Err(join_error)) => {
            warn!(
                "Audio stream '{}' teardown {} (stream was already dead); continuing",
                name,
                if join_error.is_panic() { "panicked" } else { "was cancelled" }
            );
            TeardownOutcome::Panicked
        }
        Err(_) => {
            warn!("Audio stream '{}' teardown still blocked after {:?}; abandoning it", name, timeout);
            TeardownOutcome::TimedOut
        }
    }
}

fn run_teardown_contained<F: FnOnce() -> Result<()>>(name: &str, teardown: F) -> TeardownOutcome {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(teardown)) {
        Ok(Ok(())) => TeardownOutcome::Stopped,
        Ok(Err(e)) => {
            error!("Failed to stop audio stream '{}': {}", name, e);
            TeardownOutcome::Failed
        }
        Err(_) => {
            warn!("Audio stream '{}' teardown panicked (stream was already dead)", name);
            TeardownOutcome::Panicked
        }
    }
}

/// Upper bound on one stream teardown before it is abandoned.
const STREAM_STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// Audio stream manager for handling multiple streams
pub struct AudioStreamManager {
    microphone_stream: Option<AudioStream>,
    system_stream: Option<AudioStream>,
    state: Arc<RecordingState>,
}

// SAFETY: AudioStreamManager contains AudioStream which we've marked as Send
unsafe impl Send for AudioStreamManager {}

impl AudioStreamManager {
    pub fn new(state: Arc<RecordingState>) -> Self {
        Self {
            microphone_stream: None,
            system_stream: None,
            state,
        }
    }

    /// Start audio streams for the given devices
    pub async fn start_streams(
        &mut self,
        microphone_device: Option<Arc<AudioDevice>>,
        system_audio: std::result::Result<Arc<AudioDevice>, String>,
        recording_sender: Option<mpsc::UnboundedSender<super::recording_state::AudioChunk>>,
    ) -> Result<()> {
        use super::capture::get_current_backend;
        let backend = get_current_backend();
        info!("🎙️ Starting audio streams with backend: {:?}", backend);

        // Start microphone stream
        if let Some(mic_device) = microphone_device {
            info!("🎤 Creating microphone stream: {} (always uses CPAL)", mic_device.name);
            match AudioStream::create(mic_device.clone(), self.state.clone(), DeviceType::Microphone, recording_sender.clone()).await {
                Ok(stream) => {
                    self.state.set_microphone_device(mic_device);
                    self.microphone_stream = Some(stream);
                    self.state.stream_health(DeviceType::Microphone).mark_running(Instant::now(), false);
                    info!("✅ Microphone stream created successfully");
                }
                Err(e) => {
                    error!("❌ Failed to create microphone stream: {}", e);
                    return Err(e);
                }
            }
        } else {
            info!("ℹ️ No microphone device specified, skipping microphone stream");
        }

        // Start system audio stream. If it cannot start, record mic-only and
        // say so, but only once start can no longer fail.
        let system_unavailable = match system_audio {
            Ok(sys_device) => {
                info!("🔊 Creating system audio stream: {} (backend: {:?})", sys_device.name, backend);
                let created = AudioStream::create(sys_device.clone(), self.state.clone(), DeviceType::System, recording_sender.clone()).await;
                // Recorded even on failure: the supervisor's retries reopen this device.
                self.state.set_system_device(sys_device.clone());
                match created {
                    Ok(stream) => {
                        self.system_stream = Some(stream);
                        self.state.stream_health(DeviceType::System).mark_running(Instant::now(), false);
                        info!("✅ System audio stream created with {:?} backend", backend);
                        None
                    }
                    Err(e) => {
                        warn!("⚠️ Failed to create system audio stream: {}", e);
                        Some((Some(sys_device.name.clone()), e.to_string()))
                    }
                }
            }
            Err(reason) => Some((None, reason)),
        };

        // Ensure at least one stream was created
        if self.microphone_stream.is_none() && self.system_stream.is_none() {
            return Err(anyhow::anyhow!("No audio streams could be created"));
        }

        if let Some((device_name, reason)) = system_unavailable {
            self.state.report_system_audio_unavailable(device_name, reason);
        }

        Ok(())
    }

    /// Stop all audio streams (off the async runtime; see `AudioStream::stop_off_runtime`).
    pub async fn stop_streams(&mut self) {
        info!("Stopping all audio streams");
        for stream in [self.microphone_stream.take(), self.system_stream.take()].into_iter().flatten() {
            stream.stop_off_runtime().await;
        }
        info!("All audio streams stopped");
    }

    fn stream_slot(&mut self, device_type: DeviceType) -> &mut Option<AudioStream> {
        match device_type {
            DeviceType::Microphone => &mut self.microphone_stream,
            DeviceType::System => &mut self.system_stream,
        }
    }

    /// Take one stream OUT of the manager, leaving the other running. The
    /// caller stops it OUTSIDE any lock — a cpal teardown of a dead device can
    /// stall and must not block a held mutex.
    pub fn take_stream(&mut self, device_type: DeviceType) -> Option<AudioStream> {
        self.stream_slot(device_type).take()
    }

    /// Install a replacement stream (after a rebuild or hot-swap).
    pub fn set_stream(&mut self, device_type: DeviceType, stream: AudioStream) {
        *self.stream_slot(device_type) = Some(stream);
    }

    /// Get stream count
    pub fn active_stream_count(&self) -> usize {
        let mut count = 0;
        if self.microphone_stream.is_some() {
            count += 1;
        }
        if self.system_stream.is_some() {
            count += 1;
        }
        count
    }

    /// Check if any streams are active
    pub fn has_active_streams(&self) -> bool {
        self.microphone_stream.is_some() || self.system_stream.is_some()
    }
}

impl Drop for AudioStreamManager {
    fn drop(&mut self) {
        // Normally already stopped via `stop_streams`; this covers early exits.
        for stream in [self.microphone_stream.take(), self.system_stream.take()].into_iter().flatten() {
            stream.stop_contained();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_panicking_teardown_does_not_abort_the_caller() {
        let outcome = run_teardown_off_runtime("mic", Duration::from_secs(5), || panic!("worker join failed")).await;
        assert_eq!(outcome, TeardownOutcome::Panicked);
    }

    #[tokio::test]
    async fn a_hung_teardown_is_abandoned_after_the_timeout() {
        let started = std::time::Instant::now();
        let outcome = run_teardown_off_runtime("mic", Duration::from_millis(50), || {
            std::thread::sleep(Duration::from_millis(500));
            Ok(())
        })
        .await;
        assert_eq!(outcome, TeardownOutcome::TimedOut);
        assert!(started.elapsed() < Duration::from_millis(400));
    }

    #[tokio::test]
    async fn teardown_errors_and_success_are_reported() {
        let failed = run_teardown_off_runtime("mic", Duration::from_secs(5), || Err(anyhow::anyhow!("pause failed"))).await;
        assert_eq!(failed, TeardownOutcome::Failed);
        assert_eq!(run_teardown_off_runtime("mic", Duration::from_secs(5), || Ok(())).await, TeardownOutcome::Stopped);
    }

    #[test]
    fn a_panicking_teardown_in_drop_is_contained() {
        assert_eq!(run_teardown_contained("mic", || panic!("worker join failed")), TeardownOutcome::Panicked);
        assert_eq!(run_teardown_contained("mic", || Ok(())), TeardownOutcome::Stopped);
    }
}
