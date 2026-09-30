use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::AtomicU64;

lazy_static! {
    pub static ref LAST_AUDIO_CAPTURE: AtomicU64 = AtomicU64::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
}

#[derive(Clone, Debug, PartialEq)]
pub enum AudioTranscriptionEngine {
    Deepgram,
    WhisperTiny,
    WhisperDistilLargeV3,
    WhisperLargeV3Turbo,
    WhisperLargeV3,
}

impl fmt::Display for AudioTranscriptionEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioTranscriptionEngine::Deepgram => write!(f, "Deepgram"),
            AudioTranscriptionEngine::WhisperTiny => write!(f, "WhisperTiny"),
            AudioTranscriptionEngine::WhisperDistilLargeV3 => write!(f, "WhisperLarge"),
            AudioTranscriptionEngine::WhisperLargeV3Turbo => write!(f, "WhisperLargeV3Turbo"),
            AudioTranscriptionEngine::WhisperLargeV3 => write!(f, "WhisperLargeV3"),
        }
    }
}

impl Default for AudioTranscriptionEngine {
    fn default() -> Self {
        AudioTranscriptionEngine::WhisperLargeV3Turbo
    }
}

#[derive(Clone, Debug)]
pub struct DeviceControl {
    pub is_running: bool,
    pub is_paused: bool,
}

#[derive(Clone, Eq, PartialEq, Hash, Serialize, Debug, Deserialize)]
pub enum DeviceType {
    Input,
    Output,
}

#[derive(Clone, Eq, PartialEq, Hash, Serialize, Debug)]
pub struct AudioDevice {
    pub name: String,
    pub device_type: DeviceType,
}

impl AudioDevice {
    pub fn new(name: String, device_type: DeviceType) -> Self {
        AudioDevice { name, device_type }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        if name.trim().is_empty() {
            return Err(anyhow!("Device name cannot be empty"));
        }

        let (name, device_type) = if name.to_lowercase().ends_with("(input)") {
            (
                name.trim_end_matches("(input)").trim().to_string(),
                DeviceType::Input,
            )
        } else if name.to_lowercase().ends_with("(output)") {
            (
                name.trim_end_matches("(output)").trim().to_string(),
                DeviceType::Output,
            )
        } else {
            return Err(anyhow!(
                "Device type (input/output) not specified in the name"
            ));
        };

        Ok(AudioDevice::new(name, device_type))
    }
}

impl fmt::Display for AudioDevice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} ({})",
            self.name,
            match self.device_type {
                DeviceType::Input => "input",
                DeviceType::Output => "output",
            }
        )
    }
}

/// Parse audio device from string name
pub fn parse_audio_device(name: &str) -> Result<AudioDevice> {
    AudioDevice::from_name(name)
}

/// A wedged sound server can block device lookup indefinitely; give up after this.
const DEVICE_LOOKUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Get device and config for audio operations.
///
/// The lookup is blocking (cpal talks to the sound server), so it runs on a
/// blocking thread and is abandoned after `DEVICE_LOOKUP_TIMEOUT`.
pub async fn get_device_and_config(
    audio_device: &AudioDevice,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    let requested = audio_device.clone();
    let lookup = tokio::task::spawn_blocking(move || find_device_and_config(&requested));
    match tokio::time::timeout(DEVICE_LOOKUP_TIMEOUT, lookup).await {
        Ok(joined) => joined.map_err(|e| anyhow!("Device lookup for '{}' failed: {}", audio_device.name, e))?,
        Err(_) => Err(anyhow!(
            "Timed out after {:?} looking up audio device '{}' (sound server not responding)",
            DEVICE_LOOKUP_TIMEOUT,
            audio_device.name
        )),
    }
}

fn find_device_and_config(
    audio_device: &AudioDevice,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    #[cfg(target_os = "windows")]
    {
        return super::platform::get_windows_device(audio_device);
    }

    #[cfg(target_os = "linux")]
    {
        return find_linux_capture_device(audio_device);
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        use cpal::traits::{DeviceTrait, HostTrait};

        let host = cpal::default_host();

        match audio_device.device_type {
            DeviceType::Input => {
                for device in host.input_devices()? {
                    if let Ok(name) = device.name() {
                        if name == audio_device.name {
                            let default_config = device
                                .default_input_config()
                                .map_err(|e| anyhow!("Failed to get default input config: {}", e))?;
                            return Ok((device, default_config));
                        }
                    }
                }
            }
            // Use default host for all macOS output devices
            // Core Audio backend uses direct cidre API for system capture, not cpal
            DeviceType::Output if cfg!(target_os = "macos") => {
                for device in host.output_devices()? {
                    if let Ok(name) = device.name() {
                        if name == audio_device.name {
                            let default_config = device
                                .default_output_config()
                                .map_err(|e| anyhow!("Failed to get output config: {}", e))?;
                            return Ok((device, default_config));
                        }
                    }
                }
            }
            DeviceType::Output => {}
        }

        Err(anyhow!("Device not found: {}", audio_device.name))
    }
}

/// Linux: both microphones and system audio (monitor sources) are ALSA
/// capture PCMs.
///
/// cpal's ALSA enumeration opens every PCM it steps over in both directions
/// (including `dmix`, which starts the card's playback hardware), so the name
/// is first checked against ALSA's name hints and a miss fails without any
/// enumeration. `default` needs none: cpal hands it out unopened. Any other
/// name is found by enumerating lazily and stopping at the match.
#[cfg(target_os = "linux")]
fn find_linux_capture_device(
    audio_device: &AudioDevice,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let pcm_name = super::platform::resolve_capture_pcm(audio_device)?;
    let host = cpal::default_host();

    let device = if pcm_name == "default" {
        host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device"))?
    } else {
        host.devices()?
            .find(|device| device.name().map_or(false, |name| name == pcm_name))
            .ok_or_else(|| anyhow!("ALSA lists '{}' but it could not be opened", pcm_name))?
    };

    let config = device
        .default_input_config()
        .map_err(|e| anyhow!("Failed to get input config for '{}': {}", pcm_name, e))?;
    Ok((device, config))
}
