use alsa::device_name::HintIter;
use alsa::Direction;
use anyhow::{anyhow, Result};

use crate::audio::devices::configuration::{AudioDevice, DeviceType};

/// List Linux audio devices from ALSA name hints without opening any PCM.
///
/// cpal's ALSA enumeration opens every PCM hint in both directions to test it.
/// That includes `dmix:CARD=…`, which configures and starts the card's playback
/// hardware on open. The device monitor lists devices every few seconds while
/// recording, so a USB microphone that another app (e.g. Teams via PipeWire) is
/// capturing from gets its playback interface reconfigured over and over, which
/// can wedge the device until it is unplugged. Name hints come from the ALSA
/// configuration and control interface only, so listing them is side-effect free.
pub fn configure_linux_audio(_host: &cpal::Host) -> Result<Vec<AudioDevice>> {
    Ok(devices_from_hints(pcm_hints()?.into_iter()))
}

/// Suffix the device list adds to monitor sources offered as system audio.
const SYSTEM_AUDIO_SUFFIX: &str = " (System Audio)";

/// ALSA PCM name hints. Reads configuration only; opens no PCM.
fn pcm_hints() -> Result<Vec<(String, Option<Direction>)>> {
    Ok(HintIter::new_str(None, "pcm")?
        .filter_map(|hint| hint.name.map(|name| (name, hint.direction)))
        .collect())
}

/// The first monitor source ALSA knows, as a system-audio device, if any.
///
/// Never the ALSA `default` PCM: on the capture side that is the microphone.
pub fn default_system_audio_device() -> Result<Option<AudioDevice>> {
    Ok(first_system_audio_device(devices_from_hints(pcm_hints()?.into_iter())))
}

fn first_system_audio_device(devices: Vec<AudioDevice>) -> Option<AudioDevice> {
    devices
        .into_iter()
        .find(|d| d.device_type == DeviceType::Output && d.name.ends_with(SYSTEM_AUDIO_SUFFIX))
}

/// Resolve a requested capture device to the ALSA PCM name to open, checked
/// against the name hints so a missing device fails fast without cpal
/// enumerating (and opening) every PCM.
///
/// Microphones resolve by exact name; `default` always resolves (cpal opens it
/// without enumeration). System audio refuses `default`, which on the capture
/// side is the microphone, and tries the exact name before the name with the
/// list's " (System Audio)" suffix stripped.
pub fn resolve_capture_pcm(requested: &AudioDevice) -> Result<String> {
    let capture_hints: Vec<String> = pcm_hints()?
        .into_iter()
        .filter(|(_, direction)| can_capture(direction))
        .map(|(name, _)| name)
        .collect();
    resolve_capture_pcm_from_hints(requested, &capture_hints)
}

fn resolve_capture_pcm_from_hints(requested: &AudioDevice, capture_hints: &[String]) -> Result<String> {
    let name = requested.name.as_str();
    let candidates: Vec<&str> = match requested.device_type {
        DeviceType::Input => vec![name],
        DeviceType::Output => {
            let stripped = name.strip_suffix(SYSTEM_AUDIO_SUFFIX);
            if name == "default" || stripped == Some("default") {
                return Err(anyhow!(
                    "'{}' is the default capture device (the microphone), not a system-audio source",
                    name
                ));
            }
            std::iter::once(name).chain(stripped).collect()
        }
    };

    candidates
        .into_iter()
        .find(|candidate| {
            (*candidate == "default" && requested.device_type == DeviceType::Input)
                || capture_hints.iter().any(|hint| hint == candidate)
        })
        .map(str::to_string)
        .ok_or_else(|| anyhow!("Audio device '{}' is not known to ALSA", name))
}

/// A hint with no direction supports both.
fn can_capture(direction: &Option<Direction>) -> bool {
    !matches!(direction, Some(Direction::Playback))
}

/// Map ALSA PCM hints to the device list the app exposes.
///
/// A hint with no direction supports both. Capture-capable hints are
/// microphones, capture hints named like a monitor source are also offered as
/// system audio, and playback-only hints are listed as outputs.
fn devices_from_hints(hints: impl Iterator<Item = (String, Option<Direction>)>) -> Vec<AudioDevice> {
    let hints: Vec<(String, Option<Direction>)> = hints.filter(|(name, _)| name != "null").collect();

    let mut devices: Vec<AudioDevice> = hints
        .iter()
        .filter(|(_, direction)| can_capture(direction))
        .map(|(name, _)| AudioDevice::new(name.clone(), DeviceType::Input))
        .collect();

    // PulseAudio/PipeWire monitor sources for system audio
    for (name, direction) in &hints {
        if can_capture(direction) && name.contains("monitor") {
            devices.push(AudioDevice::new(format!("{}{}", name, SYSTEM_AUDIO_SUFFIX), DeviceType::Output));
        }
    }

    for (name, direction) in &hints {
        if matches!(direction, Some(Direction::Playback)) && !devices.iter().any(|d| &d.name == name) {
            devices.push(AudioDevice::new(name.clone(), DeviceType::Output));
        }
    }

    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hint(name: &str, direction: Option<Direction>) -> (String, Option<Direction>) {
        (name.to_string(), direction)
    }

    #[test]
    fn classifies_hints_by_direction() {
        let devices = devices_from_hints(
            vec![
                hint("null", None),
                hint("default", None),
                hint("dsnoop:CARD=Mini,DEV=0", Some(Direction::Capture)),
                hint("dmix:CARD=Mini,DEV=0", Some(Direction::Playback)),
                hint("meetily_monitor", None),
            ]
            .into_iter(),
        );

        assert_eq!(
            devices,
            vec![
                AudioDevice::new("default".into(), DeviceType::Input),
                AudioDevice::new("dsnoop:CARD=Mini,DEV=0".into(), DeviceType::Input),
                AudioDevice::new("meetily_monitor".into(), DeviceType::Input),
                AudioDevice::new("meetily_monitor (System Audio)".into(), DeviceType::Output),
                AudioDevice::new("dmix:CARD=Mini,DEV=0".into(), DeviceType::Output),
            ]
        );
    }

    fn device(name: &str, device_type: DeviceType) -> AudioDevice {
        AudioDevice::new(name.to_string(), device_type)
    }

    fn hints(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn system_audio_never_resolves_to_the_default_capture_pcm() {
        let known = hints(&["default", "meetily_monitor"]);
        for name in ["default", "default (System Audio)"] {
            assert!(resolve_capture_pcm_from_hints(&device(name, DeviceType::Output), &known).is_err(), "{name}");
        }
    }

    #[test]
    fn system_audio_tries_the_exact_name_then_strips_the_list_suffix() {
        let requested = device("meetily_monitor (System Audio)", DeviceType::Output);
        assert_eq!(
            resolve_capture_pcm_from_hints(&requested, &hints(&["default", "meetily_monitor"])).unwrap(),
            "meetily_monitor"
        );
        assert_eq!(
            resolve_capture_pcm_from_hints(
                &requested,
                &hints(&["meetily_monitor", "meetily_monitor (System Audio)"])
            )
            .unwrap(),
            "meetily_monitor (System Audio)"
        );
    }

    #[test]
    fn microphones_resolve_by_exact_name_and_default_needs_no_hint() {
        let known = hints(&["dsnoop:CARD=Mini,DEV=0"]);
        assert_eq!(
            resolve_capture_pcm_from_hints(&device("default", DeviceType::Input), &known).unwrap(),
            "default"
        );
        assert_eq!(
            resolve_capture_pcm_from_hints(&device("dsnoop:CARD=Mini,DEV=0", DeviceType::Input), &known).unwrap(),
            "dsnoop:CARD=Mini,DEV=0"
        );
    }

    #[test]
    fn unknown_devices_fail_fast() {
        let known = hints(&["default"]);
        assert!(resolve_capture_pcm_from_hints(&device("hw:CARD=Gone", DeviceType::Input), &known).is_err());
        assert!(resolve_capture_pcm_from_hints(&device("gone (System Audio)", DeviceType::Output), &known).is_err());
    }

    #[test]
    fn default_system_audio_is_the_first_monitor_source() {
        let devices = devices_from_hints(
            vec![hint("default", None), hint("pulse_monitor", None), hint("meetily_monitor", None)].into_iter(),
        );
        assert_eq!(
            first_system_audio_device(devices),
            Some(device("pulse_monitor (System Audio)", DeviceType::Output))
        );
        let no_monitor = devices_from_hints(vec![hint("default", None)].into_iter());
        assert_eq!(first_system_audio_device(no_monitor), None);
    }
}
