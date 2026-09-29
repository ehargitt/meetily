use alsa::device_name::HintIter;
use alsa::Direction;
use anyhow::Result;

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
    let hints = HintIter::new_str(None, "pcm")?
        .filter_map(|hint| hint.name.map(|name| (name, hint.direction)));
    Ok(devices_from_hints(hints))
}

/// Map ALSA PCM hints to the device list the app exposes.
///
/// A hint with no direction supports both. Capture-capable hints are
/// microphones, capture hints named like a monitor source are also offered as
/// system audio, and playback-only hints are listed as outputs.
fn devices_from_hints(hints: impl Iterator<Item = (String, Option<Direction>)>) -> Vec<AudioDevice> {
    let hints: Vec<(String, Option<Direction>)> = hints.filter(|(name, _)| name != "null").collect();
    let can_capture = |direction: &Option<Direction>| !matches!(direction, Some(Direction::Playback));

    let mut devices: Vec<AudioDevice> = hints
        .iter()
        .filter(|(_, direction)| can_capture(direction))
        .map(|(name, _)| AudioDevice::new(name.clone(), DeviceType::Input))
        .collect();

    // PulseAudio/PipeWire monitor sources for system audio
    for (name, direction) in &hints {
        if can_capture(direction) && name.contains("monitor") {
            devices.push(AudioDevice::new(format!("{} (System Audio)", name), DeviceType::Output));
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
}
