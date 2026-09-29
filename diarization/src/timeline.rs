//! Mapping between the audio file clock and Meetily's transcript clock.
//!
//! Live recordings are saved as 30 s AAC checkpoints joined with `ffmpeg -f concat -c copy`.
//! Each checkpoint keeps 1024 priming samples and is padded to a whole AAC frame, so decoded
//! file time runs ahead of transcript time by ~21 ms at the start plus ~37.3 ms per checkpoint
//! (~4.5 s per hour). Retranscribed and imported meetings are already in file time.

use std::path::Path;

use crate::SpeakerTurn;

/// The recorder saves at 48 kHz (`recording_saver.rs`).
const DEFAULT_SAMPLE_RATE: u32 = 48_000;
/// The incremental saver flushes a checkpoint every 30 s of audio.
const CHECKPOINT_SECS: u64 = 30;
/// AAC encoder delay (priming samples) at the start of every checkpoint file.
const AAC_PRIMING: u64 = 1024;
/// Samples per AAC frame; each checkpoint is padded to a whole number of frames.
const AAC_FRAME: u64 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptClock {
    /// Transcript times equal file times.
    File,
    /// Live recording made of `checkpoint_samples`-long AAC checkpoints at `sample_rate`.
    Checkpointed { sample_rate: u32, checkpoint_samples: u64 },
}

/// Decide the clock from the recording folder's `metadata.json`: `retranscribed_at` present or
/// `source == "import"` → `File`; otherwise `Checkpointed` with `sample_rate` from the metadata
/// (default 48000) and 30 s checkpoints. A missing/unreadable file means `Checkpointed` at 48 kHz.
pub fn detect_clock(recording_folder: &Path) -> TranscriptClock {
    let path = recording_folder.join("metadata.json");
    let metadata = match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str::<serde_json::Value>(&text).unwrap_or_else(|e| {
            log::warn!("Ignoring unparseable {}: {}", path.display(), e);
            serde_json::Value::Null
        }),
        Err(e) => {
            log::warn!("Cannot read {}: {}; assuming a live recording", path.display(), e);
            serde_json::Value::Null
        }
    };
    let retranscribed = metadata.get("retranscribed_at").is_some_and(|v| !v.is_null());
    let imported = metadata.get("source").and_then(|v| v.as_str()) == Some("import");
    if retranscribed || imported {
        return TranscriptClock::File;
    }
    let sample_rate = metadata
        .get("sample_rate")
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&sr| sr > 0)
        .unwrap_or(DEFAULT_SAMPLE_RATE);
    TranscriptClock::Checkpointed { sample_rate, checkpoint_samples: sample_rate as u64 * CHECKPOINT_SECS }
}

impl TranscriptClock {
    /// Map a file-clock time (seconds) to the transcript clock.
    ///
    /// Checkpoint `k` occupies `P = priming + ceil(N / 1024) * 1024` decoded samples, of which
    /// the `N` after the priming are real audio. Priming and padding collapse onto the nearest
    /// checkpoint edge.
    pub fn file_to_transcript(&self, t: f64) -> f64 {
        match *self {
            TranscriptClock::File => t,
            TranscriptClock::Checkpointed { sample_rate, checkpoint_samples } => {
                let sr = sample_rate as f64;
                let n = checkpoint_samples as f64;
                let period = (AAC_PRIMING + checkpoint_samples.div_ceil(AAC_FRAME) * AAC_FRAME) as f64;
                let s = t.max(0.0) * sr;
                let k = (s / period).floor();
                let within = (s - k * period - AAC_PRIMING as f64).clamp(0.0, n);
                (k * n + within) / sr
            }
        }
    }

    /// Map every turn's start/end with [`Self::file_to_transcript`].
    pub fn map_turns(&self, turns: &[SpeakerTurn]) -> Vec<SpeakerTurn> {
        turns
            .iter()
            .map(|t| SpeakerTurn {
                start: self.file_to_transcript(t.start),
                end: self.file_to_transcript(t.end),
                speaker: t.speaker,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: TranscriptClock = TranscriptClock::Checkpointed { sample_rate: 48_000, checkpoint_samples: 1_440_000 };

    #[test]
    fn measured_click_onsets_map_back_to_transcript_time() {
        // Clicks placed at 0.5/30.5/60.5/90.5 s decode at these file times after the concat.
        for (file, expected) in [(0.5213, 0.5), (30.5587, 30.5), (60.5960, 60.5), (90.6333, 90.5)] {
            let mapped = LIVE.file_to_transcript(file);
            assert!((mapped - expected).abs() < 0.001, "{file} -> {mapped}, expected {expected}");
        }
    }

    #[test]
    fn file_clock_is_identity() {
        for t in [0.0, 0.5213, 1234.5678] {
            assert_eq!(TranscriptClock::File.file_to_transcript(t), t);
        }
        let turns = [SpeakerTurn { start: 1.0, end: 2.0, speaker: 3 }];
        assert_eq!(TranscriptClock::File.map_turns(&turns), turns.to_vec());
    }

    #[test]
    fn map_turns_maps_both_ends() {
        let turns = [SpeakerTurn { start: 30.5587, end: 60.5960, speaker: 1 }];
        let mapped = LIVE.map_turns(&turns);
        assert!((mapped[0].start - 30.5).abs() < 0.001 && (mapped[0].end - 60.5).abs() < 0.001);
        assert_eq!(mapped[0].speaker, 1);
    }

    fn folder_with(metadata: Option<&str>) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        if let Some(text) = metadata {
            std::fs::write(dir.path().join("metadata.json"), text).expect("write metadata");
        }
        dir
    }

    #[test]
    fn detect_clock_for_live_recording() {
        let dir = folder_with(Some(r#"{"audio_file":"audio.mp4","sample_rate":48000,"status":"completed"}"#));
        assert_eq!(detect_clock(dir.path()), LIVE);
        let dir = folder_with(Some(r#"{"sample_rate":44100}"#));
        assert_eq!(
            detect_clock(dir.path()),
            TranscriptClock::Checkpointed { sample_rate: 44_100, checkpoint_samples: 1_323_000 }
        );
    }

    #[test]
    fn detect_clock_for_retranscribed_and_imported() {
        let dir = folder_with(Some(r#"{"sample_rate":48000,"retranscribed_at":"2026-09-24T13:27:50Z"}"#));
        assert_eq!(detect_clock(dir.path()), TranscriptClock::File);
        let dir = folder_with(Some(r#"{"source":"import"}"#));
        assert_eq!(detect_clock(dir.path()), TranscriptClock::File);
    }

    #[test]
    fn detect_clock_defaults_to_live_48k() {
        assert_eq!(detect_clock(folder_with(None).path()), LIVE);
        assert_eq!(detect_clock(folder_with(Some("not json")).path()), LIVE);
        assert_eq!(detect_clock(folder_with(Some(r#"{"retranscribed_at":null}"#)).path()), LIVE);
    }
}
