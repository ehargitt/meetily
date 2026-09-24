//! Mapping between the audio file clock and Meetily's transcript clock.
//!
//! Live recordings are saved as 30 s AAC checkpoints joined with `ffmpeg -f concat -c copy`.
//! Each checkpoint keeps 1024 priming samples and is padded to a whole AAC frame, so decoded
//! file time runs ahead of transcript time by ~21 ms at the start plus ~37.3 ms per checkpoint
//! (~4.5 s per hour). Retranscribed and imported meetings are already in file time.

use std::path::Path;

use crate::SpeakerTurn;

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
    let _ = recording_folder;
    todo!("track A")
}

impl TranscriptClock {
    /// Map a file-clock time (seconds) to the transcript clock.
    pub fn file_to_transcript(&self, t: f64) -> f64 {
        let _ = t;
        todo!("track A")
    }

    /// Map every turn's start/end with [`Self::file_to_transcript`].
    pub fn map_turns(&self, turns: &[SpeakerTurn]) -> Vec<SpeakerTurn> {
        let _ = turns;
        todo!("track A")
    }
}
