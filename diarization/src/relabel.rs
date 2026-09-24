//! Stable speaker keys (`S1`, `S2`, …) across re-runs, so renamed speakers keep their names.

use crate::SpeakerTurn;

#[derive(Debug, Clone, PartialEq)]
pub struct PreviousTurn {
    pub speaker_key: String,
    pub start: f64,
    pub end: f64,
}

/// 1-based key for a 0-based speaker index: 0 → "S1".
pub fn speaker_key(index: usize) -> String {
    format!("S{}", index + 1)
}

/// Inverse of [`speaker_key`]: "S1" → Some(0).
pub fn parse_speaker_key(key: &str) -> Option<usize> {
    key.strip_prefix('S')?.parse::<usize>().ok()?.checked_sub(1)
}

/// Keys for the new speakers (`result[i]` is the key for speaker `i`). Each new speaker takes the
/// previous key it overlaps most in time (one-to-one, greedy by overlap); the rest take the
/// lowest unused `S<n>`. With no previous turns this is `speaker_key(i)`.
pub fn stable_keys(new_turns: &[SpeakerTurn], num_speakers: usize, prev: &[PreviousTurn]) -> Vec<String> {
    let _ = (new_turns, num_speakers, prev);
    todo!("track A")
}
