//! Assign diarized speakers to transcript segments.

use crate::SpeakerTurn;

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentSpan {
    pub transcript_id: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub transcript_id: String,
    pub speaker: usize,
    /// Fraction (0..=1) of the segment covered by the chosen speaker; 0 when assigned by the
    /// nearest-turn fallback.
    pub overlap: f64,
}

/// Give each segment the speaker with the largest temporal overlap. Segments with no overlap take
/// the nearest turn within 1 s; otherwise they are left out. `turns` must already be in the
/// transcript clock.
pub fn assign_segments(segs: &[SegmentSpan], turns: &[SpeakerTurn]) -> Vec<Assignment> {
    let _ = (segs, turns);
    todo!("track A")
}
