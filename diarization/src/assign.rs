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

/// Largest gap (seconds) between a segment and a turn for the nearest-turn fallback.
const MAX_FALLBACK_GAP: f64 = 1.0;

/// Give each segment the speaker with the largest temporal overlap. Segments with no overlap take
/// the nearest turn within 1 s; otherwise they are left out. `turns` must already be in the
/// transcript clock. Segments with `end <= start` are left out; ties go to the lower speaker.
pub fn assign_segments(segs: &[SegmentSpan], turns: &[SpeakerTurn]) -> Vec<Assignment> {
    let num_speakers = turns.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
    let mut per_speaker = vec![0f64; num_speakers];
    segs.iter()
        .filter(|s| s.end > s.start)
        .filter_map(|s| {
            per_speaker.iter_mut().for_each(|o| *o = 0.0);
            for t in turns {
                per_speaker[t.speaker] += (s.end.min(t.end) - s.start.max(t.start)).max(0.0);
            }
            let best = per_speaker.iter().enumerate().filter(|&(_, &o)| o > 0.0).fold(
                None,
                |best: Option<(usize, f64)>, (i, &o)| match best {
                    Some((_, b)) if o <= b => best,
                    _ => Some((i, o)),
                },
            );
            let (speaker, overlap) = match best {
                Some((speaker, o)) => (speaker, (o / (s.end - s.start)).min(1.0)),
                None => (nearest_turn(s, turns)?, 0.0),
            };
            Some(Assignment { transcript_id: s.transcript_id.clone(), speaker, overlap })
        })
        .collect()
}

/// Speaker of the turn closest to `s` within [`MAX_FALLBACK_GAP`]; the earlier turn on ties.
fn nearest_turn(s: &SegmentSpan, turns: &[SpeakerTurn]) -> Option<usize> {
    turns
        .iter()
        .map(|t| {
            let gap = if t.end < s.start { s.start - t.end } else { t.start - s.end };
            (gap.max(0.0), t.speaker)
        })
        .filter(|&(gap, _)| gap <= MAX_FALLBACK_GAP)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, speaker)| speaker)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(id: &str, start: f64, end: f64) -> SegmentSpan {
        SegmentSpan { transcript_id: id.into(), start, end }
    }

    fn turn(start: f64, end: f64, speaker: usize) -> SpeakerTurn {
        SpeakerTurn { start, end, speaker }
    }

    #[test]
    fn max_overlap_wins_and_is_reported_as_fraction() {
        let turns = [turn(0.0, 4.0, 0), turn(3.0, 10.0, 1)];
        let out = assign_segments(&[seg("a", 2.0, 6.0)], &turns);
        // Speaker 0 covers 2 s, speaker 1 covers 3 s of the 4 s segment.
        assert_eq!(out, vec![Assignment { transcript_id: "a".into(), speaker: 1, overlap: 0.75 }]);
    }

    #[test]
    fn segment_spanning_turns_goes_to_dominant_speaker() {
        let turns = [turn(0.0, 1.0, 0), turn(1.0, 2.0, 1), turn(2.0, 2.5, 0), turn(2.5, 5.0, 1)];
        let out = assign_segments(&[seg("a", 0.0, 5.0)], &turns);
        assert_eq!(out[0].speaker, 1);
        assert!((out[0].overlap - 0.7).abs() < 1e-9);
    }

    #[test]
    fn nearest_turn_within_one_second_is_the_fallback() {
        let turns = [turn(0.0, 1.0, 0), turn(5.0, 6.0, 1)];
        let out = assign_segments(&[seg("near", 4.2, 4.5), seg("far", 2.5, 3.0)], &turns);
        assert_eq!(out, vec![Assignment { transcript_id: "near".into(), speaker: 1, overlap: 0.0 }]);
    }

    #[test]
    fn empty_segments_and_no_turns_are_skipped() {
        let turns = [turn(0.0, 10.0, 0)];
        assert!(assign_segments(&[seg("zero", 2.0, 2.0), seg("neg", 3.0, 2.0)], &turns).is_empty());
        assert!(assign_segments(&[seg("a", 0.0, 1.0)], &[]).is_empty());
    }
}
