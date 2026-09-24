//! From per-window local activity plus the local→global mapping to speaker turns.
//!
//! Windows overlap, so every global frame is covered by several windows. The number of speakers
//! in a frame is the rounded mean count of active locals over those windows; the frame keeps that
//! many global speakers, those with the most aggregated activity (sherpa-onnx
//! `ComputeSpeakersPerFrame` / pyannote `reconstruct`).

use std::collections::HashMap;

use crate::segmentation::{ChunkLabels, SegMeta, NUM_LOCAL};
use crate::SpeakerTurn;

/// Global frame at which window `i` starts.
fn chunk_frame0(i: usize, step: usize, meta: &SegMeta) -> usize {
    ((i * step) as f64 / meta.frame_shift as f64 + 0.5) as usize
}

/// Per global frame: global speaker activity (`frames × k`, row-major) for frames up to the end
/// of the audio. `step` is the window hop in samples.
pub(crate) fn global_activity(
    chunks: &[ChunkLabels],
    chunk_map: &[[Option<usize>; NUM_LOCAL]],
    k: usize,
    step: usize,
    meta: &SegMeta,
    num_samples: usize,
) -> Vec<bool> {
    if chunks.is_empty() || k == 0 {
        return Vec::new();
    }
    let num_frames = (meta.window_size + (chunks.len() - 1) * step) / meta.frame_shift + 1;
    let mut count = vec![0f32; num_frames];
    let mut coverage = vec![0f32; num_frames];
    let mut agg = vec![0u32; num_frames * k];
    for (i, chunk) in chunks.iter().enumerate() {
        let f0 = chunk_frame0(i, step, meta);
        for (f, frame) in chunk.frames.iter().enumerate() {
            let g_frame = f0 + f;
            if g_frame >= num_frames {
                break;
            }
            count[g_frame] += frame.iter().sum::<u8>() as f32;
            coverage[g_frame] += 1.0;
            for (s, &on) in frame.iter().enumerate() {
                if let (1, Some(g)) = (on, chunk_map[i][s]) {
                    agg[g_frame * k + g] += 1;
                }
            }
        }
    }
    let last_frame = (num_samples / meta.frame_shift).min(num_frames - 1);
    let mut active = vec![false; (last_frame + 1) * k];
    let mut order: Vec<usize> = Vec::with_capacity(k);
    for f in 0..=last_frame {
        let speakers = (count[f] / (coverage[f] + 1e-12) + 0.5) as usize;
        if speakers == 0 {
            continue;
        }
        let row = &agg[f * k..(f + 1) * k];
        order.clear();
        order.extend(0..k);
        order.sort_by(|&a, &b| row[b].cmp(&row[a]));
        for &g in order.iter().take(speakers) {
            if row[g] > 0 {
                active[f * k + g] = true;
            }
        }
    }
    active
}

/// Turns from a `frames × k` activity matrix: runs of active frames per speaker, same-speaker
/// gaps shorter than `min_off` merged, turns not longer than `min_on` dropped. Sorted by start.
pub(crate) fn activity_to_turns(
    active: &[bool],
    k: usize,
    meta: &SegMeta,
    min_on: f64,
    min_off: f64,
) -> Vec<SpeakerTurn> {
    if k == 0 {
        return Vec::new();
    }
    let frames = active.len() / k;
    let mut turns = Vec::new();
    for g in 0..k {
        let mut runs: Vec<SpeakerTurn> = Vec::new();
        let mut start = None;
        for f in 0..=frames {
            let on = f < frames && active[f * k + g];
            match (on, start) {
                (true, None) => start = Some(f),
                (false, Some(s0)) => {
                    let run = SpeakerTurn { start: meta.frame_to_secs(s0), end: meta.frame_to_secs(f), speaker: g };
                    match runs.last_mut() {
                        Some(last) if run.start - last.end < min_off => last.end = run.end,
                        _ => runs.push(run),
                    }
                    start = None;
                }
                _ => {}
            }
        }
        turns.extend(runs.into_iter().filter(|t| t.end - t.start > min_on));
    }
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));
    turns
}

/// Renumber speakers by first appearance, reorder `centroids` to match (speakers without turns
/// are dropped) and total each speaker's speech seconds.
pub(crate) fn renumber(
    mut turns: Vec<SpeakerTurn>,
    centroids: &[Vec<f32>],
) -> (Vec<SpeakerTurn>, Vec<Vec<f32>>, Vec<f64>) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut ordered = Vec::new();
    let mut speech = Vec::new();
    for t in turns.iter_mut() {
        let new = *map.entry(t.speaker).or_insert_with(|| {
            ordered.push(centroids[t.speaker].clone());
            speech.push(0.0);
            ordered.len() - 1
        });
        t.speaker = new;
        speech[new] += t.end - t.start;
    }
    (turns, ordered, speech)
}

#[cfg(test)]
mod tests {
    use super::*;

    const META: SegMeta = SegMeta { window_size: 160_000, frame_shift: 270, receptive_field: 991 };
    const FRAME: f64 = 270.0 / 16000.0;

    fn frames_for(secs: f64) -> usize {
        (secs / FRAME).round() as usize
    }

    fn activity(k: usize, frames: usize, spans: &[(usize, usize, usize)]) -> Vec<bool> {
        let mut a = vec![false; frames * k];
        for &(g, s, e) in spans {
            for f in s..e {
                a[f * k + g] = true;
            }
        }
        a
    }

    #[test]
    fn overlapping_locals_give_overlapping_turns() {
        // One window: local 0 for frames 0..300, local 1 for 200..500, mapped to globals 1 and 0.
        let mut frames = vec![[0u8; 3]; 589];
        for (f, fr) in frames.iter_mut().enumerate() {
            fr[0] = (f < 300) as u8;
            fr[1] = (200..500).contains(&f) as u8;
        }
        let chunks = vec![ChunkLabels { start_sample: 0, frames }];
        let map = vec![[Some(1), Some(0), None]];
        let active = global_activity(&chunks, &map, 2, 48_000, &META, 160_000);
        let turns = activity_to_turns(&active, 2, &META, 0.1, 0.5);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker, 1);
        assert_eq!(turns[1].speaker, 0);
        assert!(turns[1].start < turns[0].end, "turns overlap: {turns:?}");
        assert!((turns[0].start - META.frame_to_secs(0)).abs() < 1e-9);
        assert!((turns[1].end - META.frame_to_secs(500)).abs() < 1e-9);
    }

    #[test]
    fn frame_speaker_count_caps_active_globals() {
        // Three windows cover global frames 200..300; only the first hears two speakers there,
        // so the rounded mean count is one and only the better-supported global speaker stays.
        let step = 270 * 100;
        let mut windows = vec![vec![[0u8; 3]; 589]; 3];
        for (w, frames) in windows.iter_mut().enumerate() {
            let local_start = 200 - 100 * w;
            for fr in &mut frames[local_start..local_start + 100] {
                *fr = if w == 0 { [1, 1, 0] } else { [1, 0, 0] };
            }
        }
        let chunks: Vec<ChunkLabels> =
            windows.into_iter().enumerate().map(|(w, frames)| ChunkLabels { start_sample: w * step, frames }).collect();
        let map = vec![[Some(0), Some(1), None], [Some(0), None, None], [Some(0), None, None]];
        let active = global_activity(&chunks, &map, 2, step, &META, 160_000 + 2 * step);
        for f in 200..300 {
            assert!(active[f * 2], "frame {f} speaker 0");
            assert!(!active[f * 2 + 1], "frame {f} speaker 1 dropped");
        }
    }

    #[test]
    fn short_gaps_merge_and_short_turns_drop() {
        let n = 2000;
        let gap_04 = frames_for(0.4);
        let gap_06 = frames_for(0.6);
        let spans = [
            (0, 0, 100),
            (0, 100 + gap_04, 300),             // 0.4 s gap: merged
            (0, 300 + gap_06, 500 + gap_06),    // 0.6 s gap: kept apart
            (0, 1500, 1500 + frames_for(0.08)), // < 0.1 s: dropped
        ];
        let turns = activity_to_turns(&activity(1, n, &spans), 1, &META, 0.1, 0.5);
        assert_eq!(turns.len(), 2, "{turns:?}");
        assert!((turns[0].start - META.frame_to_secs(0)).abs() < 1e-9);
        assert!((turns[0].end - META.frame_to_secs(300)).abs() < 1e-9);
        assert!((turns[1].start - META.frame_to_secs(300 + gap_06)).abs() < 1e-9);
    }

    #[test]
    fn renumber_by_first_appearance() {
        let turns = vec![
            SpeakerTurn { start: 0.0, end: 1.0, speaker: 2 },
            SpeakerTurn { start: 0.5, end: 2.0, speaker: 0 },
            SpeakerTurn { start: 3.0, end: 4.5, speaker: 2 },
        ];
        let cents = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        let (turns, cents, speech) = renumber(turns, &cents);
        assert_eq!(turns.iter().map(|t| t.speaker).collect::<Vec<_>>(), vec![0, 1, 0]);
        assert_eq!(cents, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(speech, vec![2.5, 1.5]);
    }
}
