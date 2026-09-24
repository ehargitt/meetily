//! Stable speaker keys (`S1`, `S2`, …) across re-runs, so renamed speakers keep their names.

use std::collections::HashMap;

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
///
/// "Unused" excludes every key in `prev`, so a new voice never inherits the name given to a
/// previous speaker it did not match.
pub fn stable_keys(new_turns: &[SpeakerTurn], num_speakers: usize, prev: &[PreviousTurn]) -> Vec<String> {
    let mut prev_keys: Vec<&str> = prev.iter().map(|p| p.speaker_key.as_str()).collect();
    prev_keys.sort_by_key(|k| (parse_speaker_key(k), *k));
    prev_keys.dedup();
    let index: HashMap<&str, usize> = prev_keys.iter().enumerate().map(|(j, k)| (*k, j)).collect();

    // overlap[i][j]: seconds new speaker i shares with previous key j.
    let mut overlap = vec![vec![0f64; prev_keys.len()]; num_speakers];
    for t in new_turns.iter().filter(|t| t.speaker < num_speakers) {
        for p in prev {
            let o = t.end.min(p.end) - t.start.max(p.start);
            if o > 0.0 {
                overlap[t.speaker][index[p.speaker_key.as_str()]] += o;
            }
        }
    }
    let mut pairs: Vec<(f64, usize, usize)> = overlap
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().filter(|(_, &o)| o > 0.0).map(move |(j, &o)| (o, i, j)))
        .collect();
    // Largest overlap first; ties by new speaker, then previous key order, for determinism.
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut keys: Vec<Option<String>> = vec![None; num_speakers];
    let mut taken = vec![false; prev_keys.len()];
    for (_, i, j) in pairs {
        if keys[i].is_none() && !taken[j] {
            keys[i] = Some(prev_keys[j].to_string());
            taken[j] = true;
        }
    }

    let mut used: Vec<String> = prev_keys.iter().map(|k| k.to_string()).collect();
    let mut next = 0;
    keys.into_iter()
        .map(|key| {
            key.unwrap_or_else(|| {
                while used.contains(&speaker_key(next)) {
                    next += 1;
                }
                let key = speaker_key(next);
                used.push(key.clone());
                key
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(start: f64, end: f64, speaker: usize) -> SpeakerTurn {
        SpeakerTurn { start, end, speaker }
    }

    fn prev(key: &str, start: f64, end: f64) -> PreviousTurn {
        PreviousTurn { speaker_key: key.into(), start, end }
    }

    #[test]
    fn key_format_round_trips() {
        assert_eq!(speaker_key(0), "S1");
        assert_eq!(parse_speaker_key("S12"), Some(11));
        assert_eq!(parse_speaker_key("S0"), None);
        assert_eq!(parse_speaker_key("X1"), None);
    }

    #[test]
    fn no_previous_turns_gives_sequential_keys() {
        let turns = [turn(0.0, 1.0, 0), turn(1.0, 2.0, 1), turn(2.0, 3.0, 2)];
        assert_eq!(stable_keys(&turns, 3, &[]), vec!["S1", "S2", "S3"]);
    }

    #[test]
    fn swapped_clusters_keep_their_keys() {
        let previous = [prev("S1", 0.0, 10.0), prev("S2", 10.0, 20.0)];
        // The re-run numbers the voices the other way round.
        let turns = [turn(0.5, 9.5, 1), turn(10.5, 19.0, 0)];
        assert_eq!(stable_keys(&turns, 2, &previous), vec!["S2", "S1"]);
    }

    #[test]
    fn mapping_is_one_to_one_and_new_voices_avoid_old_keys() {
        let previous = [prev("S1", 0.0, 10.0), prev("S2", 10.0, 12.0), prev("S3", 30.0, 40.0)];
        // Speakers 0 and 1 both overlap S1 most; 0 overlaps it more, so 1 falls back to S2.
        // Speaker 2 overlaps nothing and must not inherit the unmatched S3.
        let turns = [turn(0.0, 8.0, 0), turn(8.0, 11.0, 1), turn(50.0, 55.0, 2)];
        assert_eq!(stable_keys(&turns, 3, &previous), vec!["S1", "S2", "S4"]);
    }

    #[test]
    fn speaker_without_turns_still_gets_a_key() {
        let previous = [prev("S2", 0.0, 5.0)];
        let turns = [turn(0.0, 5.0, 1)];
        assert_eq!(stable_keys(&turns, 2, &previous), vec!["S1", "S2"]);
    }
}
