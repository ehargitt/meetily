//! The local user's voiceprint: recognise "Me" across meetings from speaker centroids.

/// Auto-label as self at or above this cosine similarity…
pub const AUTO_THRESHOLD: f32 = 0.50;
/// …when it also beats the runner-up cluster by at least this margin.
pub const AUTO_MARGIN: f32 = 0.15;
/// Offer an "Is this you?" suggestion at or above this similarity.
pub const SUGGEST_THRESHOLD: f32 = 0.35;
/// Minimum speech (seconds) a speaker needs before it can be enrolled as self.
pub const MIN_ENROLL_SECS: f64 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfMatch {
    Auto(usize),
    Suggest(usize),
    None,
}

/// Cosine similarity; 0 when either vector is all zeros.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
}

/// Compare every centroid with the stored self profile. The best match is labelled
/// automatically when it reaches [`AUTO_THRESHOLD`] and beats the runner-up centroid by
/// [`AUTO_MARGIN`] (a lone centroid has no runner-up, so the margin holds); otherwise it is
/// suggested when it reaches [`SUGGEST_THRESHOLD`]. A profile whose dimension differs from the
/// centroids' never matches.
pub fn match_self(centroids: &[Vec<f32>], profile: &[f32]) -> SelfMatch {
    if profile.is_empty() || centroids.iter().any(|c| c.len() != profile.len()) {
        return SelfMatch::None;
    }
    let sims: Vec<f32> = centroids.iter().map(|c| cosine(c, profile)).collect();
    let Some((best, &sim)) = sims.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)) else {
        return SelfMatch::None;
    };
    let runner_up =
        sims.iter().enumerate().filter(|&(i, _)| i != best).fold(f32::NEG_INFINITY, |acc, (_, &s)| acc.max(s));
    if sim >= AUTO_THRESHOLD && sim - runner_up >= AUTO_MARGIN {
        SelfMatch::Auto(best)
    } else if sim >= SUGGEST_THRESHOLD {
        SelfMatch::Suggest(best)
    } else {
        SelfMatch::None
    }
}

/// Fold a centroid into the profile as a speech-seconds-weighted running mean, L2-normalised.
/// Returns the new profile and its accumulated speech seconds. A profile of another dimension
/// (another model) is replaced by the centroid.
pub fn update_profile(profile: Option<(&[f32], f64)>, centroid: &[f32], speech_secs: f64) -> (Vec<f32>, f64) {
    let (mut merged, total) = match profile {
        Some((p, weight)) if p.len() == centroid.len() && weight + speech_secs > 0.0 => {
            let total = weight + speech_secs;
            let merged = p
                .iter()
                .zip(centroid)
                .map(|(a, b)| ((*a as f64 * weight + *b as f64 * speech_secs) / total) as f32)
                .collect();
            (merged, total)
        }
        _ => (centroid.to_vec(), speech_secs),
    };
    l2_normalize(&mut merged);
    (merged, total)
}

/// Little-endian f32 bytes for SQLite BLOB storage.
pub fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Inverse of [`to_blob`]; `None` if the length is not a multiple of 4.
pub fn from_blob(b: &[u8]) -> Option<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        return None;
    }
    Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit vector in the (x, y) plane whose cosine with [`PROFILE`] is `cos`.
    fn at(cos: f32) -> Vec<f32> {
        vec![cos, (1.0 - cos * cos).max(0.0).sqrt(), 0.0]
    }

    const PROFILE: [f32; 3] = [1.0, 0.0, 0.0];

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 3.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 1.0], &[-1.0, -1.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn auto_needs_threshold_and_margin_over_runner_up() {
        assert_eq!(match_self(&[at(0.10), at(0.55)], &PROFILE), SelfMatch::Auto(1));
        // Runner-up too close: 0.55 - 0.45 < 0.15, so only a suggestion.
        assert_eq!(match_self(&[at(0.45), at(0.55)], &PROFILE), SelfMatch::Suggest(1));
        assert_eq!(match_self(&[at(0.34), at(0.50)], &PROFILE), SelfMatch::Auto(1));
        assert_eq!(match_self(&[at(0.36), at(0.50)], &PROFILE), SelfMatch::Suggest(1));
        assert_eq!(match_self(&[at(0.49), at(0.20)], &PROFILE), SelfMatch::Suggest(0));
    }

    #[test]
    fn single_centroid_has_no_runner_up() {
        assert_eq!(match_self(&[at(0.50)], &PROFILE), SelfMatch::Auto(0));
        assert_eq!(match_self(&[at(0.49)], &PROFILE), SelfMatch::Suggest(0));
    }

    #[test]
    fn suggestion_band_and_below() {
        assert_eq!(match_self(&[at(0.35), at(0.0)], &PROFILE), SelfMatch::Suggest(0));
        assert_eq!(match_self(&[at(0.34), at(0.0)], &PROFILE), SelfMatch::None);
        assert_eq!(match_self(&[], &PROFILE), SelfMatch::None);
        assert_eq!(match_self(&[vec![1.0, 0.0]], &PROFILE), SelfMatch::None, "dimension mismatch");
    }

    #[test]
    fn update_profile_is_weighted_mean_normalised() {
        let (p, secs) = update_profile(None, &[3.0, 4.0], 12.0);
        assert_eq!(secs, 12.0);
        assert!((p[0] - 0.6).abs() < 1e-6 && (p[1] - 0.8).abs() < 1e-6);

        // 30 s of [1, 0] plus 10 s of [0, 1] points along (0.75, 0.25).
        let (p, secs) = update_profile(Some((&[1.0, 0.0], 30.0)), &[0.0, 1.0], 10.0);
        assert_eq!(secs, 40.0);
        let norm = (0.75f32 * 0.75 + 0.25 * 0.25).sqrt();
        assert!((p[0] - 0.75 / norm).abs() < 1e-6 && (p[1] - 0.25 / norm).abs() < 1e-6);
    }

    #[test]
    fn blob_round_trip() {
        let v = vec![0.0, -1.5, 3.25, f32::MIN_POSITIVE, 1e-7];
        let blob = to_blob(&v);
        assert_eq!(blob.len(), v.len() * 4);
        assert_eq!(from_blob(&blob), Some(v));
        assert_eq!(from_blob(&[0, 0, 0]), None);
    }
}
