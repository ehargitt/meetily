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

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let _ = (a, b);
    todo!("track A")
}

/// Compare every centroid with the stored self profile.
pub fn match_self(centroids: &[Vec<f32>], profile: &[f32]) -> SelfMatch {
    let _ = (centroids, profile);
    todo!("track A")
}

/// Fold a centroid into the profile as a speech-seconds-weighted running mean, L2-normalised.
/// Returns the new profile and its accumulated speech seconds.
pub fn update_profile(profile: Option<(&[f32], f64)>, centroid: &[f32], speech_secs: f64) -> (Vec<f32>, f64) {
    let _ = (profile, centroid, speech_secs);
    todo!("track A")
}

/// Little-endian f32 bytes for SQLite BLOB storage.
pub fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Inverse of [`to_blob`]; `None` if the length is not a multiple of 4.
pub fn from_blob(b: &[u8]) -> Option<Vec<f32>> {
    if b.len() % 4 != 0 {
        return None;
    }
    Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
