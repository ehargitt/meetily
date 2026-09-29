//! WeSpeaker ResNet34-LM speaker embeddings: fbank features in, 256-d embedding out.
//!
//! One embedding is taken per (window, local speaker) from the frames where that speaker talks
//! alone, so overlapped speech does not blur the voiceprint.

use std::path::Path;
use std::sync::Mutex;

use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::fbank::{Fbank, NUM_BINS};
use crate::segmentation::{ChunkLabels, NUM_LOCAL};
use crate::{DiarizationError, Result, EMBEDDING_DIM};

/// Clean (single-speaker) frames a local speaker needs before it is embedded at all.
const MIN_EMBED_FRAMES: usize = 10;

/// Audio of one local speaker in one window, to be embedded.
pub(crate) struct EmbedJob {
    pub chunk: usize,
    pub local: usize,
    /// Frames where this local speaker is the only one active.
    pub clean_frames: usize,
    /// Sample ranges `[start, end)` of those frames in the full recording.
    pub ranges: Vec<(usize, usize)>,
}

/// One job per (window, local speaker) with at least [`MIN_EMBED_FRAMES`] clean frames.
/// `window_size` maps window-relative frames to samples (sherpa-onnx convention).
pub(crate) fn embed_jobs(chunks: &[ChunkLabels], window_size: usize) -> Vec<EmbedJob> {
    let mut jobs = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let nf = chunk.frames.len();
        let to_sample = |k: usize| (k as f64 / nf as f64 * window_size as f64) as usize + chunk.start_sample;
        for s in 0..NUM_LOCAL {
            let active: Vec<bool> = chunk.frames.iter().map(|f| f[s] == 1 && f.iter().sum::<u8>() < 2).collect();
            let clean_frames = active.iter().filter(|&&a| a).count();
            if clean_frames < MIN_EMBED_FRAMES {
                continue;
            }
            let mut ranges = Vec::new();
            let mut start = None;
            for (k, &a) in active.iter().enumerate() {
                match (a, start) {
                    (true, None) => start = Some(k),
                    (false, Some(s0)) => {
                        ranges.push((to_sample(s0), to_sample(k)));
                        start = None;
                    }
                    _ => {}
                }
            }
            if let Some(s0) = start {
                ranges.push((to_sample(s0), to_sample(nf - 1)));
            }
            jobs.push(EmbedJob { chunk: i, local: s, clean_frames, ranges });
        }
    }
    jobs
}

pub(crate) struct Embedder {
    /// `Session::run` takes `&mut self` in ort rc.10, so each worker thread gets its own session.
    pool: Vec<Mutex<Session>>,
    fbank: Fbank,
    input_name: String,
    output_name: String,
}

impl Embedder {
    pub(crate) fn load(model: &Path, sessions: usize) -> Result<Self> {
        let open = || -> Result<Session> {
            Ok(Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([CPUExecutionProvider::default().build()])?
                .with_intra_threads(1)?
                .commit_from_file(model)?)
        };
        let first = open()?;
        let dim = first.metadata()?.custom("output_dim")?;
        if dim.as_deref() != Some(&EMBEDDING_DIM.to_string()) {
            return Err(DiarizationError::Runtime(format!(
                "embedding model output_dim {dim:?}, expected {EMBEDDING_DIM}"
            )));
        }
        let input_name = first.inputs[0].name.clone();
        let output_name = first.outputs[0].name.clone();
        let mut pool = vec![Mutex::new(first)];
        for _ in 1..sessions.max(1) {
            pool.push(Mutex::new(open()?));
        }
        Ok(Self { pool, fbank: Fbank::new(), input_name, output_name })
    }

    /// Embed the concatenated `ranges` of `audio` using session `worker` (a rayon thread index).
    /// Returns `None` when the audio is too short for one fbank frame or the model output is not
    /// finite; such a local speaker is simply not clustered.
    pub(crate) fn embed(&self, audio: &[f32], ranges: &[(usize, usize)], worker: usize) -> Result<Option<Vec<f32>>> {
        let mut buf = Vec::new();
        for &(a, b) in ranges {
            let b = b.min(audio.len());
            if a < b {
                buf.extend_from_slice(&audio[a..b]);
            }
        }
        let (feats, nf) = self.fbank.compute(&buf);
        if nf == 0 {
            return Ok(None);
        }
        let input = TensorRef::from_array_view(([1usize, nf, NUM_BINS], feats.as_slice()))?;
        let mut session = self.pool[worker % self.pool.len()].lock().unwrap_or_else(|e| e.into_inner());
        let outputs = session.run(ort::inputs![self.input_name.as_str() => input])?;
        let (_, data) = outputs[self.output_name.as_str()].try_extract_tensor::<f32>()?;
        if data.len() != EMBEDDING_DIM || !data.iter().all(|x| x.is_finite()) {
            return Ok(None);
        }
        Ok(Some(data.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_use_only_single_speaker_frames() {
        // 20 frames: local 0 alone for 0..12, overlapped with local 1 for 12..16, silent after.
        let mut frames = vec![[0u8; 3]; 20];
        for f in frames.iter_mut().take(12) {
            *f = [1, 0, 0];
        }
        for f in frames.iter_mut().take(16).skip(12) {
            *f = [1, 1, 0];
        }
        let chunks = vec![ChunkLabels { start_sample: 1000, frames }];
        let jobs = embed_jobs(&chunks, 2000); // 100 samples per frame
        assert_eq!(jobs.len(), 1, "local 1 has no clean frames");
        assert_eq!((jobs[0].chunk, jobs[0].local, jobs[0].clean_frames), (0, 0, 12));
        assert_eq!(jobs[0].ranges, vec![(1000, 2200)]);
    }
}
