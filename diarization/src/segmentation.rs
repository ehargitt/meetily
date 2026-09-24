//! pyannote segmentation-3.0 (sherpa-onnx ONNX export): sliding-window inference and powerset
//! decoding into per-frame local-speaker activity.
//!
//! Model input `x (N, 1, T)` is 16 kHz audio with T = 160 000 (10 s); output `y (N, 589, 7)` holds
//! log-probabilities over the powerset classes {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2} of three local
//! speakers. One output frame covers 270 samples with a 991-sample receptive field.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::{DiarizationError, Result, Stage};

/// Local speakers per window.
pub(crate) const NUM_LOCAL: usize = 3;
const NUM_CLASSES: usize = 7;
/// Windows per ONNX run; bounds the input tensor to ~10 MB.
const BATCH: usize = 16;

/// Per-frame activity of the three local speakers (0 or 1).
pub(crate) type Frame = [u8; NUM_LOCAL];

/// Window geometry read from the model's metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SegMeta {
    pub window_size: usize,
    pub frame_shift: usize,
    pub receptive_field: usize,
}

impl SegMeta {
    /// Centre of output frame `f` (window-relative or global) in seconds.
    pub(crate) fn frame_to_secs(&self, f: usize) -> f64 {
        let sr = crate::SAMPLE_RATE as f64;
        f as f64 * self.frame_shift as f64 / sr + 0.5 * self.receptive_field as f64 / sr
    }
}

/// One window after powerset decoding.
pub(crate) struct ChunkLabels {
    pub start_sample: usize,
    pub frames: Vec<Frame>,
}

/// Powerset class → local-speaker activity, in `pyannote.audio.utils.powerset.Powerset` order.
pub(crate) const POWERSET: [Frame; NUM_CLASSES] =
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]];

/// Window start offsets: full windows every `step` samples plus a zero-padded last window when the
/// audio does not end on a step boundary (sherpa-onnx convention).
pub(crate) fn chunk_starts(n: usize, window: usize, step: usize) -> Vec<usize> {
    if n <= window {
        return vec![0];
    }
    let num = (n - window) / step + 1;
    let mut starts: Vec<usize> = (0..num).map(|i| i * step).collect();
    if !(n - window).is_multiple_of(step) {
        starts.push(num * step);
    }
    starts
}

/// Argmax over each frame's class log-probabilities, mapped through [`POWERSET`].
pub(crate) fn decode_powerset(logits: &[f32]) -> Vec<Frame> {
    logits
        .chunks_exact(NUM_CLASSES)
        .map(|row| {
            let (arg, _) =
                row.iter().enumerate().fold((0, f32::MIN), |best, (i, &v)| if v > best.1 { (i, v) } else { best });
            POWERSET[arg]
        })
        .collect()
}

pub(crate) struct Segmenter {
    session: Mutex<Session>,
    pub meta: SegMeta,
}

fn meta_usize(session: &Session, key: &str) -> Result<usize> {
    let metadata = session.metadata()?;
    let value = metadata
        .custom(key)?
        .ok_or_else(|| DiarizationError::Runtime(format!("segmentation model lacks metadata '{key}'")))?;
    value
        .parse()
        .map_err(|_| DiarizationError::Runtime(format!("segmentation metadata '{key}' is not a number: {value}")))
}

impl Segmenter {
    pub(crate) fn load(model: &Path, intra_threads: usize) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_intra_threads(intra_threads)?
            .commit_from_file(model)?;
        let meta = SegMeta {
            window_size: meta_usize(&session, "window_size")?,
            frame_shift: meta_usize(&session, "receptive_field_shift")?,
            receptive_field: meta_usize(&session, "receptive_field_size")?,
        };
        if meta_usize(&session, "num_speakers")? != NUM_LOCAL || meta_usize(&session, "num_classes")? != NUM_CLASSES {
            return Err(DiarizationError::Runtime("unexpected segmentation powerset layout".into()));
        }
        Ok(Self { session: Mutex::new(session), meta })
    }

    /// Decode every window of `audio`, checking `cancel` between batches.
    pub(crate) fn run(
        &self,
        audio: &[f32],
        step: usize,
        cancel: &AtomicBool,
        progress: &(dyn Fn(Stage, f32) + Sync),
    ) -> Result<Vec<ChunkLabels>> {
        let w = self.meta.window_size;
        let starts = chunk_starts(audio.len(), w, step);
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(starts.len());
        for group in starts.chunks(BATCH) {
            if cancel.load(Ordering::Relaxed) {
                return Err(DiarizationError::Cancelled);
            }
            let mut buf = vec![0f32; group.len() * w];
            for (b, &s) in group.iter().enumerate() {
                let end = (s + w).min(audio.len());
                buf[b * w..b * w + (end - s)].copy_from_slice(&audio[s..end]);
            }
            let input = TensorRef::from_array_view(([group.len(), 1usize, w], buf.as_slice()))?;
            let outputs = session.run(ort::inputs!["x" => input])?;
            let (shape, data) = outputs["y"].try_extract_tensor::<f32>()?;
            let (nb, nf, nc) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            if nb != group.len() || nc != NUM_CLASSES {
                return Err(DiarizationError::Runtime(format!("unexpected segmentation output shape {shape:?}")));
            }
            for (b, &s) in group.iter().enumerate() {
                let frames = decode_powerset(&data[b * nf * nc..(b + 1) * nf * nc]);
                out.push(ChunkLabels { start_sample: s, frames });
            }
            progress(Stage::Segmenting, out.len() as f32 / starts.len() as f32);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powerset_table_matches_pyannote_order() {
        // {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}
        let expected: [&[usize]; 7] = [&[], &[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]];
        for (class, speakers) in expected.iter().enumerate() {
            let active: Vec<usize> = (0..NUM_LOCAL).filter(|&s| POWERSET[class][s] == 1).collect();
            assert_eq!(&active, speakers, "class {class}");
        }
    }

    #[test]
    fn decode_takes_argmax_per_frame() {
        let mut logits = vec![-5.0f32; 3 * NUM_CLASSES];
        logits[0] = 0.0; // frame 0 -> {}
        logits[NUM_CLASSES + 5] = 0.0; // frame 1 -> {0,2}
        logits[2 * NUM_CLASSES + 2] = 0.0; // frame 2 -> {1}
        assert_eq!(decode_powerset(&logits), vec![[0, 0, 0], [1, 0, 1], [0, 1, 0]]);
    }

    #[test]
    fn chunking_adds_zero_padded_last_window() {
        let w = 160_000;
        assert_eq!(chunk_starts(1000, w, 48_000), vec![0]);
        assert_eq!(chunk_starts(w, w, 48_000), vec![0]);
        // 10 s + 6 s = two full steps, ends exactly on a boundary.
        assert_eq!(chunk_starts(w + 96_000, w, 48_000), vec![0, 48_000, 96_000]);
        // One sample more needs a padded window.
        assert_eq!(chunk_starts(w + 96_001, w, 48_000), vec![0, 48_000, 96_000, 144_000]);
    }

    #[test]
    fn frame_time_is_shift_plus_half_receptive_field() {
        let meta = SegMeta { window_size: 160_000, frame_shift: 270, receptive_field: 991 };
        assert!((meta.frame_to_secs(0) - 0.5 * 991.0 / 16000.0).abs() < 1e-12);
        let f = 100;
        let expected = f as f64 * 270.0 / 16000.0 + 0.5 * 991.0 / 16000.0;
        assert!((meta.frame_to_secs(f) - expected).abs() < 1e-12);
    }
}
