//! Offline speaker diarization for Meetily.
//!
//! Pipeline: pyannote segmentation-3.0 (powerset, 10 s windows) finds up to three local speakers
//! per window, WeSpeaker ResNet34-LM embeds each local speaker, and average-linkage clustering
//! links them into global speakers. Runs on ONNX Runtime (CPU) through the same `ort` pin as the
//! app, so there is one ONNX Runtime per process.
//!
//! All turn times produced by [`DiarizationEngine::diarize`] are in the audio *file* clock. Use
//! [`timeline::TranscriptClock`] to map them onto Meetily's transcript clock before calling
//! [`assign::assign_segments`].

use std::path::Path;
use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};

pub mod assign;
pub mod models;
pub mod relabel;
pub mod timeline;
pub mod voiceprint;

/// Identifies the model pair and algorithm version that produced stored results.
pub const MODEL_ID: &str = "pyannote-seg3.0+wespeaker-r34lm/v1";
/// Dimension of speaker embeddings and centroids.
pub const EMBEDDING_DIM: usize = 256;
/// Sample rate the engine expects (mono f32 in [-1, 1]).
pub const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, thiserror::Error)]
pub enum DiarizationError {
    #[error("cancelled")]
    Cancelled,
    #[error("diarization models missing: {0:?}")]
    ModelsMissing(Vec<String>),
    #[error("model file failed verification: {0}")]
    ModelCorrupt(String),
    #[error("download failed: {0}")]
    Download(String),
    #[error("onnx runtime error: {0}")]
    Runtime(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, DiarizationError>;

#[derive(Debug, Clone, PartialEq)]
pub struct DiarizationConfig {
    /// Sliding-window step in seconds: 3.0 = fast (default), 1.0 = accurate.
    pub step_secs: f32,
    /// Average-linkage cosine-distance cut.
    pub threshold: f32,
    /// Force an exact number of speakers instead of threshold clustering.
    pub num_speakers: Option<usize>,
    /// Drop turns shorter than this (seconds).
    pub min_duration_on: f32,
    /// Merge same-speaker turns separated by less than this gap (seconds).
    pub min_duration_off: f32,
    /// Fold clusters holding less than this fraction of total speech into the nearest large one.
    pub min_cluster_frac: f32,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self::fast()
    }
}

impl DiarizationConfig {
    pub fn fast() -> Self {
        Self {
            step_secs: 3.0,
            threshold: 0.6,
            num_speakers: None,
            min_duration_on: 0.1,
            min_duration_off: 0.5,
            min_cluster_frac: 0.02,
        }
    }

    pub fn accurate() -> Self {
        Self { step_secs: 1.0, ..Self::fast() }
    }
}

/// One speaker turn in the audio file clock. `speaker` is 0-based, numbered in order of first
/// appearance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

#[derive(Debug, Clone)]
pub struct DiarizationOutput {
    /// Sorted by start; turns of different speakers may overlap.
    pub turns: Vec<SpeakerTurn>,
    /// One L2-normalised [`EMBEDDING_DIM`]-d centroid per speaker, indexed by `SpeakerTurn::speaker`.
    pub centroids: Vec<Vec<f32>>,
    /// Seconds of speech per speaker, indexed by `SpeakerTurn::speaker`.
    pub speech_secs: Vec<f64>,
    pub model_id: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Segmenting,
    Embedding,
    Clustering,
}

/// Loaded ONNX sessions. Load per job and drop afterwards; sessions hold ~1 GB at 8 workers.
pub struct DiarizationEngine {
    _private: (),
}

impl DiarizationEngine {
    /// Load both models from `models_dir` (see [`models::MODELS`]). `workers` bounds the number
    /// of concurrent embedding sessions and the segmentation intra-op threads.
    pub fn load(models_dir: &Path, workers: usize) -> Result<Self> {
        let _ = (models_dir, workers);
        todo!("track A: load segmentation + embedding sessions")
    }

    /// Diarize 16 kHz mono samples. Checks `cancel` between batches and returns
    /// [`DiarizationError::Cancelled`] when set. `progress` receives the stage and a 0..=1
    /// fraction within that stage. Audio shorter than 0.5 s or with no speech yields empty output.
    pub fn diarize(
        &self,
        samples_16k: &[f32],
        cfg: &DiarizationConfig,
        cancel: &AtomicBool,
        progress: &(dyn Fn(Stage, f32) + Sync),
    ) -> Result<DiarizationOutput> {
        let _ = (samples_16k, cfg, cancel, progress);
        todo!("track A: segmentation -> embeddings -> clustering -> reconstruction")
    }
}
