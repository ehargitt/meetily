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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use cluster::{ClusterParams, Embedded};
use embedding::{embed_jobs, EmbedJob, Embedder};
use segmentation::Segmenter;

pub mod assign;
pub mod models;
pub mod relabel;
pub mod timeline;
pub mod voiceprint;

mod cluster;
mod embedding;
mod fbank;
mod reconstruct;
mod segmentation;

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

impl From<ort::Error> for DiarizationError {
    fn from(e: ort::Error) -> Self {
        DiarizationError::Runtime(e.to_string())
    }
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
    segmenter: Segmenter,
    embedder: Embedder,
    /// Dedicated pool so embedding work never competes with the app's global rayon pool; its
    /// thread `i` uses embedding session `i`.
    pool: rayon::ThreadPool,
}

impl DiarizationEngine {
    /// Load both models from `models_dir` (see [`models::MODELS`]). `workers` bounds the number
    /// of concurrent embedding sessions and the segmentation intra-op threads.
    ///
    /// Does not initialise the ONNX Runtime environment; the host app may already have done so.
    pub fn load(models_dir: &Path, workers: usize) -> Result<Self> {
        let workers = workers.max(1);
        let missing = models::missing(models_dir);
        if !missing.is_empty() {
            return Err(DiarizationError::ModelsMissing(missing.iter().map(|m| m.file_name.to_string()).collect()));
        }
        let started = Instant::now();
        let segmenter = Segmenter::load(&models_dir.join(models::SEGMENTATION.file_name), workers)?;
        let embedder = Embedder::load(&models_dir.join(models::EMBEDDING.file_name), workers)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("diarization-{i}"))
            .build()
            .map_err(|e| DiarizationError::Runtime(format!("failed to start diarization thread pool: {e}")))?;
        log::info!(
            "Diarization models loaded from {} with {} workers in {:.2}s",
            models_dir.display(),
            workers,
            started.elapsed().as_secs_f64()
        );
        Ok(Self { segmenter, embedder, pool })
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
        if !(cfg.step_secs.is_finite() && cfg.step_secs > 0.0) {
            return Err(DiarizationError::InvalidInput(format!("step_secs must be positive, got {}", cfg.step_secs)));
        }
        if cfg.num_speakers == Some(0) {
            return Err(DiarizationError::InvalidInput("num_speakers must be at least 1".into()));
        }
        if samples_16k.len() < SAMPLE_RATE as usize / 2 {
            return Ok(DiarizationOutput::empty());
        }
        let started = Instant::now();
        let step = ((cfg.step_secs * SAMPLE_RATE as f32).round() as usize).max(1);
        let meta = self.segmenter.meta;

        let chunks = self.segmenter.run(samples_16k, step, cancel, progress)?;
        let segmented = started.elapsed().as_secs_f64();

        let jobs = embed_jobs(&chunks, meta.window_size);
        let items = self.embed_all(samples_16k, &jobs, cancel, progress)?;
        let embedded = started.elapsed().as_secs_f64();
        if items.is_empty() {
            return Ok(DiarizationOutput::empty());
        }

        progress(Stage::Clustering, 0.0);
        let params = ClusterParams {
            threshold: cfg.threshold,
            num_speakers: cfg.num_speakers,
            min_cluster_size: cluster::min_cluster_size(cfg.step_secs),
            min_cluster_frac: cfg.min_cluster_frac,
        };
        let cents = cluster::global_centroids(&items, &params);
        let chunk_map = cluster::assign_chunks(&items, chunks.len(), &cents);
        let active = reconstruct::global_activity(&chunks, &chunk_map, cents.len(), step, &meta, samples_16k.len());
        let turns = reconstruct::activity_to_turns(
            &active,
            cents.len(),
            &meta,
            cfg.min_duration_on as f64,
            cfg.min_duration_off as f64,
        );
        let (turns, centroids, speech_secs) = reconstruct::renumber(turns, &cents);
        progress(Stage::Clustering, 1.0);

        log::info!(
            "Diarized {:.1}s of audio: {} speakers, {} turns from {} embeddings (segmentation {:.2}s, embedding {:.2}s, total {:.2}s)",
            samples_16k.len() as f64 / SAMPLE_RATE as f64,
            centroids.len(),
            turns.len(),
            items.len(),
            segmented,
            embedded - segmented,
            started.elapsed().as_secs_f64()
        );
        Ok(DiarizationOutput { turns, centroids, speech_secs, model_id: MODEL_ID })
    }

    /// Embed every job on the dedicated pool, checking `cancel` before each one.
    fn embed_all(
        &self,
        samples: &[f32],
        jobs: &[EmbedJob],
        cancel: &AtomicBool,
        progress: &(dyn Fn(Stage, f32) + Sync),
    ) -> Result<Vec<Embedded>> {
        progress(Stage::Embedding, 0.0);
        let done = AtomicUsize::new(0);
        let results: Vec<Option<Embedded>> = self.pool.install(|| {
            jobs.par_iter()
                .map(|job| {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(DiarizationError::Cancelled);
                    }
                    let worker = rayon::current_thread_index().unwrap_or(0);
                    let emb = self.embedder.embed(samples, &job.ranges, worker)?;
                    let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                    progress(Stage::Embedding, finished as f32 / jobs.len() as f32);
                    Ok(emb.map(|emb| Embedded {
                        chunk: job.chunk,
                        local: job.local,
                        clean_frames: job.clean_frames,
                        emb,
                    }))
                })
                .collect::<Result<_>>()
        })?;
        progress(Stage::Embedding, 1.0);
        Ok(results.into_iter().flatten().collect())
    }
}

impl DiarizationOutput {
    fn empty() -> Self {
        Self { turns: Vec::new(), centroids: Vec::new(), speech_secs: Vec::new(), model_id: MODEL_ID }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_can_move_into_a_blocking_task() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DiarizationEngine>();
    }
}
