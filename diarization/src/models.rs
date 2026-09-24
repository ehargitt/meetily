//! Model artifacts: commit-pinned URLs, sizes and sha256, plus verified download.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    pub file_name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    /// Short license name shown in the UI (the embedding model requires attribution).
    pub license: &'static str,
    pub attribution: &'static str,
}

pub const SEGMENTATION: ModelSpec = ModelSpec {
    file_name: "segmentation-3.0.onnx",
    url: "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.onnx",
    sha256: "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    size_bytes: 5_992_913,
    license: "MIT",
    attribution: "pyannote segmentation-3.0 (c) 2022 CNRS, MIT License",
};

pub const EMBEDDING: ModelSpec = ModelSpec {
    file_name: "wespeaker_en_voxceleb_resnet34_LM.onnx",
    url: "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/0743f301363dec56491a490f6d6cbc9d67f9a3bf/wespeaker_en_voxceleb_resnet34_LM.onnx",
    sha256: "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
    size_bytes: 26_530_550,
    license: "CC-BY-4.0",
    attribution: "WeSpeaker ResNet34-LM trained on VoxCeleb, CC BY 4.0",
};

pub const MODELS: [ModelSpec; 2] = [SEGMENTATION, EMBEDDING];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadProgress {
    pub file_name: &'static str,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

/// Total download size of all models.
pub fn total_bytes() -> u64 {
    MODELS.iter().map(|m| m.size_bytes).sum()
}

/// Models absent from `models_dir` or whose size does not match (hashes are checked at download).
pub fn missing(models_dir: &Path) -> Vec<&'static ModelSpec> {
    let _ = models_dir;
    todo!("track A")
}

/// Download every missing model into `models_dir`: stream to `<file>.part`, verify sha256,
/// then rename. A mismatch deletes the `.part` file and returns `ModelCorrupt`.
pub async fn download_all(
    models_dir: &Path,
    cancel: &AtomicBool,
    progress: &(dyn Fn(DownloadProgress) + Send + Sync),
) -> Result<()> {
    let _ = (models_dir, cancel, progress);
    todo!("track A")
}
