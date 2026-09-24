//! Model artifacts: commit-pinned URLs, sizes and sha256, plus verified download.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::{DiarizationError, Result};

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

/// A download that receives no data for this long fails instead of hanging (a dropped Wi-Fi
/// or VPN connection often stalls without a reset).
const STALL_TIMEOUT: Duration = Duration::from_secs(30);
/// How often a download waiting on the network looks at the cancel flag.
const CANCEL_POLL: Duration = Duration::from_millis(250);
/// Upper bound on a whole download, matching the Parakeet model download.
const TOTAL_TIMEOUT: Duration = Duration::from_secs(3600);

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
    MODELS
        .iter()
        .filter(|m| {
            std::fs::metadata(models_dir.join(m.file_name))
                .map(|md| !md.is_file() || md.len() != m.size_bytes)
                .unwrap_or(true)
        })
        .collect()
}

/// Download every missing model into `models_dir`: stream to `<file>.part`, verify sha256,
/// then rename. A mismatch deletes the `.part` file and returns `ModelCorrupt`.
pub async fn download_all(
    models_dir: &Path,
    cancel: &AtomicBool,
    progress: &(dyn Fn(DownloadProgress) + Send + Sync),
) -> Result<()> {
    download_specs(models_dir, &missing(models_dir), cancel, progress, STALL_TIMEOUT).await
}

async fn download_specs(
    models_dir: &Path,
    pending: &[&'static ModelSpec],
    cancel: &AtomicBool,
    progress: &(dyn Fn(DownloadProgress) + Send + Sync),
    stall_timeout: Duration,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    tokio::fs::create_dir_all(models_dir).await?;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(TOTAL_TIMEOUT)
        .build()
        .map_err(|e| DiarizationError::Download(e.to_string()))?;
    for &spec in pending {
        let part = part_path(models_dir, spec);
        let result = download_one(&client, spec, &part, cancel, progress, stall_timeout).await;
        let digest = match result {
            Ok(digest) => digest,
            Err(e) => {
                let _ = tokio::fs::remove_file(&part).await;
                return Err(e);
            }
        };
        install_verified(&part, &models_dir.join(spec.file_name), &digest, spec)?;
        log::info!("Diarization model {} downloaded and verified", spec.file_name);
    }
    Ok(())
}

fn part_path(models_dir: &Path, spec: &ModelSpec) -> PathBuf {
    models_dir.join(format!("{}.part", spec.file_name))
}

/// Stream `spec.url` into `part`, returning the lowercase hex sha256 of what was written.
/// Fails with `Download` when no data arrives for `stall_timeout`, and observes `cancel`
/// while waiting for data as well as between chunks.
async fn download_one(
    client: &reqwest::Client,
    spec: &'static ModelSpec,
    part: &Path,
    cancel: &AtomicBool,
    progress: &(dyn Fn(DownloadProgress) + Send + Sync),
    stall_timeout: Duration,
) -> Result<String> {
    let response = client
        .get(spec.url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| DiarizationError::Download(format!("{}: {e}", spec.file_name)))?;
    let mut file = tokio::fs::File::create(part).await?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();
    progress(DownloadProgress { file_name: spec.file_name, downloaded_bytes: 0, total_bytes: spec.size_bytes });
    let mut last_data = tokio::time::Instant::now();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(DiarizationError::Cancelled);
        }
        // `Next` holds no data between polls, so dropping it on a poll timeout loses nothing.
        let next = match tokio::time::timeout(CANCEL_POLL, stream.next()).await {
            Ok(next) => next,
            Err(_) if last_data.elapsed() >= stall_timeout => {
                return Err(DiarizationError::Download(format!(
                    "{}: no data received for {} seconds",
                    spec.file_name,
                    stall_timeout.as_secs()
                )));
            }
            Err(_) => continue,
        };
        let Some(chunk) = next else { break };
        last_data = tokio::time::Instant::now();
        let chunk = chunk.map_err(|e| DiarizationError::Download(format!("{}: {e}", spec.file_name)))?;
        downloaded += chunk.len() as u64;
        if downloaded > spec.size_bytes {
            return Err(DiarizationError::ModelCorrupt(format!(
                "{} is larger than the expected {} bytes",
                spec.file_name, spec.size_bytes
            )));
        }
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        progress(DownloadProgress {
            file_name: spec.file_name,
            downloaded_bytes: downloaded,
            total_bytes: spec.size_bytes,
        });
    }
    file.flush().await?;
    file.sync_all().await?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Move a finished `.part` file into place if its sha256 matches `spec`; otherwise delete it and
/// return `ModelCorrupt`.
fn install_verified(part: &Path, dest: &Path, actual_sha256: &str, spec: &ModelSpec) -> Result<()> {
    if !actual_sha256.eq_ignore_ascii_case(spec.sha256) {
        let _ = std::fs::remove_file(part);
        return Err(DiarizationError::ModelCorrupt(format!(
            "{}: sha256 {actual_sha256}, expected {}",
            spec.file_name, spec.sha256
        )));
    }
    std::fs::rename(part, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: ModelSpec = ModelSpec {
        file_name: "tiny.onnx",
        url: "http://invalid.invalid/tiny.onnx",
        // sha256("hello")
        sha256: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        size_bytes: 5,
        license: "MIT",
        attribution: "test",
    };

    #[test]
    fn missing_checks_presence_and_size() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(missing(dir.path()).len(), 2);
        std::fs::write(dir.path().join(SEGMENTATION.file_name), b"short").unwrap();
        let names: Vec<&str> = missing(dir.path()).iter().map(|m| m.file_name).collect();
        assert_eq!(names, vec![SEGMENTATION.file_name, EMBEDDING.file_name], "wrong size counts as missing");
        std::fs::File::create(dir.path().join(SEGMENTATION.file_name))
            .unwrap()
            .set_len(SEGMENTATION.size_bytes)
            .unwrap();
        let names: Vec<&str> = missing(dir.path()).iter().map(|m| m.file_name).collect();
        assert_eq!(names, vec![EMBEDDING.file_name]);
    }

    #[test]
    fn verified_part_is_renamed_into_place() {
        let dir = tempfile::tempdir().unwrap();
        let part = part_path(dir.path(), &TINY);
        std::fs::write(&part, b"hello").unwrap();
        let mut hasher = Sha256::new();
        hasher.update(b"hello");
        let digest = format!("{:x}", hasher.finalize());
        assert_eq!(digest, TINY.sha256);
        install_verified(&part, &dir.path().join(TINY.file_name), &digest, &TINY).unwrap();
        assert!(!part.exists());
        assert_eq!(std::fs::read(dir.path().join(TINY.file_name)).unwrap(), b"hello");
    }

    #[test]
    fn sha_mismatch_deletes_part_and_reports_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let part = part_path(dir.path(), &TINY);
        std::fs::write(&part, b"hellO").unwrap();
        let err = install_verified(&part, &dir.path().join(TINY.file_name), &"0".repeat(64), &TINY).unwrap_err();
        assert!(matches!(err, DiarizationError::ModelCorrupt(_)), "{err:?}");
        assert!(!part.exists());
        assert!(!dir.path().join(TINY.file_name).exists());
    }

    #[test]
    fn total_matches_design_download_size() {
        assert_eq!(total_bytes(), 5_992_913 + 26_530_550);
    }

    /// Serves one GET with the headers and the first 64 KiB of a 1 MiB body, then keeps the
    /// connection open without sending more, like a connection that dropped mid-download.
    async fn stalling_server() -> (&'static ModelSpec, tokio::task::JoinHandle<()>) {
        use tokio::io::AsyncReadExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/stalled.onnx", listener.local_addr().unwrap());
        let spec: &'static ModelSpec = Box::leak(Box::new(ModelSpec {
            file_name: "stalled.onnx",
            url: Box::leak(url.into_boxed_str()),
            sha256: TINY.sha256,
            size_bytes: 1 << 20,
            license: "MIT",
            attribution: "test",
        }));
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await.unwrap();
            let headers = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", 1 << 20);
            socket.write_all(headers.as_bytes()).await.unwrap();
            socket.write_all(&vec![0u8; 64 * 1024]).await.unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(Duration::from_secs(3600)).await;
            drop(socket);
        });
        (spec, server)
    }

    #[tokio::test]
    async fn cancel_interrupts_a_stalled_download_and_removes_the_part_file() {
        let dir = tempfile::tempdir().unwrap();
        let (spec, server) = stalling_server().await;
        let cancel = AtomicBool::new(false);
        let pending = [spec];
        let started = tokio::time::Instant::now();

        let download = download_specs(dir.path(), &pending, &cancel, &|_| {}, Duration::from_secs(60));
        let set_cancel = async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            cancel.store(true, Ordering::SeqCst);
        };
        let (result, ()) = tokio::join!(download, set_cancel);

        assert!(matches!(result, Err(DiarizationError::Cancelled)), "{result:?}");
        assert!(started.elapsed() < Duration::from_secs(5), "cancel took {:?}", started.elapsed());
        assert!(!part_path(dir.path(), spec).exists());
        server.abort();
    }

    #[tokio::test]
    async fn a_stalled_download_times_out_and_removes_the_part_file() {
        let dir = tempfile::tempdir().unwrap();
        let (spec, server) = stalling_server().await;
        let received = std::sync::Mutex::new(0u64);
        let on_progress = |p: DownloadProgress| *received.lock().unwrap() = p.downloaded_bytes;

        let result =
            download_specs(dir.path(), &[spec], &AtomicBool::new(false), &on_progress, Duration::from_secs(1)).await;

        match result {
            Err(DiarizationError::Download(message)) => assert!(message.contains("no data received"), "{message}"),
            other => panic!("expected a stall error, got {other:?}"),
        }
        assert!(*received.lock().unwrap() > 0, "the server's first bytes should arrive before the stall");
        assert!(!part_path(dir.path(), spec).exists());
        server.abort();
    }
}
