// Model manager for built-in AI models - handles downloads and lifecycle
// Follows the same pattern as whisper_engine/whisper_engine.rs for consistency

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::fs::{self, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::RwLock;
use tokio::time::timeout;

use super::models::{get_available_models, get_model_by_name, ModelDef};

/// Progress callbacks are only `Send`; the mutex lets a download borrow one across awaits.
type SharedProgress = std::sync::Mutex<Box<dyn Fn(DownloadProgress) + Send>>;

fn report(progress: Option<&SharedProgress>, update: DownloadProgress) {
    if let Some(progress) = progress {
        let callback = progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        callback(update);
    }
}

/// A download that receives nothing for this long fails instead of hanging.
const STALL_TIMEOUT: Duration = Duration::from_secs(30);
/// How often a download waiting on the network checks for cancellation.
const CANCEL_POLL: Duration = Duration::from_millis(250);
/// An installed file counts as complete between these shares of the registry size. Downloads
/// only reach the install path after matching the server's byte count, so the window screens
/// files that older versions downloaded in place, and tolerates same-quantization re-uploads.
const INSTALLED_MIN_RATIO: f64 = 0.98;
const INSTALLED_MAX_RATIO: f64 = 1.1;

// ============================================================================
// Model Status Types
// ============================================================================

/// Detailed download progress info (MB-based with speed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Bytes downloaded so far
    pub downloaded_bytes: u64,
    /// Total file size in bytes
    pub total_bytes: u64,
    /// Downloaded in MB (for display)
    pub downloaded_mb: f64,
    /// Total size in MB (for display)
    pub total_mb: f64,
    /// Download speed in MB/s
    pub speed_mbps: f64,
    /// Percentage complete (0-100)
    pub percent: u8,
}

impl DownloadProgress {
    pub fn new(downloaded: u64, total: u64, speed_mbps: f64) -> Self {
        let percent = if total > 0 {
            ((downloaded as f64 / total as f64) * 100.0) as u8
        } else {
            0
        };
        Self {
            downloaded_bytes: downloaded,
            total_bytes: total,
            downloaded_mb: downloaded as f64 / (1024.0 * 1024.0),
            total_mb: total as f64 / (1024.0 * 1024.0),
            speed_mbps,
            percent,
        }
    }
}

/// Model status in the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelStatus {
    /// Model is not yet downloaded
    NotDownloaded,

    /// Model is currently being downloaded (progress 0-100)
    Downloading { progress: u8 },

    /// Model is downloaded and ready to use
    Available,

    /// Model file is corrupted and needs redownload
    Corrupted { file_size: u64, expected_min_size: u64 },

    /// Error occurred with the model
    Error(String),
}

/// Model information for UI display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name (e.g., "gemma3:1b")
    pub name: String,

    /// Display name for UI
    pub display_name: String,

    /// Current status
    pub status: ModelStatus,

    /// File path (if available)
    pub path: PathBuf,

    /// Size in MB
    pub size_mb: u64,

    /// Context window size in tokens
    pub context_size: u32,

    /// Description
    pub description: String,

    /// GGUF filename on disk
    pub gguf_file: String,
}

// ============================================================================
// Model Manager
// ============================================================================

pub struct ModelManager {
    /// Directory where models are stored
    models_dir: PathBuf,

    /// Currently available models with their status
    available_models: Arc<RwLock<HashMap<String, ModelInfo>>>,

    /// Active downloads (model names)
    active_downloads: Arc<RwLock<HashSet<String>>>,

    /// Cancellation flag for current download
    cancel_download_flag: Arc<RwLock<Option<String>>>,
}

impl ModelManager {
    /// Create a new model manager with default models directory
    pub fn new() -> Result<Self> {
        Self::new_with_models_dir(None)
    }

    /// Create a new model manager with custom models directory
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = if let Some(dir) = models_dir {
            dir
        } else {
            // Fallback: Use current directory in development
            let current_dir = std::env::current_dir()
                .map_err(|e| anyhow!("Failed to get current directory: {}", e))?;

            if cfg!(debug_assertions) {
                // Development mode
                current_dir.join("models").join("summary")
            } else {
                // Production mode fallback (caller should provide path)
                log::warn!("ModelManager: No models directory provided, using fallback path");
                dirs::data_dir()
                    .or_else(|| dirs::home_dir())
                    .ok_or_else(|| anyhow!("Could not find system data directory"))?
                    .join("Meetily")
                    .join("models")
                    .join("summary")
            }
        };

        log::info!(
            "Built-in AI ModelManager using directory: {}",
            models_dir.display()
        );

        Ok(Self {
            models_dir,
            available_models: Arc::new(RwLock::new(HashMap::new())),
            active_downloads: Arc::new(RwLock::new(HashSet::new())),
            cancel_download_flag: Arc::new(RwLock::new(None)),
        })
    }

    /// Initialize and scan for existing models
    pub async fn init(&self) -> Result<()> {
        // Create models directory if it doesn't exist
        if !self.models_dir.exists() {
            fs::create_dir_all(&self.models_dir).await?;
            log::info!("Created models directory: {}", self.models_dir.display());
        }

        // Scan for existing models
        self.scan_models().await?;

        Ok(())
    }

    /// Scan models directory and update status
    pub async fn scan_models(&self) -> Result<()> {
        let start = std::time::Instant::now();

        log::info!(
            "Starting model scan in directory: {}",
            self.models_dir.display()
        );

        let model_defs = get_available_models();
        let mut models_map = HashMap::new();

        for model_def in model_defs {
            let model_path = self.models_dir.join(&model_def.gguf_file);
            log::debug!(
                "Checking model '{}' at path: {}",
                model_def.name,
                model_path.display()
            );

            let is_actively_downloading = {
                let active = self.active_downloads.read().await;
                active.contains(&model_def.name)
            };

            // If actively downloading, preserve existing status from memory
            if is_actively_downloading {
                let existing_info = {
                    let models = self.available_models.read().await;
                    models.get(&model_def.name).cloned()
                };

                if let Some(info) = existing_info {
                    // Preserve existing status (should be Downloading)
                    models_map.insert(model_def.name.clone(), info);
                    log::debug!(
                        "Model '{}': Preserving Downloading status during scan",
                        model_def.name
                    );
                    continue;
                }
            }

            // In-progress downloads live in `<file>.part`, so they never reach this path.
            let status = if model_path.exists() {
                // Check if file size matches expected size (basic validation)
                match fs::metadata(&model_path).await {
                    Ok(metadata) => {
                        let file_size_mb = metadata.len() / (1024 * 1024);
                        let expected_min = (model_def.size_mb as f64 * INSTALLED_MIN_RATIO) as u64;

                        log::info!(
                            "Model '{}': found {} MB (expected {} MB)",
                            model_def.name,
                            file_size_mb,
                            model_def.size_mb
                        );

                        if check_installed_size(metadata.len(), model_def.size_mb)
                            == InstalledSize::Complete
                        {
                            log::info!("Model '{}': AVAILABLE", model_def.name);
                            ModelStatus::Available
                        } else {
                            log::warn!(
                                "Model '{}': CORRUPTED (size mismatch: {} MB, expected {} MB)",
                                model_def.name,
                                file_size_mb,
                                model_def.size_mb
                            );
                            ModelStatus::Corrupted {
                                file_size: file_size_mb,
                                expected_min_size: expected_min,
                            }
                        }
                    }
                    Err(e) => {
                        log::error!(
                            "Model '{}': Failed to read metadata: {}",
                            model_def.name,
                            e
                        );
                        ModelStatus::Error(format!("Failed to read metadata: {}", e))
                    }
                }
            } else {
                log::debug!("Model '{}': NOT FOUND", model_def.name);
                ModelStatus::NotDownloaded
            };

            let model_info = ModelInfo {
                name: model_def.name.clone(),
                display_name: model_def.display_name.clone(),
                status,
                path: model_path,
                size_mb: model_def.size_mb,
                context_size: model_def.context_size,
                description: model_def.description.clone(),
                gguf_file: model_def.gguf_file.clone(),
            };

            models_map.insert(model_def.name.clone(), model_info);
        }

        let model_count = models_map.len();

        let mut models = self.available_models.write().await;
        *models = models_map;

        let elapsed = start.elapsed();
        log::info!(
            "Model scan complete: {} models checked in {:?}",
            model_count,
            elapsed
        );
        Ok(())
    }

    /// Get list of all models with their status
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.available_models
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    /// Get info for a specific model
    pub async fn get_model_info(&self, model_name: &str) -> Option<ModelInfo> {
        self.available_models
            .read()
            .await
            .get(model_name)
            .cloned()
    }

    /// Check if a model is ready to use
    /// If refresh=true, scans filesystem before checking (slower but accurate)
    pub async fn is_model_ready(&self, model_name: &str, refresh: bool) -> bool {
        if refresh {
            if let Err(e) = self.scan_models().await {
                log::error!("Failed to scan models: {}", e);
                return false;
            }
        }

        if let Some(info) = self.get_model_info(model_name).await {
            info.status == ModelStatus::Available
        } else {
            false
        }
    }

    /// Download a model with simple percentage callback (backward compatible)
    pub async fn download_model(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(u8) + Send>>,
    ) -> Result<()> {
        // Wrap the simple callback to use detailed progress internally
        let detailed_callback: Option<Box<dyn Fn(DownloadProgress) + Send>> =
            progress_callback.map(|cb| {
                Box::new(move |p: DownloadProgress| cb(p.percent)) as Box<dyn Fn(DownloadProgress) + Send>
            });
        self.download_model_detailed(model_name, detailed_callback).await
    }

    /// Download a model with detailed progress (MB, speed, etc.)
    ///
    /// Bytes stream into `<file>.part`, which is renamed into place only once its length matches
    /// the server's and it passes the GGUF magic check, so an interrupted download is never
    /// mistaken for an installed model. A later attempt resumes the `.part` file.
    pub async fn download_model_detailed(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<()> {
        let model_def = get_model_by_name(model_name)
            .ok_or_else(|| anyhow!("Unknown model: {}", model_name))?;
        let progress = progress_callback.map(std::sync::Mutex::new);
        self.download_model_def(&model_def, progress.as_ref())
            .await
    }

    async fn download_model_def(
        &self,
        model_def: &ModelDef,
        progress_callback: Option<&SharedProgress>,
    ) -> Result<()> {
        let model_name = model_def.name.as_str();
        log::info!("Starting download for model: {}", model_name);

        if !self
            .active_downloads
            .write()
            .await
            .insert(model_name.to_string())
        {
            log::warn!("Download already in progress for model: {}", model_name);
            return Err(anyhow!("Download already in progress"));
        }

        // Clear cancellation flag
        {
            let mut cancel_flag = self.cancel_download_flag.write().await;
            *cancel_flag = None;
        }

        self.set_status(model_name, ModelStatus::Downloading { progress: 0 })
            .await;

        let file_path = self.models_dir.join(&model_def.gguf_file);
        let part = part_path(&file_path);

        if let Ok(metadata) = fs::metadata(&file_path).await {
            match check_installed_size(metadata.len(), model_def.size_mb) {
                InstalledSize::Complete => {
                    log::info!(
                        "Model '{}' already exists and is valid ({} bytes), skipping download",
                        model_name,
                        metadata.len()
                    );
                    self.set_status(model_name, ModelStatus::Available).await;
                    self.active_downloads.write().await.remove(model_name);
                    let total = metadata.len();
                    report(progress_callback, DownloadProgress::new(total, total, 0.0));
                    return Ok(());
                }
                InstalledSize::TooLarge => {
                    log::warn!(
                        "Model '{}' exists but is too large ({} bytes), deleting and re-downloading",
                        model_name,
                        metadata.len()
                    );
                    if let Err(e) = fs::remove_file(&file_path).await {
                        log::warn!("Failed to delete oversized model file: {}", e);
                    }
                }
                InstalledSize::TooSmall => {
                    // Versions before `.part` downloads wrote partial files in place: resume it.
                    log::info!(
                        "Model '{}' exists but is incomplete ({} bytes), resuming it as a partial download",
                        model_name,
                        metadata.len()
                    );
                    let moved = if fs::metadata(&part).await.is_err() {
                        fs::rename(&file_path, &part).await
                    } else {
                        fs::remove_file(&file_path).await
                    };
                    if let Err(e) = moved {
                        log::warn!("Failed to move incomplete model file aside: {}", e);
                    }
                }
            }
        }

        let expected_total = match self.fetch_to_part(model_def, &part, progress_callback).await {
            Ok(expected_total) => expected_total,
            Err(DownloadStop::Cancelled) => {
                log::info!("Download cancelled for model: {}", model_name);
                return self
                    .end_download(
                        model_name,
                        ModelStatus::NotDownloaded,
                        // Special marker prefix distinguishes cancellation from other errors
                        "CANCELLED: Download cancelled by user".to_string(),
                    )
                    .await;
            }
            Err(DownloadStop::Failed(message)) => {
                log::error!("Download failed for {}: {}", model_name, message);
                return self
                    .end_download(model_name, ModelStatus::Error(message.clone()), message)
                    .await;
            }
        };

        log::info!("Download completed for model: {}", model_name);
        self.set_status(model_name, ModelStatus::Downloading { progress: 100 })
            .await;
        let total = expected_total.unwrap_or_default();
        report(progress_callback, DownloadProgress::new(total, total, 0.0));

        // Small delay to ensure UI receives 100% event
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        if let Err(e) = finalize_download(&part, &file_path, expected_total, model_def.size_mb).await {
            log::error!("Downloaded file for {} was not installed: {}", model_name, e);
            let message = format!("Download could not be installed: {}", e);
            return self
                .end_download(model_name, ModelStatus::Error(message.clone()), message)
                .await;
        }

        {
            let mut models = self.available_models.write().await;
            if let Some(model_info) = models.get_mut(model_name) {
                model_info.status = ModelStatus::Available;
                model_info.path = file_path.clone();
            }
        }
        self.active_downloads.write().await.remove(model_name);

        Ok(())
    }

    /// Streams the model into `part`, resuming from its current length when the server allows.
    /// Returns the size the finished file must have, when the server reported one.
    async fn fetch_to_part(
        &self,
        model_def: &ModelDef,
        part: &Path,
        progress_callback: Option<&SharedProgress>,
    ) -> std::result::Result<Option<u64>, DownloadStop> {
        let model_name = model_def.name.as_str();
        log::info!("Downloading from: {}", model_def.download_url);
        log::info!("Saving to: {}", part.display());

        fs::create_dir_all(&self.models_dir)
            .await
            .map_err(|e| DownloadStop::Failed(format!("Failed to create models directory: {}", e)))?;

        let client = Client::builder()
            .tcp_nodelay(true) // Disable Nagle's algorithm for faster streaming
            .pool_max_idle_per_host(1) // Keep connection alive
            .timeout(Duration::from_secs(3600)) // 1 hour timeout for large files
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| DownloadStop::Failed(format!("Failed to create HTTP client: {}", e)))?;

        let mut existing_size = fs::metadata(part).await.map(|m| m.len()).unwrap_or(0);
        let mut response = self
            .send_download_request(&client, model_def, existing_size)
            .await?;
        if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            // The partial file is not a prefix the server can continue (e.g. the file changed).
            log::warn!(
                "Server cannot resume '{}' at byte {}, restarting the download",
                model_name,
                existing_size
            );
            existing_size = 0;
            response = self.send_download_request(&client, model_def, 0).await?;
        }

        let status = response.status();
        let (resuming, expected_total) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            let content_range = response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|value| value.to_str().ok());
            match resumed_total(content_range, response.content_length(), existing_size) {
                Ok(total) => (true, total),
                Err(message) => {
                    let _ = fs::remove_file(part).await;
                    return Err(DownloadStop::Failed(message));
                }
            }
        } else if status.is_success() {
            if existing_size > 0 {
                log::warn!("Server doesn't support resume, starting fresh download");
            }
            (false, response.content_length())
        } else {
            return Err(DownloadStop::Failed(format!("Download failed with status: {}", status)));
        };

        let total_size = expected_total.unwrap_or(0);
        log::info!("Total size: {} MB", total_size / (1024 * 1024));

        // Open file for append if resuming, or create new
        let file = if resuming {
            OpenOptions::new().append(true).open(part).await
        } else {
            fs::File::create(part).await
        }
        .map_err(|e| DownloadStop::Failed(format!("Failed to open download file: {}", e)))?;

        // Use 8MB buffer to reduce disk I/O syscalls (major performance improvement)
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

        let mut downloaded: u64 = if resuming { existing_size } else { 0 };

        // Emit initial progress (showing resumed position if applicable)
        report(progress_callback, DownloadProgress::new(downloaded, total_size, 0.0));
        log::info!(
            "Starting at {:.1} MB / {:.1} MB",
            downloaded as f64 / (1024.0 * 1024.0),
            total_size as f64 / (1024.0 * 1024.0)
        );

        let mut last_progress_percent = if total_size > 0 {
            ((downloaded as f64 / total_size as f64) * 100.0) as u8
        } else {
            0
        };
        let mut last_report_time = std::time::Instant::now();
        let mut bytes_since_last_report: u64 = 0;
        let download_start_time = std::time::Instant::now();
        let start_downloaded = downloaded;

        use futures_util::StreamExt;
        let mut stream = response.bytes_stream();

        loop {
            let chunk = match self.wait_for_network(model_name, stream.next()).await {
                NetworkWait::Ready(None) => break,
                NetworkWait::Ready(Some(Ok(chunk))) => chunk,
                NetworkWait::Ready(Some(Err(e))) => {
                    let _ = writer.flush().await;
                    // Categorize error for user-friendly message
                    let error_msg = if e.is_timeout() {
                        "Connection timeout - Check your internet"
                    } else if e.is_connect() {
                        "Connection failed - Check your internet"
                    } else if e.is_body() {
                        "Stream interrupted - Network unstable"
                    } else {
                        "Download error"
                    };
                    return Err(DownloadStop::Failed(format!("{}: {}", error_msg, e)));
                }
                NetworkWait::Cancelled => {
                    // Keep the partial file for resume on the next attempt
                    let _ = writer.flush().await;
                    return Err(DownloadStop::Cancelled);
                }
                NetworkWait::Stalled => {
                    let _ = writer.flush().await;
                    return Err(DownloadStop::Failed(format!(
                        "Download timeout - No data received for {} seconds",
                        STALL_TIMEOUT.as_secs()
                    )));
                }
            };
            let chunk_len = chunk.len() as u64;
            writer
                .write_all(&chunk)
                .await
                .map_err(|e| DownloadStop::Failed(format!("Error writing to file: {}", e)))?;

            downloaded += chunk_len;
            bytes_since_last_report += chunk_len;

            // Calculate progress
            let progress_percent = if total_size > 0 {
                let exact_percent = (downloaded as f64 / total_size as f64) * 100.0;
                exact_percent.min(100.0) as u8
            } else {
                0
            };

            let elapsed_since_report = last_report_time.elapsed();
            let is_download_complete = downloaded >= total_size;
            let should_report = progress_percent > last_progress_percent
                || is_download_complete  // Force report on completion
                || elapsed_since_report.as_millis() >= 500;

            if should_report {
                // Calculate speed based on bytes downloaded since last report
                let speed_mbps = if elapsed_since_report.as_secs_f64() > 0.0 {
                    (bytes_since_last_report as f64 / (1024.0 * 1024.0)) / elapsed_since_report.as_secs_f64()
                } else {
                    // Fallback to overall average speed
                    let total_elapsed = download_start_time.elapsed().as_secs_f64();
                    if total_elapsed > 0.0 {
                        ((downloaded - start_downloaded) as f64 / (1024.0 * 1024.0)) / total_elapsed
                    } else {
                        0.0
                    }
                };

                log::info!(
                    "Download: {:.1} MB / {:.1} MB ({:.1} MB/s)",
                    downloaded as f64 / (1024.0 * 1024.0),
                    total_size as f64 / (1024.0 * 1024.0),
                    speed_mbps
                );

                self.set_status(
                    model_name,
                    ModelStatus::Downloading {
                        progress: if is_download_complete { 100 } else { progress_percent },
                    },
                )
                .await;

                // Call progress callback with detailed info
                report(progress_callback, DownloadProgress::new(downloaded, total_size, speed_mbps));

                last_progress_percent = progress_percent;
                last_report_time = std::time::Instant::now();
                bytes_since_last_report = 0;
            }
        }

        let flushed = match writer.flush().await {
            Ok(()) => writer.into_inner().sync_all().await,
            Err(e) => Err(e),
        };
        flushed.map_err(|e| DownloadStop::Failed(format!("Error writing to file: {}", e)))?;
        Ok(expected_total)
    }

    /// Sends the download request, with a `Range` header when resuming from `resume_from`.
    async fn send_download_request(
        &self,
        client: &Client,
        model_def: &ModelDef,
        resume_from: u64,
    ) -> std::result::Result<reqwest::Response, DownloadStop> {
        let mut request = client.get(&model_def.download_url);
        if resume_from > 0 {
            log::info!(
                "Resuming download from byte {} ({:.1} MB)",
                resume_from,
                resume_from as f64 / (1024.0 * 1024.0)
            );
            request = request.header(reqwest::header::RANGE, format!("bytes={}-", resume_from));
        }
        match self.wait_for_network(&model_def.name, request.send()).await {
            NetworkWait::Ready(Ok(response)) => Ok(response),
            NetworkWait::Ready(Err(e)) => {
                Err(DownloadStop::Failed(format!("Failed to start download: {}", e)))
            }
            NetworkWait::Cancelled => Err(DownloadStop::Cancelled),
            NetworkWait::Stalled => Err(DownloadStop::Failed(format!(
                "Download timeout - No response for {} seconds",
                STALL_TIMEOUT.as_secs()
            ))),
        }
    }

    /// Awaits one network step, checking for cancellation every [`CANCEL_POLL`] and giving up
    /// after [`STALL_TIMEOUT`]. reqwest's only read bound is the one-hour total timeout, which a
    /// connection that dropped without a reset would otherwise wait out.
    async fn wait_for_network<F: Future>(&self, model_name: &str, step: F) -> NetworkWait<F::Output> {
        let mut step = std::pin::pin!(step);
        let started = tokio::time::Instant::now();
        loop {
            if self.cancel_download_flag.read().await.as_deref() == Some(model_name) {
                return NetworkWait::Cancelled;
            }
            match timeout(CANCEL_POLL, &mut step).await {
                Ok(output) => return NetworkWait::Ready(output),
                Err(_) if started.elapsed() >= STALL_TIMEOUT => return NetworkWait::Stalled,
                Err(_) => {}
            }
        }
    }

    async fn set_status(&self, model_name: &str, status: ModelStatus) {
        if let Some(model_info) = self.available_models.write().await.get_mut(model_name) {
            model_info.status = status;
        }
    }

    /// Ends a download attempt: drops it from the active set, records `status` and returns
    /// `message` as the error.
    async fn end_download(&self, model_name: &str, status: ModelStatus, message: String) -> Result<()> {
        self.active_downloads.write().await.remove(model_name);
        self.set_status(model_name, status).await;
        Err(anyhow!(message))
    }

    /// Cancel an ongoing download
    pub async fn cancel_download(&self, model_name: &str) -> Result<()> {
        log::info!("Cancelling download for model: {}", model_name);

        // Set cancellation flag - download loop will detect this and handle cleanup
        {
            let mut cancel_flag = self.cancel_download_flag.write().await;
            *cancel_flag = Some(model_name.to_string());
        }

        // Note: active_downloads cleanup is handled by the download loop when it detects
        // the cancellation flag. This avoids double-removal race condition.

        // Update status immediately for UI responsiveness
        {
            let mut models = self.available_models.write().await;
            if let Some(model_info) = models.get_mut(model_name) {
                model_info.status = ModelStatus::NotDownloaded;
            }
        }

        // Brief delay to let download loop detect cancellation
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(())
    }

    /// Delete a corrupted or available model file
    pub async fn delete_model(&self, model_name: &str) -> Result<()> {
        log::info!("Deleting model: {}", model_name);

        let model_def = get_model_by_name(model_name)
            .ok_or_else(|| anyhow!("Unknown model: {}", model_name))?;

        let file_path = self.models_dir.join(&model_def.gguf_file);

        for path in [part_path(&file_path), file_path] {
            if path.exists() {
                fs::remove_file(&path).await?;
                log::info!("Deleted model file: {}", path.display());
            }
        }

        // Update status
        {
            let mut models = self.available_models.write().await;
            if let Some(model_info) = models.get_mut(model_name) {
                model_info.status = ModelStatus::NotDownloaded;
            }
        }

        Ok(())
    }

    /// Get models directory path
    pub fn get_models_directory(&self) -> PathBuf {
        self.models_dir.clone()
    }
}

/// Why a download attempt stopped before the file was complete.
enum DownloadStop {
    Cancelled,
    Failed(String),
}

enum NetworkWait<T> {
    Ready(T),
    Cancelled,
    Stalled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstalledSize {
    Complete,
    TooSmall,
    TooLarge,
}

fn check_installed_size(len_bytes: u64, expected_mb: u64) -> InstalledSize {
    let len_mb = len_bytes as f64 / (1024.0 * 1024.0);
    if len_mb < expected_mb as f64 * INSTALLED_MIN_RATIO {
        InstalledSize::TooSmall
    } else if len_mb > expected_mb as f64 * INSTALLED_MAX_RATIO {
        InstalledSize::TooLarge
    } else {
        InstalledSize::Complete
    }
}

/// Where a download of `file_path` is written until it is complete.
fn part_path(file_path: &Path) -> PathBuf {
    let mut name = file_path.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

/// Size of the whole file from a `206` response to a request resuming at `resume_from`: the
/// `Content-Range` total, or the offset plus the body length when the header is absent.
/// `Err` when the server resumed at another offset, which would corrupt the file.
fn resumed_total(
    content_range: Option<&str>,
    content_length: Option<u64>,
    resume_from: u64,
) -> std::result::Result<Option<u64>, String> {
    let Some(content_range) = content_range else {
        return Ok(content_length.map(|length| resume_from + length));
    };
    let (range, total) = content_range
        .trim()
        .strip_prefix("bytes ")
        .and_then(|rest| rest.split_once('/'))
        .ok_or_else(|| format!("Unexpected Content-Range from server: {}", content_range))?;
    let start = range
        .split_once('-')
        .and_then(|(start, _)| start.parse::<u64>().ok())
        .ok_or_else(|| format!("Unexpected Content-Range from server: {}", content_range))?;
    if start != resume_from {
        return Err(format!(
            "Server resumed the download at byte {} instead of {}",
            start, resume_from
        ));
    }
    Ok(total.parse::<u64>().ok())
}

/// Installs a finished `.part` download at `dest`. The file must be exactly `expected_bytes`
/// long when the server reported a size (else within the registry size window) and start with
/// a GGUF/GGML magic number. A short file is kept so the next attempt resumes it; an oversized
/// or invalid one is deleted.
async fn finalize_download(
    part: &Path,
    dest: &Path,
    expected_bytes: Option<u64>,
    registry_size_mb: u64,
) -> Result<()> {
    let len = fs::metadata(part).await?.len();
    let size_check = match expected_bytes {
        Some(expected) if len < expected => InstalledSize::TooSmall,
        Some(expected) if len > expected => InstalledSize::TooLarge,
        Some(_) => InstalledSize::Complete,
        None => check_installed_size(len, registry_size_mb),
    };
    match size_check {
        InstalledSize::Complete => {}
        InstalledSize::TooSmall => {
            return Err(anyhow!(
                "download incomplete ({} of {} bytes); retry to resume",
                len,
                expected_bytes.map_or_else(|| format!("~{} MiB", registry_size_mb), |b| b.to_string())
            ));
        }
        InstalledSize::TooLarge => {
            let _ = fs::remove_file(part).await;
            return Err(anyhow!("downloaded file is larger than expected ({} bytes)", len));
        }
    }
    if let Err(e) = validate_gguf_file(part).await {
        let _ = fs::remove_file(part).await;
        return Err(e);
    }
    fs::rename(part, dest).await?;
    Ok(())
}

/// Validate that a file is a valid GGUF model
async fn validate_gguf_file(path: &Path) -> Result<()> {
    let mut file = fs::File::open(path).await?;

    // Read first 4 bytes to check for GGUF magic number
    use tokio::io::AsyncReadExt;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).await?;

    // GGUF magic number is "GGUF" (0x47475546)
    if &magic == b"GGUF" {
        Ok(())
    } else if &magic == b"ggjt" || &magic == b"ggla" || &magic == b"ggml" {
        // Older formats (GGML, GGJT)
        Ok(())
    } else {
        Err(anyhow!(
            "Invalid model file: magic number {:?} doesn't match GGUF/GGML",
            magic
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    const MODEL_BYTES: &[u8] = b"GGUF\x03\x00\x00\x00 tiny model payload for download tests";

    fn tiny_model(url: String) -> ModelDef {
        ModelDef {
            name: "tiny:test".to_string(),
            gguf_file: "tiny.gguf".to_string(),
            download_url: url,
            size_mb: 0,
            ..get_available_models().remove(0)
        }
    }

    #[tokio::test]
    async fn complete_part_is_installed() {
        let dir = tempfile::tempdir().unwrap();
        let (part, dest) = (dir.path().join("m.gguf.part"), dir.path().join("m.gguf"));
        std::fs::write(&part, MODEL_BYTES).unwrap();

        finalize_download(&part, &dest, Some(MODEL_BYTES.len() as u64), 0).await.unwrap();

        assert!(!part.exists());
        assert_eq!(std::fs::read(&dest).unwrap(), MODEL_BYTES);
    }

    #[tokio::test]
    async fn short_part_is_kept_for_resume_and_not_installed() {
        let dir = tempfile::tempdir().unwrap();
        let (part, dest) = (dir.path().join("m.gguf.part"), dir.path().join("m.gguf"));
        std::fs::write(&part, &MODEL_BYTES[..10]).unwrap();

        let error = finalize_download(&part, &dest, Some(MODEL_BYTES.len() as u64), 0)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("incomplete"), "{error}");
        assert!(part.exists());
        assert!(!dest.exists());
    }

    #[tokio::test]
    async fn oversized_or_invalid_part_is_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let (part, dest) = (dir.path().join("m.gguf.part"), dir.path().join("m.gguf"));

        std::fs::write(&part, MODEL_BYTES).unwrap();
        assert!(finalize_download(&part, &dest, Some(10), 0).await.is_err());
        assert!(!part.exists() && !dest.exists());

        std::fs::write(&part, b"<html>not a model</html>").unwrap();
        assert!(finalize_download(&part, &dest, Some(24), 0).await.is_err());
        assert!(!part.exists() && !dest.exists());
    }

    #[tokio::test]
    async fn part_of_unknown_length_must_match_the_registry_size() {
        let dir = tempfile::tempdir().unwrap();
        let (part, dest) = (dir.path().join("m.gguf.part"), dir.path().join("m.gguf"));
        std::fs::write(&part, MODEL_BYTES).unwrap();
        std::fs::OpenOptions::new().write(true).open(&part).unwrap().set_len(1024 * 1024).unwrap();

        assert!(finalize_download(&part, &dest, None, 2).await.is_err());
        assert!(part.exists() && !dest.exists());
        finalize_download(&part, &dest, None, 1).await.unwrap();
        assert!(dest.exists());
    }

    #[test]
    fn resumed_total_checks_the_resume_offset() {
        assert_eq!(resumed_total(Some("bytes 100-999/1000"), Some(900), 100), Ok(Some(1000)));
        assert_eq!(resumed_total(Some("bytes 100-999/*"), Some(900), 100), Ok(None));
        assert_eq!(resumed_total(None, Some(900), 100), Ok(Some(1000)));
        assert_eq!(resumed_total(None, None, 100), Ok(None));
        assert!(resumed_total(Some("bytes 0-999/1000"), Some(1000), 100).is_err());
        assert!(resumed_total(Some("garbage"), Some(1000), 100).is_err());
    }

    #[tokio::test]
    async fn scan_never_reports_a_partial_download_as_available() {
        let dir = tempfile::tempdir().unwrap();
        let model = get_model_by_name("gemma3:1b").unwrap();
        let expected_bytes = model.size_mb * 1024 * 1024;
        let manager = ModelManager::new_with_models_dir(Some(dir.path().to_path_buf())).unwrap();
        let installed = dir.path().join(&model.gguf_file);

        // Nearly complete, but still a .part file.
        std::fs::File::create(part_path(&installed)).unwrap().set_len(expected_bytes - 1).unwrap();
        manager.scan_models().await.unwrap();
        assert_eq!(manager.get_model_info(&model.name).await.unwrap().status, ModelStatus::NotDownloaded);

        // A file an older version left in place at 95% is not complete either.
        std::fs::File::create(&installed).unwrap().set_len(expected_bytes * 95 / 100).unwrap();
        manager.scan_models().await.unwrap();
        assert!(matches!(
            manager.get_model_info(&model.name).await.unwrap().status,
            ModelStatus::Corrupted { .. }
        ));

        std::fs::File::create(&installed).unwrap().set_len(expected_bytes).unwrap();
        manager.scan_models().await.unwrap();
        assert_eq!(manager.get_model_info(&model.name).await.unwrap().status, ModelStatus::Available);
    }

    /// Serves `MODEL_BYTES` once, honoring a `Range` request, and returns the Range it saw.
    /// With `truncate_at`, it declares the full length but closes after that many bytes.
    async fn serve_model_once(truncate_at: Option<usize>) -> (String, tokio::task::JoinHandle<Option<String>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/tiny.gguf", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = socket.read(&mut buffer).await.unwrap();
                request.extend_from_slice(&buffer[..read]);
            }
            let request = String::from_utf8_lossy(&request).to_string();
            let range = request.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("range").then(|| value.trim().to_string())
            });
            let start = range
                .as_deref()
                .and_then(|range| range.strip_prefix("bytes="))
                .and_then(|range| range.trim_end_matches('-').parse::<usize>().ok())
                .unwrap_or(0);
            let total = MODEL_BYTES.len();
            let headers = if start > 0 {
                format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                    total - start, start, total - 1, total
                )
            } else {
                format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", total)
            };
            socket.write_all(headers.as_bytes()).await.unwrap();
            let end = truncate_at.unwrap_or(total);
            socket.write_all(&MODEL_BYTES[start..end]).await.unwrap();
            socket.flush().await.unwrap();
            range
        });
        (url, server)
    }

    #[tokio::test]
    async fn download_resumes_a_part_file_and_installs_it_complete() {
        let dir = tempfile::tempdir().unwrap();
        let manager = ModelManager::new_with_models_dir(Some(dir.path().to_path_buf())).unwrap();
        let (url, server) = serve_model_once(None).await;
        let model = tiny_model(url);
        let installed = dir.path().join(&model.gguf_file);
        std::fs::write(part_path(&installed), &MODEL_BYTES[..12]).unwrap();

        manager.download_model_def(&model, None).await.unwrap();

        assert_eq!(server.await.unwrap().as_deref(), Some("bytes=12-"));
        assert_eq!(std::fs::read(&installed).unwrap(), MODEL_BYTES);
        assert!(!part_path(&installed).exists());
    }

    #[tokio::test]
    async fn interrupted_download_keeps_only_the_part_file() {
        let dir = tempfile::tempdir().unwrap();
        let manager = ModelManager::new_with_models_dir(Some(dir.path().to_path_buf())).unwrap();
        let (url, server) = serve_model_once(Some(20)).await;
        let model = tiny_model(url);
        let installed = dir.path().join(&model.gguf_file);

        assert!(manager.download_model_def(&model, None).await.is_err());
        server.await.unwrap();

        assert!(!installed.exists(), "a partial download must never reach the install path");
        assert_eq!(std::fs::read(part_path(&installed)).unwrap(), &MODEL_BYTES[..20]);
        assert!(!manager.active_downloads.read().await.contains(&model.name));
    }
}
