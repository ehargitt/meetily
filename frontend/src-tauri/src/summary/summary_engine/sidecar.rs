// Sidecar process lifecycle management for llama-helper
// Handles spawning, health checking, keep-alive, and graceful shutdown

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use super::models;

/// How long a health-check ping may take before the sidecar is considered wedged.
const PING_TIMEOUT: Duration = Duration::from_secs(5);

// ============================================================================
// Sidecar State Management
// ============================================================================

/// Sidecar process manager with keep-alive and health monitoring
pub struct SidecarManager {
    /// Child process handle
    child_process: Arc<Mutex<Option<Child>>>,

    /// Stdin writer for sending requests
    stdin_writer: Arc<Mutex<Option<ChildStdin>>>,

    /// Stdout lines for receiving responses. `Lines::next_line` is cancel safe, so a read that
    /// times out keeps any partial line for the next read.
    stdout_reader: Arc<Mutex<Option<Lines<BufReader<ChildStdout>>>>>,

    /// Held for each request/response exchange (and the spawn before it), so concurrent
    /// requests and health pings never interleave on the pipes.
    exchange_lock: Arc<Mutex<()>>,

    /// Incremented on every spawn; background loops of an earlier process exit when it changes.
    generation: Arc<AtomicU64>,

    /// Last activity timestamp
    last_activity: Arc<RwLock<Instant>>,

    /// Health status
    is_healthy: Arc<AtomicBool>,

    /// Shutdown flag
    should_shutdown: Arc<AtomicBool>,

    /// Active request count (for graceful shutdown)
    active_request_count: Arc<AtomicUsize>,

    /// Path to llama-helper binary
    helper_binary_path: PathBuf,

    /// Current model path (if loaded)
    current_model_path: Arc<RwLock<Option<PathBuf>>>,

    /// Idle timeout in seconds (configurable via env var)
    idle_timeout_secs: u64,
}

/// RAII guard for tracking active requests
/// Decrements the active request count when dropped
struct RequestGuard {
    counter: Arc<AtomicUsize>,
}

impl RequestGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self { counter }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}

const GENERATION_CANCELLED: &str = "Generation cancelled by user";

/// Resolves when `token` is cancelled; never without a token.
async fn cancelled(token: Option<&CancellationToken>) {
    match token {
        Some(token) => token.cancelled().await,
        None => std::future::pending().await,
    }
}

/// Whether a protocol line is a health-check `pong`.
fn is_pong(line: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .and_then(|value| value.get("type")?.as_str().map(|kind| kind == "pong"))
        .unwrap_or(false)
}

/// Reads the next non-empty protocol line. Unless `expect_pong`, a `pong` is skipped: it can
/// only be the late answer to a health ping, never the reply to the pending request.
async fn next_reply<R: AsyncBufRead + Unpin>(lines: &mut Lines<R>, expect_pong: bool) -> Result<String> {
    loop {
        let line = lines
            .next_line()
            .await
            .context("Failed to read response from stdout")?
            .ok_or_else(|| anyhow!("Sidecar closed stdout (process may have crashed)"))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if !expect_pong && is_pong(line) {
            log::debug!("Skipping stray pong from an earlier health check");
            continue;
        }
        return Ok(line.to_string());
    }
}

impl SidecarManager {
    /// Create a new sidecar manager
    pub fn new(_app_data_dir: PathBuf) -> Result<Self> {
        let helper_binary_path = Self::resolve_helper_binary()?;

        // Get idle timeout from env var or use default
        let idle_timeout_secs = std::env::var("LLAMA_IDLE_TIMEOUT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(models::DEFAULT_IDLE_TIMEOUT_SECS);

        log::info!(
            "SidecarManager initialized with idle timeout: {}s",
            idle_timeout_secs
        );
        log::info!("Helper binary path: {}", helper_binary_path.display());

        Ok(Self::with_helper_binary(helper_binary_path, idle_timeout_secs))
    }

    fn with_helper_binary(helper_binary_path: PathBuf, idle_timeout_secs: u64) -> Self {
        Self {
            child_process: Arc::new(Mutex::new(None)),
            stdin_writer: Arc::new(Mutex::new(None)),
            stdout_reader: Arc::new(Mutex::new(None)),
            exchange_lock: Arc::new(Mutex::new(())),
            generation: Arc::new(AtomicU64::new(0)),
            last_activity: Arc::new(RwLock::new(Instant::now())),
            is_healthy: Arc::new(AtomicBool::new(false)),
            should_shutdown: Arc::new(AtomicBool::new(false)),
            active_request_count: Arc::new(AtomicUsize::new(0)),
            helper_binary_path,
            current_model_path: Arc::new(RwLock::new(None)),
            idle_timeout_secs,
        }
    }

    /// Another handle on the same sidecar state, for background loops.
    fn handle(&self) -> Self {
        Self {
            child_process: self.child_process.clone(),
            stdin_writer: self.stdin_writer.clone(),
            stdout_reader: self.stdout_reader.clone(),
            exchange_lock: self.exchange_lock.clone(),
            generation: self.generation.clone(),
            last_activity: self.last_activity.clone(),
            is_healthy: self.is_healthy.clone(),
            should_shutdown: self.should_shutdown.clone(),
            active_request_count: self.active_request_count.clone(),
            helper_binary_path: self.helper_binary_path.clone(),
            current_model_path: self.current_model_path.clone(),
            idle_timeout_secs: self.idle_timeout_secs,
        }
    }

    /// Resolve the path to llama-helper binary
    fn resolve_helper_binary() -> Result<PathBuf> {
        // 1. Check environment variable (dev mode or manual override)
        if let Ok(env_path) = std::env::var("MEETILY_LLAMA_HELPER") {
            if !env_path.is_empty() {
                let path = PathBuf::from(env_path);
                if path.exists() {
                    log::info!("Using llama-helper from MEETILY_LLAMA_HELPER: {}", path.display());
                    return Ok(path);
                }
            }
        }

        // In production, Tauri bundles the binary with target triple suffix
        // 2. Check relative to current executable (most reliable for AppImage/bundled apps)
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                log::info!("Searching for llama-helper relative to executable: {}", exe_dir.display());
                
                // Get the target triple (same logic as before)
                let target_triple = std::env::var("TARGET")
                    .unwrap_or_else(|_| {
                        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
                        { "x86_64-unknown-linux-gnu".to_string() }
                        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
                        { "aarch64-unknown-linux-gnu".to_string() }
                        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
                        { "x86_64-apple-darwin".to_string() }
                        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                        { "aarch64-apple-darwin".to_string() }
                        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
                        { "x86_64-pc-windows-msvc".to_string() }
                        #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
                        { "aarch64-pc-windows-msvc".to_string() }
                        #[cfg(not(any(
                            all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")),
                            all(target_os = "macos", any(target_arch = "x86_64", target_arch = "aarch64")),
                            all(target_os = "windows", any(target_arch = "x86_64", target_arch = "aarch64"))
                        )))]
                        { "unknown".to_string() }
                    });

                let binary_name = if cfg!(windows) {
                    format!("llama-helper-{}.exe", target_triple)
                } else {
                    format!("llama-helper-{}", target_triple)
                };

                // Try exact match in exe dir
                let bundled = exe_dir.join(&binary_name);
                if bundled.exists() {
                    log::info!("Found exact match next to executable: {}", bundled.display());
                    return Ok(bundled);
                }

                // Fuzzy match in exe dir
                log::info!("Attempting fuzzy match in exe dir: {}", exe_dir.display());
                if let Ok(entries) = std::fs::read_dir(exe_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                            if name.starts_with("llama-helper") && !name.ends_with(".d") {
                                log::info!("Found fuzzy match next to executable: {}", path.display());
                                return Ok(path);
                            }
                        }
                    }
                }
            }
        }

        // 3. Check bundled resources (RESOURCE_DIR) - Fallback
        if let Ok(resource_dir) = std::env::var("RESOURCE_DIR") {
            log::info!("Searching for llama-helper in RESOURCE_DIR: {}", resource_dir);
            let resource_path = PathBuf::from(&resource_dir);
             // Get the target triple again (or we could have shared it, but code duplication is safer for this tool usage)
            let target_triple = std::env::var("TARGET")
                .unwrap_or_else(|_| {
                     #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
                    { "x86_64-unknown-linux-gnu".to_string() }
                    // ... (abbreviated for brevity in thought, but must be full in tool)
                     #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
                    { "aarch64-unknown-linux-gnu".to_string() }
                    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
                    { "x86_64-apple-darwin".to_string() }
                    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                    { "aarch64-apple-darwin".to_string() }
                    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
                    { "x86_64-pc-windows-msvc".to_string() }
                    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
                    { "aarch64-pc-windows-msvc".to_string() }
                    #[cfg(not(any(
                        all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")),
                        all(target_os = "macos", any(target_arch = "x86_64", target_arch = "aarch64")),
                        all(target_os = "windows", any(target_arch = "x86_64", target_arch = "aarch64"))
                    )))]
                    { "unknown".to_string() }
                });

            let binary_name = if cfg!(windows) {
                format!("llama-helper-{}.exe", target_triple)
            } else {
                format!("llama-helper-{}", target_triple)
            };

            let bundled = resource_path.join(&binary_name);
            if bundled.exists() {
                log::info!("Found exact match in RESOURCE_DIR: {}", bundled.display());
                return Ok(bundled);
            }

            // Fuzzy match in RESOURCE_DIR
            if let Ok(entries) = std::fs::read_dir(&resource_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("llama-helper") && !name.ends_with(".d") {
                            log::info!("Found fuzzy match in RESOURCE_DIR: {}", path.display());
                            return Ok(path);
                        }
                    }
                }
            }
        } else {
            log::warn!("RESOURCE_DIR environment variable not set");
        }

        // 3. Fallback for dev: try relative paths from workspace (no target triple in dev builds)
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let project_root = PathBuf::from(&manifest_dir)
                .parent()
                .and_then(|p| p.parent())
                .ok_or_else(|| anyhow!("Failed to determine project root"))?
                .to_path_buf();

            let candidates = vec![
                project_root.join("target/release/llama-helper"),
                project_root.join("target/debug/llama-helper"),
                project_root.join("target/release/llama-helper.exe"),
                project_root.join("target/debug/llama-helper.exe"),
            ];

            for candidate in candidates {
                if candidate.exists() {
                    log::info!("Using dev llama-helper: {}", candidate.display());
                    return Ok(candidate);
                }
            }
        }

        Err(anyhow!(
            "llama-helper binary not found. Build with 'cd llama-helper && cargo build --release' or set MEETILY_LLAMA_HELPER env var."
        ))
    }

    /// Runs one request against a sidecar serving `model_path` and returns its reply line.
    ///
    /// The sidecar is (re)spawned first when it is not running, runs another model, was marked
    /// unhealthy, or its process has exited. Requests run one at a time; `timeout` covers only
    /// this request's own exchange, not the wait for earlier requests to finish. Any I/O
    /// failure marks the sidecar unhealthy so the next request starts a fresh process.
    ///
    /// Cancelling while the request still waits its turn just abandons it. Cancelling once it
    /// owns the exchange shuts the sidecar down to stop the generation.
    pub async fn request(
        &self,
        model_path: PathBuf,
        request_json: String,
        timeout: Duration,
        cancellation: Option<&CancellationToken>,
    ) -> Result<String> {
        // Track active request
        let _guard = RequestGuard::new(self.active_request_count.clone());
        let _exchange = tokio::select! {
            biased;
            _ = cancelled(cancellation) => return Err(anyhow!(GENERATION_CANCELLED)),
            exchange = self.exchange_lock.lock() => exchange,
        };

        self.ensure_running(model_path).await?;

        let exchange = tokio::time::timeout(timeout, async {
            self.write_line(&request_json).await?;
            self.read_reply(false).await
        });
        // Resolve to a value first so the exchange future (and the pipe locks it may hold) is
        // dropped before any shutdown below.
        let outcome = tokio::select! {
            biased;
            _ = cancelled(cancellation) => None,
            result = exchange => Some(result),
        };
        match outcome {
            Some(Ok(Ok(response))) => {
                self.update_activity().await;
                Ok(response)
            }
            Some(Ok(Err(e))) => {
                log::error!("Sidecar request failed, will respawn on the next request: {:#}", e);
                self.is_healthy.store(false, Ordering::SeqCst);
                Err(e)
            }
            Some(Err(_)) => {
                // Timeout reached - shutdown sidecar to stop generation
                log::error!("Request timeout after {:?}, shutting down sidecar", timeout);
                if let Err(shutdown_err) = self.shutdown().await {
                    log::error!("Failed to shutdown sidecar after timeout: {}", shutdown_err);
                }
                Err(anyhow!("Request timed out after {:?}", timeout))
            }
            None => {
                log::warn!("Generation cancelled by user, shutting down sidecar");
                if let Err(e) = self.shutdown().await {
                    log::error!("Failed to shutdown sidecar during cancellation: {}", e);
                }
                Err(anyhow!(GENERATION_CANCELLED))
            }
        }
    }

    /// Spawns the sidecar unless one is already serving `model_path` and alive. Callers hold
    /// `exchange_lock`.
    async fn ensure_running(&self, model_path: PathBuf) -> Result<()> {
        let same_model = self.current_model_path.read().await.as_ref() == Some(&model_path);
        if same_model && self.is_healthy() && !self.child_has_exited().await {
            log::debug!("Sidecar already running with correct model");
            self.update_activity().await;
            return Ok(());
        }

        // Need to spawn or restart
        self.spawn(model_path).await
    }

    /// Whether the helper process is gone. `try_wait` also reaps an exited child.
    async fn child_has_exited(&self) -> bool {
        let mut child = self.child_process.lock().await;
        match child.as_mut().map(Child::try_wait) {
            Some(Ok(None)) => false,
            Some(Ok(Some(status))) => {
                log::warn!("llama-helper exited with status: {}", status);
                true
            }
            Some(Err(e)) => {
                log::warn!("Failed to check llama-helper status: {}", e);
                true
            }
            None => true,
        }
    }

    /// Spawn the sidecar process
    async fn spawn(&self, model_path: PathBuf) -> Result<()> {
        // Shutdown existing process if running
        self.shutdown().await?;

        log::info!("Spawning llama-helper sidecar");
        log::info!("Model path: {}", model_path.display());

        #[cfg(unix)]
        let mut command = tokio::process::Command::new("nice");

        #[cfg(not(unix))]
        let mut command = tokio::process::Command::new(&self.helper_binary_path);

        #[cfg(unix)]
        command.arg("-n").arg("10").arg(&self.helper_binary_path);

        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Log stderr to main process
            .env("LLAMA_IDLE_TIMEOUT", self.idle_timeout_secs.to_string());

        #[cfg(target_os = "windows")]
        {
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x00004000;

            command.creation_flags(CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS);
        }

        let mut child = command
            .spawn()
            .with_context(|| format!("Failed to spawn llama-helper at {:?}", self.helper_binary_path))?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("Failed to get stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("Failed to get stdout"))?;

        // Store handles
        {
            let mut child_lock = self.child_process.lock().await;
            *child_lock = Some(child);
        }

        {
            let mut stdin_lock = self.stdin_writer.lock().await;
            *stdin_lock = Some(stdin);
        }

        {
            let mut stdout_lock = self.stdout_reader.lock().await;
            *stdout_lock = Some(BufReader::new(stdout).lines());
        }

        // Update state
        {
            let mut current_model = self.current_model_path.write().await;
            *current_model = Some(model_path);
        }

        self.is_healthy.store(true, Ordering::SeqCst);
        self.should_shutdown.store(false, Ordering::SeqCst);
        self.update_activity().await;

        log::info!("Sidecar spawned successfully");

        // Start background tasks for this process only
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.start_health_check_loop(generation);
        self.start_idle_check_loop(generation);

        Ok(())
    }

    /// Writes one request line to the sidecar's stdin.
    async fn write_line(&self, line: &str) -> Result<()> {
        let mut stdin_lock = self.stdin_writer.lock().await;
        let stdin = stdin_lock
            .as_mut()
            .ok_or_else(|| anyhow!("Sidecar not running"))?;

        stdin
            .write_all(line.as_bytes())
            .await
            .context("Failed to write request to stdin")?;
        stdin
            .write_all(b"\n")
            .await
            .context("Failed to write newline")?;
        stdin.flush().await.context("Failed to flush stdin")?;
        Ok(())
    }

    /// Reads the reply to the request just written (see [`next_reply`]).
    async fn read_reply(&self, expect_pong: bool) -> Result<String> {
        let mut stdout_lock = self.stdout_reader.lock().await;
        let lines = stdout_lock
            .as_mut()
            .ok_or_else(|| anyhow!("Sidecar not running"))?;
        next_reply(lines, expect_pong).await
    }

    /// Send ping to keep sidecar alive. Skipped while a request holds the pipes: a sidecar
    /// answering a request is alive.
    async fn send_ping(&self) -> Result<()> {
        // Note: We don't use request() here to avoid incrementing active_request_count
        // for internal health checks, as that would prevent graceful shutdown
        let Ok(_exchange) = self.exchange_lock.try_lock() else {
            return Ok(());
        };
        let request = serde_json::json!({"type": "ping"}).to_string();
        let response = tokio::time::timeout(PING_TIMEOUT, async {
            self.write_line(&request).await?;
            self.read_reply(true).await
        })
        .await
        .map_err(|_| anyhow!("No ping response within {:?}", PING_TIMEOUT))??;

        if is_pong(&response) {
            Ok(())
        } else {
            Err(anyhow!("Unexpected ping response: {}", response))
        }
    }

    /// Gracefully shutdown the sidecar
    /// Waits for active requests to complete before killing the process
    pub async fn shutdown_gracefully(&self) -> Result<()> {
        log::info!("Initiating graceful shutdown of sidecar");

        // Set shutdown flag to prevent new internal tasks
        self.should_shutdown.store(true, Ordering::SeqCst);

        // Wait for active requests to complete
        // We poll every 500ms
        let start = Instant::now();
        let max_wait = Duration::from_secs(600); // Wait up to 10 minutes for long generations

        loop {
            let count = self.active_request_count.load(Ordering::SeqCst);
            if count == 0 {
                log::info!("No active requests, proceeding with shutdown");
                break;
            }

            if start.elapsed() > max_wait {
                log::warn!("Timed out waiting for active requests ({} active), forcing shutdown", count);
                break;
            }

            log::debug!("Waiting for {} active requests to complete...", count);
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        self.shutdown().await
    }

    /// Force shutdown the sidecar
    pub async fn shutdown(&self) -> Result<()> {
        // Set shutdown flag
        self.should_shutdown.store(true, Ordering::SeqCst);

        // Send shutdown command
        if self.is_healthy() {
            let request = serde_json::json!({"type": "shutdown"}).to_string();

            // Try to send shutdown command, but ignore errors
            // We don't use request() to avoid incrementing counter
            let _ = self.write_line(&request).await;
        }

        // Kill process if still running
        {
            let mut child_lock = self.child_process.lock().await;
            if let Some(mut child) = child_lock.take() {
                match tokio::time::timeout(Duration::from_secs(3), child.wait()).await {
                    Ok(Ok(status)) => {
                        log::info!("Sidecar exited with status: {}", status);
                    }
                    Ok(Err(e)) => {
                        log::error!("Failed to wait for sidecar: {}", e);
                    }
                    Err(_) => {
                        log::warn!("Sidecar didn't exit gracefully, killing");
                        let _ = child.kill().await;
                    }
                }
            }
        }

        // Clear handles
        {
            let mut stdin_lock = self.stdin_writer.lock().await;
            *stdin_lock = None;
        }

        {
            let mut stdout_lock = self.stdout_reader.lock().await;
            *stdout_lock = None;
        }

        {
            let mut current_model = self.current_model_path.write().await;
            *current_model = None;
        }

        self.is_healthy.store(false, Ordering::SeqCst);

        log::info!("Sidecar shutdown complete");
        Ok(())
    }

    /// Check if sidecar is healthy
    pub fn is_healthy(&self) -> bool {
        self.is_healthy.load(Ordering::SeqCst)
    }

    /// Update last activity timestamp
    async fn update_activity(&self) {
        let mut last_activity = self.last_activity.write().await;
        *last_activity = Instant::now();
    }

    /// Get seconds since last activity
    async fn seconds_since_activity(&self) -> u64 {
        let last_activity = self.last_activity.read().await;
        last_activity.elapsed().as_secs()
    }

    /// Whether the loops started for process `generation` should stop.
    fn loop_is_stale(&self, generation: u64) -> bool {
        self.should_shutdown.load(Ordering::SeqCst)
            || self.generation.load(Ordering::SeqCst) != generation
    }

    /// Start health check loop (runs in background)
    fn start_health_check_loop(&self, generation: u64) {
        let manager = self.handle();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                if manager.loop_is_stale(generation) {
                    log::debug!("Health check loop: sidecar replaced or shut down, exiting");
                    break;
                }

                if !manager.is_healthy() {
                    log::debug!("Health check loop: sidecar unhealthy, skipping ping");
                    continue;
                }

                // Don't ping if we are busy with a request
                if manager.active_request_count.load(Ordering::SeqCst) > 0 {
                    continue;
                }

                if manager.child_has_exited().await {
                    manager.is_healthy.store(false, Ordering::SeqCst);
                    continue;
                }

                log::debug!("Health check: sending ping");
                if let Err(e) = manager.send_ping().await {
                    log::warn!("Health check failed: {}", e);
                    manager.is_healthy.store(false, Ordering::SeqCst);
                }
            }

            log::debug!("Health check loop exited");
        });
    }

    /// Start idle check loop (runs in background)
    fn start_idle_check_loop(&self, generation: u64) {
        let manager = self.handle();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                if manager.loop_is_stale(generation) {
                    log::debug!("Idle check loop: sidecar replaced or shut down, exiting");
                    break;
                }

                // Don't shutdown if we are busy
                if manager.active_request_count.load(Ordering::SeqCst) > 0 {
                    // Update activity to prevent timeout immediately after request finishes
                    manager.update_activity().await;
                    continue;
                }

                let idle_secs = manager.seconds_since_activity().await;
                log::debug!("Idle check: {}s since last activity", idle_secs);

                if idle_secs > manager.idle_timeout_secs {
                    // A request that started since the check above holds the pipes.
                    let Ok(_exchange) = manager.exchange_lock.try_lock() else {
                        continue;
                    };
                    log::info!(
                        "Sidecar idle for {}s (timeout: {}s), shutting down",
                        idle_secs,
                        manager.idle_timeout_secs
                    );

                    if let Err(e) = manager.shutdown().await {
                        log::error!("Failed to shutdown idle sidecar: {}", e);
                    }

                    break;
                }
            }

            log::debug!("Idle check loop exited");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stray_pongs_are_skipped_unless_a_pong_is_expected() {
        let input: &[u8] = b"{\"type\":\"pong\"}\n\n{\"type\":\"response\",\"text\":\"ok\",\"error\":null}\n";
        let mut lines = BufReader::new(input).lines();
        assert_eq!(
            next_reply(&mut lines, false).await.unwrap(),
            r#"{"type":"response","text":"ok","error":null}"#
        );

        let mut lines = BufReader::new(input).lines();
        assert_eq!(next_reply(&mut lines, true).await.unwrap(), r#"{"type":"pong"}"#);
    }

    #[tokio::test]
    async fn a_read_that_times_out_keeps_its_partial_line() {
        let (mut writer, reader) = tokio::io::duplex(64);
        let mut lines = BufReader::new(reader).lines();

        writer.write_all(br#"{"type":"resp"#).await.unwrap();
        let timed_out = tokio::time::timeout(Duration::from_millis(100), next_reply(&mut lines, false)).await;
        assert!(timed_out.is_err());

        writer.write_all(b"onse\",\"text\":\"late\"}\n").await.unwrap();
        assert_eq!(
            next_reply(&mut lines, false).await.unwrap(),
            r#"{"type":"response","text":"late"}"#
        );
    }

    #[tokio::test]
    async fn closed_stdout_is_an_error() {
        let mut lines = BufReader::new(&b""[..]).lines();
        assert!(next_reply(&mut lines, false).await.is_err());
    }

    /// A stand-in for llama-helper that echoes each request's prompt, counts its spawns in
    /// `spawns`, and misbehaves on the prompts `crash`, `exit-after-reply`, `stray` and `slow`.
    #[cfg(unix)]
    fn fake_helper(dir: &std::path::Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let script = dir.join("fake-llama-helper");
        std::fs::write(
            &script,
            format!(
                r#"#!/bin/sh
echo spawn >> '{spawns}'
while IFS= read -r line; do
  prompt=$(printf '%s' "$line" | sed -n 's/.*"prompt":"\([^"]*\)".*/\1/p')
  case "$line" in
    *'"ping"'*) echo '{{"type":"pong"}}' ;;
    *'"shutdown"'*) echo '{{"type":"goodbye"}}'; exit 0 ;;
    *) case "$prompt" in
         crash) exit 3 ;;
         stray) echo '{{"type":"pong"}}' ;;
         slow) sleep 1 ;;
       esac
       echo "{{\"type\":\"response\",\"text\":\"$prompt\",\"error\":null}}"
       [ "$prompt" = exit-after-reply ] && exit 0 ;;
  esac
done
"#,
                spawns = dir.join("spawns").display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    #[cfg(unix)]
    fn spawn_count(dir: &std::path::Path) -> usize {
        std::fs::read_to_string(dir.join("spawns")).map_or(0, |spawns| spawns.lines().count())
    }

    fn generate(prompt: &str) -> String {
        serde_json::json!({"type": "generate", "prompt": prompt}).to_string()
    }

    fn reply(prompt: &str) -> String {
        format!(r#"{{"type":"response","text":"{prompt}","error":null}}"#)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_io_error_marks_the_sidecar_unhealthy_and_the_next_request_respawns_it() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SidecarManager::with_helper_binary(fake_helper(dir.path()), 300);
        let model = dir.path().join("model.gguf");
        let timeout = Duration::from_secs(5);

        assert!(manager.request(model.clone(), generate("crash"), timeout, None).await.is_err());
        assert!(!manager.is_healthy());

        let response = manager.request(model, generate("hello"), timeout, None).await.unwrap();
        assert_eq!(response, reply("hello"));
        assert!(manager.is_healthy());
        assert_eq!(spawn_count(dir.path()), 2);
        manager.shutdown().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_exited_helper_is_respawned_although_no_error_was_seen() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SidecarManager::with_helper_binary(fake_helper(dir.path()), 300);
        let model = dir.path().join("model.gguf");
        let timeout = Duration::from_secs(5);

        let first = manager.request(model.clone(), generate("exit-after-reply"), timeout, None).await;
        assert_eq!(first.unwrap(), reply("exit-after-reply"));
        assert!(manager.is_healthy(), "the exit is not observed yet");
        tokio::time::sleep(Duration::from_millis(500)).await;

        let response = manager.request(model, generate("again"), timeout, None).await.unwrap();
        assert_eq!(response, reply("again"));
        assert_eq!(spawn_count(dir.path()), 2);
        manager.shutdown().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_stray_pong_does_not_become_the_reply() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SidecarManager::with_helper_binary(fake_helper(dir.path()), 300);
        let model = dir.path().join("model.gguf");

        let response = manager
            .request(model, generate("stray"), Duration::from_secs(5), None)
            .await
            .unwrap();
        assert_eq!(response, reply("stray"));
        manager.shutdown().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn concurrent_requests_are_serialized_and_time_out_only_on_their_own_work() {
        let dir = tempfile::tempdir().unwrap();
        let manager = Arc::new(SidecarManager::with_helper_binary(fake_helper(dir.path()), 300));
        let model = dir.path().join("model.gguf");

        let slow = {
            let (manager, model) = (manager.clone(), model.clone());
            tokio::spawn(async move {
                manager.request(model, generate("slow"), Duration::from_secs(5), None).await
            })
        };
        tokio::time::sleep(Duration::from_millis(200)).await;
        // Waits ~800 ms for the slow request, longer than its own 500 ms timeout.
        let quick = manager
            .request(model, generate("quick"), Duration::from_millis(500), None)
            .await;

        assert_eq!(slow.await.unwrap().unwrap(), reply("slow"));
        assert_eq!(quick.unwrap(), reply("quick"));
        assert_eq!(spawn_count(dir.path()), 1);
        manager.shutdown().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_a_queued_request_leaves_the_running_one_alone() {
        let dir = tempfile::tempdir().unwrap();
        let manager = Arc::new(SidecarManager::with_helper_binary(fake_helper(dir.path()), 300));
        let model = dir.path().join("model.gguf");

        let slow = {
            let (manager, model) = (manager.clone(), model.clone());
            tokio::spawn(async move {
                manager.request(model, generate("slow"), Duration::from_secs(5), None).await
            })
        };
        tokio::time::sleep(Duration::from_millis(200)).await;
        let token = CancellationToken::new();
        let queued = manager.request(model, generate("queued"), Duration::from_secs(5), Some(&token));
        let cancel_soon = async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            token.cancel();
        };
        let (queued, ()) = tokio::join!(queued, cancel_soon);

        assert_eq!(queued.unwrap_err().to_string(), GENERATION_CANCELLED);
        assert_eq!(slow.await.unwrap().unwrap(), reply("slow"));
        assert!(manager.is_healthy());
        assert_eq!(spawn_count(dir.path()), 1);
        manager.shutdown().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_the_running_request_shuts_the_sidecar_down() {
        let dir = tempfile::tempdir().unwrap();
        let manager = SidecarManager::with_helper_binary(fake_helper(dir.path()), 300);
        let model = dir.path().join("model.gguf");
        let token = CancellationToken::new();

        let running = manager.request(model.clone(), generate("slow"), Duration::from_secs(5), Some(&token));
        let cancel_soon = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            token.cancel();
        };
        let (running, ()) = tokio::join!(running, cancel_soon);

        assert_eq!(running.unwrap_err().to_string(), GENERATION_CANCELLED);
        assert!(!manager.is_healthy());
        let next = manager.request(model, generate("next"), Duration::from_secs(5), None).await;
        assert_eq!(next.unwrap(), reply("next"));
        assert_eq!(spawn_count(dir.path()), 2);
        manager.shutdown().await.unwrap();
    }
}
