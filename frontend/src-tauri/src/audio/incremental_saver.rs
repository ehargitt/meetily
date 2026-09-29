use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use anyhow::{Result, anyhow};
use log::{info, warn, error};
use super::recording_state::AudioChunk;
use serde::{Serialize, Deserialize};

use super::ffmpeg::find_ffmpeg_path;

/// Seconds of audio per checkpoint file.
const CHECKPOINT_SECONDS: usize = 30;
/// Unsaved audio kept in memory while checkpoint writes keep failing. Beyond
/// this the oldest unsaved audio is dropped so a full disk cannot exhaust RAM.
const MAX_PENDING_SECONDS: usize = 600;
const RETRY_BACKOFF_BASE: Duration = Duration::from_secs(5);
const RETRY_BACKOFF_MAX: Duration = Duration::from_secs(60);
/// While audio is being dropped at the cap, log the running total at most this often.
const DROP_LOG_INTERVAL: Duration = Duration::from_secs(30);

/// Incremental audio saver that writes a 16-bit PCM WAV checkpoint every 30 seconds
/// to bound memory use and enable crash recovery, then encodes all checkpoints
/// into one AAC file at finalize.
///
/// Checkpoints are PCM rather than AAC so they concatenate sample-exactly: every
/// AAC encode adds its own priming/padding, and `concat -c copy` of per-checkpoint
/// AAC files accumulated that as drift (~4.5 s per hour against the transcript).
pub struct IncrementalAudioSaver {
    pending: Vec<f32>,
    checkpoint_interval_samples: usize,
    max_pending_samples: usize,
    checkpoint_count: u32,
    checkpoints_dir: PathBuf,
    meeting_folder: PathBuf,
    sample_rate: u32,
    consecutive_failures: u32,
    next_attempt_at: Option<Instant>,
    /// Samples dropped at the in-memory cap since the last drop log.
    dropped_since_log: usize,
    last_drop_log: Option<Instant>,
    #[cfg(test)]
    write_attempts: u64,
}

impl IncrementalAudioSaver {
    /// Create a new incremental saver
    ///
    /// # Arguments
    /// * `meeting_folder` - Path to the meeting folder (contains .checkpoints/)
    /// * `sample_rate` - Sample rate of audio (typically 48000)
    pub fn new(meeting_folder: PathBuf, sample_rate: u32) -> Result<Self> {
        let checkpoints_dir = meeting_folder.join(".checkpoints");

        // Verify checkpoints directory exists
        if !checkpoints_dir.exists() {
            return Err(anyhow!("Checkpoints directory does not exist: {}", checkpoints_dir.display()));
        }

        Ok(Self {
            pending: Vec::new(),
            checkpoint_interval_samples: sample_rate as usize * CHECKPOINT_SECONDS,
            max_pending_samples: sample_rate as usize * MAX_PENDING_SECONDS,
            checkpoint_count: 0,
            checkpoints_dir,
            meeting_folder,
            sample_rate,
            consecutive_failures: 0,
            next_attempt_at: None,
            dropped_since_log: 0,
            last_drop_log: None,
            #[cfg(test)]
            write_attempts: 0,
        })
    }

    /// Add an audio chunk to the buffer, writing a checkpoint once 30 seconds are buffered.
    ///
    /// An error means audio could not be written to disk (the caller reports it);
    /// the audio stays buffered and the write is retried with backoff, not on
    /// every chunk.
    pub fn add_chunk(&mut self, chunk: AudioChunk) -> Result<()> {
        self.add_samples_at(&chunk.data, Instant::now())
    }

    fn add_samples_at(&mut self, samples: &[f32], now: Instant) -> Result<()> {
        self.pending.extend_from_slice(samples);

        if self.pending.len() < self.checkpoint_interval_samples {
            return Ok(());
        }
        if self.next_attempt_at.is_some_and(|retry_at| now < retry_at) {
            return self.enforce_pending_limit(now);
        }

        match self.write_pending_checkpoint() {
            Ok(()) => {
                self.consecutive_failures = 0;
                self.next_attempt_at = None;
                Ok(())
            }
            Err(e) => {
                self.consecutive_failures += 1;
                let backoff = RETRY_BACKOFF_BASE
                    .saturating_mul(1u32 << (self.consecutive_failures - 1).min(4))
                    .min(RETRY_BACKOFF_MAX);
                self.next_attempt_at = Some(now + backoff);
                warn!(
                    "Checkpoint write failed ({} in a row); retrying in {}s: {}",
                    self.consecutive_failures,
                    backoff.as_secs(),
                    e
                );
                self.enforce_pending_limit(now)?;
                Err(e)
            }
        }
    }

    /// Drop the oldest unsaved audio beyond the in-memory cap.
    fn enforce_pending_limit(&mut self, now: Instant) -> Result<()> {
        let excess = self.pending.len().saturating_sub(self.max_pending_samples);
        if excess == 0 {
            return Ok(());
        }
        self.pending.drain(..excess);
        self.dropped_since_log += excess;
        if self.last_drop_log.map_or(true, |at| now.duration_since(at) >= DROP_LOG_INTERVAL) {
            error!(
                "Dropped {:.1}s of unsaved audio: checkpoints cannot be written",
                self.dropped_since_log as f64 / self.sample_rate as f64
            );
            self.dropped_since_log = 0;
            self.last_drop_log = Some(now);
        }
        let dropped_seconds = excess as f64 / self.sample_rate as f64;
        Err(anyhow!(
            "Audio cannot be saved to disk; {:.0} seconds of audio were lost",
            dropped_seconds
        ))
    }

    /// Write the buffered audio as the next checkpoint and clear the buffer.
    fn write_pending_checkpoint(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }

        #[cfg(test)]
        {
            self.write_attempts += 1;
        }
        let checkpoint_path = self.checkpoints_dir
            .join(format!("audio_chunk_{:03}.wav", self.checkpoint_count));
        write_wav_atomically(&checkpoint_path, &self.pending, self.sample_rate)?;

        let duration_seconds = self.pending.len() as f32 / self.sample_rate as f32;
        self.checkpoint_count += 1;
        info!("Saved checkpoint {}: {:.2}s of audio ({} samples)",
              self.checkpoint_count,
              duration_seconds,
              self.pending.len());
        self.pending.clear();

        Ok(())
    }

    /// Finalize the recording: save final checkpoint, encode all checkpoints, cleanup
    ///
    /// Returns the path to the final audio.mp4 file. On failure the checkpoints
    /// stay on disk so the recording can still be recovered.
    pub async fn finalize(&mut self) -> Result<PathBuf> {
        info!("Finalizing incremental recording...");

        if !self.pending.is_empty() {
            info!("Saving final checkpoint with remaining {} samples", self.pending.len());
            // Up to MAX_PENDING_SECONDS of audio after write failures: write it off the runtime.
            let path = self.checkpoints_dir
                .join(format!("audio_chunk_{:03}.wav", self.checkpoint_count));
            let samples = std::mem::take(&mut self.pending);
            let sample_rate = self.sample_rate;
            let (written, samples) = tokio::task::spawn_blocking(move || {
                let written = write_wav_atomically(&path, &samples, sample_rate);
                (written, samples)
            })
            .await
            .map_err(|e| anyhow!("Final checkpoint write task failed: {}", e))?;
            if let Err(e) = written {
                self.pending = samples;
                return Err(e);
            }
            self.checkpoint_count += 1;
        }

        if self.checkpoint_count == 0 {
            return Err(anyhow!("No audio checkpoints to merge - recording may have failed"));
        }

        let checkpoints: Vec<PathBuf> = (0..self.checkpoint_count)
            .map(|i| self.checkpoints_dir.join(format!("audio_chunk_{:03}.wav", i)))
            .collect();
        let final_audio_path = self.meeting_folder.join("audio.mp4");
        encode_wav_checkpoints(&checkpoints, &self.checkpoints_dir, &final_audio_path).await?;

        // Clean up checkpoints directory
        info!("Cleaning up {} checkpoint files", self.checkpoint_count);
        if let Err(e) = std::fs::remove_dir_all(&self.checkpoints_dir) {
            warn!("Failed to clean up checkpoints directory: {}", e);
            // Non-fatal - user can manually delete
        }

        info!("Finalized recording: {}", final_audio_path.display());

        Ok(final_audio_path)
    }

    /// Get the meeting folder path
    pub fn get_meeting_folder(&self) -> &PathBuf {
        &self.meeting_folder
    }

    /// Get current checkpoint count
    pub fn get_checkpoint_count(&self) -> u32 {
        self.checkpoint_count
    }

    /// Audio is buffered that no checkpoint holds yet.
    pub fn has_unsaved_audio(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// Write mono samples as a 16-bit PCM WAV via a temp file and rename, so a
/// crash never leaves a truncated checkpoint under the real name.
fn write_wav_atomically(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    const BYTES_PER_SAMPLE: u32 = 2;
    let data_len = u32::try_from(samples.len() as u64 * BYTES_PER_SAMPLE as u64)
        .map_err(|_| anyhow!("Checkpoint too large for a WAV file: {} samples", samples.len()))?;

    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * BYTES_PER_SAMPLE).to_le_bytes()); // byte rate
    bytes.extend_from_slice(&(BYTES_PER_SAMPLE as u16).to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }

    let temp_path = path.with_extension("wav.tmp");
    let mut file = std::fs::File::create(&temp_path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temp_path, path)?;
    Ok(())
}

/// Checkpoint files in `dir` with the given extension, in recording order.
fn checkpoint_files(dir: &Path, extension: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some(extension))
        .collect();
    // audio_chunk_000, audio_chunk_001, ... sort lexically into recording order
    files.sort();
    Ok(files)
}

/// Write an FFmpeg concat-demuxer list for `files` and return its path.
fn write_concat_list(files: &[PathBuf], work_dir: &Path) -> Result<PathBuf> {
    let mut list_content = String::new();
    for file in files {
        // Use absolute path for FFmpeg (required for safe mode); quote per the
        // concat format so names with apostrophes (e.g. "Bob's sync") work.
        let abs_path = file.canonicalize()?;
        let quoted = abs_path.to_string_lossy().replace('\'', "'\\''");
        list_content.push_str(&format!("file '{}'\n", quoted));
    }
    let list_file = work_dir.join("concat_list.txt");
    std::fs::write(&list_file, list_content)?;
    Ok(list_file)
}

/// Run FFmpeg with `args` on the async runtime. The child is killed if the
/// future is dropped, so a caller's timeout really ends the encode.
async fn run_ffmpeg(args: &[&std::ffi::OsStr]) -> Result<()> {
    let ffmpeg_path = find_ffmpeg_path()
        .ok_or_else(|| anyhow!("FFmpeg not found. Please install FFmpeg to finalize recordings."))?;
    info!("Using FFmpeg at: {:?}", ffmpeg_path);

    let mut command = tokio::process::Command::new(ffmpeg_path);
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
        .args(args)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);

    // Hide console window on Windows to prevent CMD popup during finalization
    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let output = command.output().await
        .map_err(|e| anyhow!("Failed to run FFmpeg: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("FFmpeg failed: {}", stderr);
        return Err(anyhow!("FFmpeg failed ({}): {}", output.status, stderr.trim()));
    }
    Ok(())
}

/// Encode WAV checkpoints, in order, into one AAC `output` (written via a temp
/// file). The concat demuxer joins PCM sample-exactly and a single encode adds
/// a single priming delay, so the result is exactly as long as the input.
async fn encode_wav_checkpoints(checkpoints: &[PathBuf], work_dir: &Path, output: &Path) -> Result<()> {
    info!("Encoding {} checkpoints into {}", checkpoints.len(), output.display());
    for checkpoint in checkpoints {
        if !checkpoint.exists() {
            return Err(anyhow!("Checkpoint file missing: {}", checkpoint.display()));
        }
    }

    let list_file = write_concat_list(checkpoints, work_dir)?;
    let temp_output = output.with_extension("mp4.part");
    run_ffmpeg(&[
        "-f".as_ref(), "concat".as_ref(),
        "-safe".as_ref(), "0".as_ref(),
        "-i".as_ref(), list_file.as_os_str(),
        "-c:a".as_ref(), "aac".as_ref(),
        "-b:a".as_ref(), "192k".as_ref(),
        "-profile:a".as_ref(), "aac_low".as_ref(),
        "-movflags".as_ref(), "+faststart".as_ref(),
        "-f".as_ref(), "mp4".as_ref(),
        "-y".as_ref(), temp_output.as_os_str(),
    ]).await?;
    std::fs::rename(&temp_output, output)?;

    info!("Successfully encoded {} checkpoints → {}", checkpoints.len(), output.display());
    Ok(())
}

/// Join AAC checkpoints written by earlier versions (before WAV checkpoints).
async fn concat_legacy_mp4_checkpoints(checkpoints: &[PathBuf], work_dir: &Path, output: &Path) -> Result<()> {
    let list_file = write_concat_list(checkpoints, work_dir)?;
    let temp_output = output.with_extension("mp4.part");
    run_ffmpeg(&[
        "-f".as_ref(), "concat".as_ref(),
        "-safe".as_ref(), "0".as_ref(),
        "-i".as_ref(), list_file.as_os_str(),
        "-c".as_ref(), "copy".as_ref(),
        "-f".as_ref(), "mp4".as_ref(),
        "-y".as_ref(), temp_output.as_os_str(),
    ]).await?;
    std::fs::rename(&temp_output, output)?;
    Ok(())
}

/// Audio recovery status for transcript recovery feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioRecoveryStatus {
    pub status: String, // "success" | "partial" | "failed" | "none"
    pub chunk_count: u32,
    pub estimated_duration_seconds: f64,
    pub audio_file_path: Option<String>,
    pub message: String,
}

/// Recover audio from checkpoint files
/// This is called by the transcript recovery system to merge audio chunks after a crash
#[tauri::command]
pub async fn recover_audio_from_checkpoints(
    meeting_folder: String,
    _sample_rate: u32
) -> Result<AudioRecoveryStatus, String> {
    info!("Starting audio recovery for folder: {}", meeting_folder);

    let folder_path = PathBuf::from(&meeting_folder);
    let checkpoints_dir = folder_path.join(".checkpoints");

    // Check if checkpoints directory exists
    if !checkpoints_dir.exists() {
        info!("No checkpoints directory found at: {}", checkpoints_dir.display());
        return Ok(AudioRecoveryStatus {
            status: "none".to_string(),
            chunk_count: 0,
            estimated_duration_seconds: 0.0,
            audio_file_path: None,
            message: "No audio checkpoints found".to_string(),
        });
    }

    // WAV checkpoints come from this version; MP4 ones from a crash under an older one.
    let wav_files = checkpoint_files(&checkpoints_dir, "wav")
        .map_err(|e| format!("Failed to read checkpoints directory: {}", e))?;
    let (checkpoint_files, is_legacy) = if wav_files.is_empty() {
        let mp4_files = checkpoint_files(&checkpoints_dir, "mp4")
            .map_err(|e| format!("Failed to read checkpoints directory: {}", e))?;
        (mp4_files, true)
    } else {
        (wav_files, false)
    };

    if checkpoint_files.is_empty() {
        info!("No checkpoint files found in: {}", checkpoints_dir.display());
        return Ok(AudioRecoveryStatus {
            status: "none".to_string(),
            chunk_count: 0,
            estimated_duration_seconds: 0.0,
            audio_file_path: None,
            message: "No audio checkpoint files found".to_string(),
        });
    }

    let chunk_count = checkpoint_files.len() as u32;
    let estimated_duration = if is_legacy {
        (chunk_count as f64) * CHECKPOINT_SECONDS as f64
    } else {
        // 16-bit mono PCM after a 44-byte header
        checkpoint_files
            .iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .map(|meta| meta.len().saturating_sub(44) as f64 / 2.0 / 48000.0)
            .sum()
    };

    info!("Found {} checkpoint files, estimated duration: {:.2}s", chunk_count, estimated_duration);

    let output_path = folder_path.join("audio.mp4");
    let result = if is_legacy {
        concat_legacy_mp4_checkpoints(&checkpoint_files, &checkpoints_dir, &output_path).await
    } else {
        encode_wav_checkpoints(&checkpoint_files, &checkpoints_dir, &output_path).await
    };

    match result {
        Ok(()) => {
            let _ = std::fs::remove_file(checkpoints_dir.join("concat_list.txt"));
            let output_path_str = output_path.to_string_lossy().to_string();
            info!("Successfully recovered audio: {}", output_path_str);

            Ok(AudioRecoveryStatus {
                status: "success".to_string(),
                chunk_count,
                estimated_duration_seconds: estimated_duration,
                audio_file_path: Some(output_path_str),
                message: format!("Successfully recovered {} audio chunks", chunk_count),
            })
        }
        Err(e) => {
            error!("Audio recovery failed: {}", e);
            Ok(AudioRecoveryStatus {
                status: "failed".to_string(),
                chunk_count,
                estimated_duration_seconds: estimated_duration,
                audio_file_path: None,
                message: format!("Audio recovery failed: {}", e),
            })
        }
    }
}

/// Clean up checkpoint files after successful recording or recovery
/// This command is called by the frontend after successful save to clean up checkpoint files
#[tauri::command]
pub async fn cleanup_checkpoints(meeting_folder: String) -> Result<(), String> {
    info!("Cleaning up checkpoints for folder: {}", meeting_folder);

    let folder_path = PathBuf::from(&meeting_folder);
    let checkpoints_dir = folder_path.join(".checkpoints");

    if checkpoints_dir.exists() {
        std::fs::remove_dir_all(&checkpoints_dir)
            .map_err(|e| format!("Failed to remove checkpoints directory: {}", e))?;
        info!("Successfully cleaned up checkpoints directory");
    } else {
        info!("No checkpoints directory to clean up");
    }

    Ok(())
}

/// Check if a meeting folder has audio checkpoint files
/// Returns true if .checkpoints/ directory exists and contains .wav (or legacy .mp4) files
#[tauri::command]
pub async fn has_audio_checkpoints(meeting_folder: String) -> Result<bool, String> {
    let folder_path = PathBuf::from(&meeting_folder);
    let checkpoints_dir = folder_path.join(".checkpoints");

    // Check if checkpoints directory exists
    if !checkpoints_dir.exists() {
        return Ok(false);
    }

    let has_checkpoints = std::fs::read_dir(&checkpoints_dir)
        .map_err(|e| format!("Failed to read checkpoints directory: {}", e))?
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            matches!(entry.path().extension().and_then(|s| s.to_str()), Some("wav") | Some("mp4"))
        });

    Ok(has_checkpoints)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use super::super::recording_state::DeviceType;

    fn meeting_with_checkpoints_dir(name: &str) -> (tempfile::TempDir, PathBuf) {
        let temp_dir = tempdir().unwrap();
        let meeting_folder = temp_dir.path().join(name);
        std::fs::create_dir_all(meeting_folder.join(".checkpoints")).unwrap();
        (temp_dir, meeting_folder)
    }

    fn chunk(samples: usize, index: u64) -> AudioChunk {
        AudioChunk {
            data: vec![0.5f32; samples],
            sample_rate: 48000,
            timestamp: index as f64 * 0.5,  // timestamp in seconds
            chunk_id: index,
            device_type: DeviceType::Microphone,
        }
    }

    /// One AAC frame. A single encode pads the end of the file to a frame
    /// boundary (FFmpeg's decoder does not trim it), so a correct file decodes
    /// to at most this many extra samples however many checkpoints it joined.
    /// Per-checkpoint AAC + `concat -c copy` added ~1800 samples per checkpoint.
    const AAC_FRAME: f64 = 1024.0;

    /// Decode `path` with FFmpeg and return its length in 48 kHz mono samples.
    fn decoded_sample_count(path: &Path) -> usize {
        let output = std::process::Command::new(find_ffmpeg_path().expect("tests need FFmpeg"))
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(path)
            .args(["-f", "s16le", "-ac", "1", "-ar", "48000", "pipe:1"])
            .output()
            .unwrap();
        assert!(output.status.success(), "decode failed: {}", String::from_utf8_lossy(&output.stderr));
        output.stdout.len() / 2
    }

    #[tokio::test]
    async fn test_checkpoint_creation() {
        let (_temp_dir, meeting_folder) = meeting_with_checkpoints_dir("Test_Meeting");
        let mut saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000).unwrap();

        // Add 60 seconds worth of audio (should create 2 checkpoints)
        for i in 0..120 {  // 120 chunks of 0.5s each
            saver.add_chunk(chunk(24000, i)).unwrap();
        }

        // Verify 2 checkpoints created
        assert_eq!(saver.checkpoint_count, 2);
        assert!(meeting_folder.join(".checkpoints/audio_chunk_000.wav").exists());

        // Finalize and verify merge
        let final_path = saver.finalize().await.unwrap();
        assert!(final_path.exists());

        // Verify checkpoints directory deleted
        assert!(!meeting_folder.join(".checkpoints").exists());
    }

    #[tokio::test]
    async fn finalized_audio_does_not_drift_with_checkpoint_count() {
        let (_temp_dir, meeting_folder) = meeting_with_checkpoints_dir("Bob's long sync");
        let mut saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000).unwrap();

        // 5 full checkpoints plus a 7.3 s tail, in uneven 0.37 s chunks.
        let total_samples = 48000 * 157 + 14400;
        let chunk_samples = 17760;
        let mut fed = 0;
        let mut index = 0;
        while fed < total_samples {
            let n = chunk_samples.min(total_samples - fed);
            saver.add_chunk(chunk(n, index)).unwrap();
            fed += n;
            index += 1;
        }

        let final_path = saver.finalize().await.unwrap();

        let decoded = decoded_sample_count(&final_path);
        let extra = decoded as f64 - total_samples as f64;
        assert!(
            (0.0..AAC_FRAME).contains(&extra),
            "decoded {decoded} samples for {total_samples} recorded ({:.1} ms off)",
            extra / 48.0
        );
    }

    #[test]
    fn failed_checkpoint_retries_with_backoff_instead_of_on_every_chunk() {
        let (_temp_dir, meeting_folder) = meeting_with_checkpoints_dir("Disk_Full");
        let mut saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000).unwrap();
        let checkpoints_dir = meeting_folder.join(".checkpoints");
        std::fs::remove_dir(&checkpoints_dir).unwrap(); // every write now fails

        let start = Instant::now();
        let half_second = vec![0.1f32; 24000];
        let mut errors = 0;
        // 40 s of audio arriving within the first backoff window.
        for _ in 0..80 {
            if saver.add_samples_at(&half_second, start).is_err() {
                errors += 1;
            }
        }
        assert_eq!(saver.write_attempts, 1, "one failed write, then back off");
        assert_eq!(errors, 1);
        assert_eq!(saver.pending.len(), 48000 * 40, "unsaved audio stays buffered");

        // After the backoff the next chunk retries, and once the disk is back it succeeds.
        std::fs::create_dir(&checkpoints_dir).unwrap();
        saver.add_samples_at(&half_second, start + RETRY_BACKOFF_BASE).unwrap();
        assert_eq!(saver.write_attempts, 2);
        assert_eq!(saver.checkpoint_count, 1);
        assert!(saver.pending.is_empty());
    }

    #[test]
    fn unsaved_audio_is_capped_while_checkpoints_keep_failing() {
        let (_temp_dir, meeting_folder) = meeting_with_checkpoints_dir("Disk_Gone");
        let mut saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000).unwrap();
        std::fs::remove_dir(meeting_folder.join(".checkpoints")).unwrap();

        let start = Instant::now();
        let one_minute = vec![0.1f32; 48000 * 60];
        let mut last = Ok(());
        for minute in 0..12 {
            last = saver.add_samples_at(&one_minute, start + Duration::from_secs(minute * 60));
        }

        assert!(last.unwrap_err().to_string().contains("were lost"));
        assert_eq!(saver.pending.len(), 48000 * MAX_PENDING_SECONDS);
    }

    #[tokio::test]
    async fn a_failed_finalize_keeps_the_checkpoints_for_recovery() {
        let (_temp_dir, meeting_folder) = meeting_with_checkpoints_dir("Encode_Fails");
        let mut saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000).unwrap();
        for i in 0..130 {  // two checkpoints plus 5 s pending
            saver.add_chunk(chunk(24000, i)).unwrap();
        }
        std::fs::remove_file(meeting_folder.join(".checkpoints/audio_chunk_000.wav")).unwrap();

        assert!(saver.finalize().await.is_err());

        let checkpoints = meeting_folder.join(".checkpoints");
        assert!(checkpoints.join("audio_chunk_001.wav").exists());
        assert!(checkpoints.join("audio_chunk_002.wav").exists(), "the pending tail is written too");
        assert!(!meeting_folder.join("audio.mp4").exists());
    }

    #[tokio::test]
    async fn test_empty_recording() {
        let (_temp_dir, meeting_folder) = meeting_with_checkpoints_dir("Empty_Test");
        let mut saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000).unwrap();

        // Try to finalize without adding any chunks
        let result = saver.finalize().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No audio checkpoints"));
    }

    #[tokio::test]
    async fn recovery_encodes_wav_checkpoints_left_by_a_crash() {
        let (_temp_dir, meeting_folder) = meeting_with_checkpoints_dir("Crashed");
        let mut saver = IncrementalAudioSaver::new(meeting_folder.clone(), 48000).unwrap();
        for i in 0..130 {  // 65 s: two checkpoints written, 5 s still in memory when "crashing"
            saver.add_chunk(chunk(24000, i)).unwrap();
        }
        drop(saver);

        assert!(has_audio_checkpoints(meeting_folder.to_string_lossy().to_string()).await.unwrap());
        let status = recover_audio_from_checkpoints(meeting_folder.to_string_lossy().to_string(), 48000)
            .await
            .unwrap();

        assert_eq!(status.status, "success", "{}", status.message);
        assert_eq!(status.chunk_count, 2);
        assert!((status.estimated_duration_seconds - 60.0).abs() < 0.01);
        let decoded = decoded_sample_count(&meeting_folder.join("audio.mp4"));
        let extra = decoded as f64 - 48000.0 * 60.0;
        assert!((0.0..AAC_FRAME).contains(&extra), "decoded {decoded} samples");
    }
}
