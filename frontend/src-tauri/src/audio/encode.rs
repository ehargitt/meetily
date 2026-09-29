use super::ffmpeg::find_ffmpeg_path; // Correct path to encode module
use super::AudioDevice;
use std::io::Write;
use std::sync::Arc;
use std::{
    path::PathBuf,
    process::{Command, Stdio},
};
use tracing::{debug, error};

pub struct AudioInput {
    pub data: Arc<Vec<f32>>,
    pub sample_rate: u32,
    pub channels: u16,
    pub device: Arc<AudioDevice>,
}

pub fn encode_single_audio(
    data: &[u8],
    sample_rate: u32,
    channels: u16,
    output_path: &PathBuf,
) -> anyhow::Result<()> {
    debug!("Starting FFmpeg process for {} bytes of audio data", data.len());

    if data.is_empty() {
        return Err(anyhow::anyhow!("No audio data provided for encoding"));
    }

    let ffmpeg_path = find_ffmpeg_path().ok_or_else(|| {
        anyhow::anyhow!("FFmpeg not found. Please install FFmpeg to save recordings.")
    })?;

    debug!("Using FFmpeg at: {:?}", ffmpeg_path);

    let output_path_str = output_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Output path is not valid UTF-8: {}", output_path.display()))?;

    let mut command = Command::new(ffmpeg_path);
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "f32le",
            "-ar",
            &sample_rate.to_string(),
            "-ac",
            &channels.to_string(),
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k", // Increased from 64k for better audio quality (especially for speech)
            "-profile:a",
            "aac_low", // Use AAC-LC profile for better compatibility
            "-movflags",
            "+faststart", // Optimize for web streaming
            "-f",
            "mp4",
            output_path_str,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    // Hide console window on Windows to prevent CMD popup during recording
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    debug!("FFmpeg command: {:?}", command);

    let mut ffmpeg = command
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to spawn FFmpeg process: {}", e))?;
    debug!("FFmpeg process spawned");
    let write_result = match ffmpeg.stdin.take() {
        Some(mut stdin) => stdin.write_all(data),
        None => Err(std::io::Error::other("FFmpeg stdin was not captured")),
    };
    // stdin is closed by now; always reap the child, even if the write failed.
    debug!("Waiting for FFmpeg process to exit");
    let output = ffmpeg
        .wait_with_output()
        .map_err(|e| anyhow::anyhow!("Failed to wait for FFmpeg process: {}", e))?;
    let status = output.status;
    let stderr = String::from_utf8_lossy(&output.stderr);

    debug!("FFmpeg process exited with status: {}", status);
    debug!("FFmpeg stderr: {}", stderr);

    if let Err(e) = write_result {
        error!("Failed to write audio to FFmpeg: {}", e);
        return Err(anyhow::anyhow!("Failed to write audio to FFmpeg: {}", e));
    }

    if !status.success() {
        error!("FFmpeg process failed with status: {}", status);
        error!("FFmpeg stderr: {}", stderr);
        return Err(anyhow::anyhow!(
            "FFmpeg process failed with status: {}",
            status
        ));
    }

    Ok(())
}
