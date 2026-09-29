//! Headless end-to-end run of the identification job on a copy of the local Meetily database
//! and the Teams echo-test recording. Never touches the originals: the database files and the
//! recording folder are copied into a temp dir first. If the meeting has since been deleted
//! from the database, its row and transcripts are recreated in the copy from the recording's
//! `transcripts.json`.
//!
//! Run with:
//! `MEETILY_DIARIZATION_MODELS_DIR=<dir with both models> cargo test -p meetily --lib \
//!    diarization::e2e_tests -- --ignored --nocapture`

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use meetily_diarization::DiarizationConfig;
use serde::Deserialize;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;

use super::job::{run_identification, JobOptions};

const MEETING_ID: &str = "meeting-267f3e63-7b39-4339-9b66-11d7fc57f79f";
const RECORDING_FOLDER: &str =
    "Documents/meetily-recordings/Meeting 24_09_26_09_27_12_2026-09-24_13-27";
const DATABASE: &str = ".local/share/com.meetily.ai/meeting_minutes.sqlite";

fn home() -> PathBuf {
    dirs::home_dir().expect("home directory")
}

/// Copy the database with its WAL/SHM side files, so pages committed but not yet
/// checkpointed come along.
fn copy_database(target_dir: &Path) -> PathBuf {
    let source = home().join(DATABASE);
    let target = target_dir.join("meeting_minutes.sqlite");
    std::fs::copy(&source, &target).expect("copy database");
    for suffix in ["-wal", "-shm"] {
        let side = PathBuf::from(format!("{}{suffix}", source.display()));
        if side.exists() {
            std::fs::copy(&side, format!("{}{suffix}", target.display())).expect("copy side file");
        }
    }
    target
}

fn copy_folder(source: &Path, target: &Path) {
    std::fs::create_dir_all(target).expect("create recording copy");
    for entry in std::fs::read_dir(source).expect("read recording folder") {
        let entry = entry.expect("dir entry");
        if entry.file_type().expect("file type").is_file() {
            std::fs::copy(entry.path(), target.join(entry.file_name()))
                .expect("copy recording file");
        }
    }
}

#[derive(Deserialize)]
struct SavedTranscripts {
    segments: Vec<SavedSegment>,
}

#[derive(Deserialize)]
struct SavedSegment {
    id: String,
    text: String,
    display_time: String,
    audio_start_time: Option<f64>,
    audio_end_time: Option<f64>,
    duration: Option<f64>,
}

/// Point the meeting at the recording copy, recreating it from `transcripts.json` when the
/// database no longer has it.
async fn attach_meeting(pool: &SqlitePool, recording: &Path) {
    let folder = recording.to_string_lossy().to_string();
    let updated = sqlx::query("UPDATE meetings SET folder_path = ? WHERE id = ?")
        .bind(&folder)
        .bind(MEETING_ID)
        .execute(pool)
        .await
        .expect("point meeting at the recording copy");
    if updated.rows_affected() == 1 {
        return;
    }

    let saved: SavedTranscripts = serde_json::from_str(
        &std::fs::read_to_string(recording.join("transcripts.json"))
            .expect("read transcripts.json"),
    )
    .expect("parse transcripts.json");
    let now = chrono::Utc::now();
    sqlx::query(
        "INSERT INTO meetings (id, title, created_at, updated_at, folder_path)
         VALUES (?, 'Teams echo test', ?, ?, ?)",
    )
    .bind(MEETING_ID)
    .bind(now)
    .bind(now)
    .bind(&folder)
    .execute(pool)
    .await
    .expect("recreate meeting");
    for segment in saved.segments {
        sqlx::query(
            "INSERT INTO transcripts
                 (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time, duration)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(format!("{MEETING_ID}-{}", segment.id))
        .bind(MEETING_ID)
        .bind(segment.text)
        .bind(segment.display_time)
        .bind(segment.audio_start_time)
        .bind(segment.audio_end_time)
        .bind(segment.duration)
        .execute(pool)
        .await
        .expect("recreate transcript");
    }
}

#[tokio::test]
#[ignore = "needs MEETILY_DIARIZATION_MODELS_DIR and the local Teams echo-test meeting"]
async fn identifies_two_speakers_in_the_teams_echo_meeting() {
    let models_dir = PathBuf::from(
        std::env::var("MEETILY_DIARIZATION_MODELS_DIR").expect("MEETILY_DIARIZATION_MODELS_DIR"),
    );
    let temp = tempfile::tempdir().expect("temp dir");
    let database = copy_database(temp.path());
    let recording = temp.path().join("recording");
    copy_folder(&home().join(RECORDING_FOLDER), &recording);

    let pool = SqlitePool::connect_with(SqliteConnectOptions::new().filename(&database))
        .await
        .expect("open database copy");
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("migrate copy");
    attach_meeting(&pool, &recording).await;

    let options = JobOptions {
        config: DiarizationConfig::default(),
        workers: 8,
    };
    let saved = run_identification(
        &pool,
        MEETING_ID,
        &models_dir,
        &options,
        Arc::new(AtomicBool::new(false)),
        Arc::new(|stage, percent| println!("{stage:?} {percent}%")),
    )
    .await
    .expect("identification");
    println!("{saved:?}");

    assert_eq!(saved.speaker_count, 2, "{saved:?}");
    let unlabeled: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM transcripts t
         WHERE t.meeting_id = ? AND t.audio_start_time IS NOT NULL AND t.audio_end_time IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM transcript_speakers ts WHERE ts.transcript_id = t.id)",
    )
    .bind(MEETING_ID)
    .fetch_one(&pool)
    .await
    .expect("count unlabeled rows");
    assert_eq!(unlabeled, 0);
    let (speakers,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM meeting_speakers WHERE meeting_id = ?")
            .bind(MEETING_ID)
            .fetch_one(&pool)
            .await
            .expect("count speakers");
    assert_eq!(speakers as usize, saved.speaker_count);
    assert!(saved.labeled_segments > 0);
}
