use super::*;
use crate::database::repositories::meeting::MeetingsRepository;
use meetily_diarization::{EMBEDDING_DIM, MODEL_ID};

const MEETING: &str = "meeting-1";

async fn test_pool() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    pool
}

async fn seed_meeting(pool: &SqlitePool, meeting_id: &str) {
    sqlx::query(
        "INSERT INTO meetings (id, title, created_at, updated_at) VALUES (?, 'Test', ?, ?)",
    )
    .bind(meeting_id)
    .bind(Utc::now())
    .bind(Utc::now())
    .execute(pool)
    .await
    .unwrap();
}

async fn seed_transcript(pool: &SqlitePool, id: &str, times: Option<(f64, f64)>) {
    sqlx::query(
        "INSERT INTO transcripts (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time)
         VALUES (?, ?, 'text', '00:00', ?, ?)",
    )
    .bind(id)
    .bind(MEETING)
    .bind(times.map(|t| t.0))
    .bind(times.map(|t| t.1))
    .execute(pool)
    .await
    .unwrap();
}

async fn seed_speaker(pool: &SqlitePool, key: &str, name: Option<&str>, speech_seconds: f64) {
    sqlx::query(
        "INSERT INTO meeting_speakers
             (meeting_id, speaker_key, display_name, speech_seconds, embedding, embedding_model,
              created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(MEETING)
    .bind(key)
    .bind(name)
    .bind(speech_seconds)
    .bind(voiceprint::to_blob(&unit_vector(0)))
    .bind(MODEL_ID)
    .bind(Utc::now())
    .bind(Utc::now())
    .execute(pool)
    .await
    .unwrap();
}

async fn seed_turn(pool: &SqlitePool, key: &str, start: f64, end: f64) {
    sqlx::query(
        "INSERT INTO speaker_turns (meeting_id, start_time, end_time, speaker_key) VALUES (?, ?, ?, ?)",
    )
    .bind(MEETING)
    .bind(start)
    .bind(end)
    .bind(key)
    .execute(pool)
    .await
    .unwrap();
}

async fn seed_label(pool: &SqlitePool, transcript_id: &str, key: &str) {
    sqlx::query(
        "INSERT INTO transcript_speakers (transcript_id, meeting_id, speaker_key, overlap)
         VALUES (?, ?, ?, 1.0)",
    )
    .bind(transcript_id)
    .bind(MEETING)
    .bind(key)
    .execute(pool)
    .await
    .unwrap();
}

async fn seed_self_profile(
    pool: &SqlitePool,
    id: &str,
    embedding: &[f32],
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO speaker_profiles
             (id, display_name, is_self, embedding, embedding_model, speech_seconds, created_at, updated_at)
         VALUES (?, 'Me', 1, ?, ?, 30.0, ?, ?)",
    )
    .bind(id)
    .bind(voiceprint::to_blob(embedding))
    .bind(MODEL_ID)
    .bind(Utc::now())
    .bind(Utc::now())
    .execute(pool)
    .await
    .map(|_| ())
}

async fn count(pool: &SqlitePool, table: &str) -> i64 {
    sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn label_of(pool: &SqlitePool, transcript_id: &str) -> Option<String> {
    sqlx::query_scalar("SELECT speaker_key FROM transcript_speakers WHERE transcript_id = ?")
        .bind(transcript_id)
        .fetch_optional(pool)
        .await
        .unwrap()
}

async fn speaker(pool: &SqlitePool, key: &str) -> MeetingSpeaker {
    SpeakerRepository::get_meeting_speakers(pool, MEETING)
        .await
        .unwrap()
        .into_iter()
        .find(|s| s.speaker_key == key)
        .unwrap_or_else(|| panic!("speaker {key} missing"))
}

fn unit_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0; EMBEDDING_DIM];
    v[axis] = 1.0;
    v
}

fn turn(start: f64, end: f64, speaker: usize) -> SpeakerTurn {
    SpeakerTurn {
        start,
        end,
        speaker,
    }
}

fn output(turns: Vec<SpeakerTurn>, speakers: usize) -> DiarizationOutput {
    DiarizationOutput {
        turns,
        centroids: (0..speakers).map(unit_vector).collect(),
        speech_secs: vec![20.0; speakers],
        model_id: MODEL_ID,
    }
}

/// A meeting with an earlier result: S1 (0-10 s) and S2 "Bob" (10-20 s), rows t1/t2 labelled.
async fn seed_previous_result(pool: &SqlitePool) {
    seed_meeting(pool, MEETING).await;
    seed_transcript(pool, "t1", Some((1.0, 4.0))).await;
    seed_transcript(pool, "t2", Some((12.0, 18.0))).await;
    seed_speaker(pool, "S1", None, 10.0).await;
    seed_speaker(pool, "S2", Some("Bob"), 10.0).await;
    seed_turn(pool, "S1", 0.0, 10.0).await;
    seed_turn(pool, "S2", 10.0, 20.0).await;
    seed_label(pool, "t1", "S1").await;
    seed_label(pool, "t2", "S2").await;
}

#[tokio::test]
async fn save_without_speakers_clears_the_previous_result_and_completes_the_job() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;

    let saved =
        SpeakerRepository::save_result(&pool, MEETING, &output(vec![], 0), TranscriptClock::File)
            .await
            .unwrap();

    assert_eq!(
        saved,
        SavedIdentification {
            speaker_count: 0,
            labeled_segments: 0
        }
    );
    assert_eq!(count(&pool, "speaker_turns").await, 0);
    assert_eq!(count(&pool, "meeting_speakers").await, 0);
    assert_eq!(count(&pool, "transcript_speakers").await, 0);
    let job = SpeakerRepository::get_job(&pool, MEETING)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(job.status, "completed");
    assert_eq!(job.speaker_count, Some(0));
}

#[tokio::test]
async fn save_for_a_deleted_meeting_reports_meeting_deleted() {
    let pool = test_pool().await;

    let result =
        SpeakerRepository::save_result(&pool, "gone", &output(vec![], 0), TranscriptClock::File)
            .await;

    assert!(
        matches!(result, Err(SpeakerRepoError::MeetingDeleted)),
        "{result:?}"
    );
    assert_eq!(count(&pool, "speaker_identification_jobs").await, 0);
}

#[tokio::test]
async fn retranscription_delete_drops_labels_but_keeps_turns_and_names() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;

    sqlx::query("DELETE FROM transcripts WHERE meeting_id = ?")
        .bind(MEETING)
        .execute(&pool)
        .await
        .unwrap();

    assert_eq!(count(&pool, "transcript_speakers").await, 0);
    assert_eq!(count(&pool, "speaker_turns").await, 2);
    assert_eq!(
        speaker(&pool, "S2").await.display_name.as_deref(),
        Some("Bob")
    );
}

#[tokio::test]
async fn realign_without_turns_does_nothing() {
    let pool = test_pool().await;
    seed_meeting(&pool, MEETING).await;
    seed_transcript(&pool, "t1", Some((1.0, 4.0))).await;

    let labeled = SpeakerRepository::realign_meeting(&pool, MEETING, TranscriptClock::File)
        .await
        .unwrap();

    assert_eq!(labeled, 0);
    assert_eq!(count(&pool, "transcript_speakers").await, 0);
}

#[tokio::test]
async fn deleting_the_meeting_cascades_to_every_speaker_table() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;
    SpeakerRepository::set_job_status(&pool, MEETING, JobStatus::Running, None)
        .await
        .unwrap();

    assert!(MeetingsRepository::delete_meeting(&pool, MEETING)
        .await
        .unwrap());

    for table in [
        "meeting_speakers",
        "speaker_turns",
        "transcript_speakers",
        "speaker_identification_jobs",
    ] {
        assert_eq!(count(&pool, table).await, 0, "{table}");
    }
}

#[tokio::test]
async fn paginated_transcripts_carry_speaker_labels() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;
    seed_transcript(&pool, "t3", None).await;

    let (rows, total) =
        MeetingsRepository::get_meeting_transcripts_paginated(&pool, MEETING, 10, 0)
            .await
            .unwrap();

    assert_eq!(total, 3);
    let by_id = |id: &str| rows.iter().find(|r| r.id == id).unwrap();
    assert_eq!(by_id("t1").speaker_key.as_deref(), Some("S1"));
    assert_eq!(by_id("t2").speaker_key.as_deref(), Some("S2"));
    assert_eq!(by_id("t2").speaker_overlap, Some(1.0));
    assert_eq!(by_id("t3").speaker_key, None);
}

#[tokio::test]
async fn rename_trims_and_empty_name_resets_to_default() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;

    let renamed =
        SpeakerRepository::update_meeting_speaker(&pool, MEETING, "S1", Some("  Alice "), None)
            .await
            .unwrap();
    assert_eq!(renamed.display_name.as_deref(), Some("Alice"));
    assert_eq!(renamed.segment_count, 1);

    let unchanged = SpeakerRepository::update_meeting_speaker(&pool, MEETING, "S1", None, None)
        .await
        .unwrap();
    assert_eq!(unchanged.display_name.as_deref(), Some("Alice"));

    let reset = SpeakerRepository::update_meeting_speaker(&pool, MEETING, "S1", Some(""), None)
        .await
        .unwrap();
    assert_eq!(reset.display_name, None);
}

#[tokio::test]
async fn marking_me_moves_between_speakers_and_short_speech_is_not_enrolled() {
    let pool = test_pool().await;
    seed_meeting(&pool, MEETING).await;
    seed_speaker(&pool, "S1", None, 5.0).await;
    seed_speaker(&pool, "S2", None, 5.0).await;

    SpeakerRepository::update_meeting_speaker(&pool, MEETING, "S1", None, Some(true))
        .await
        .unwrap();
    SpeakerRepository::update_meeting_speaker(&pool, MEETING, "S2", None, Some(true))
        .await
        .unwrap();
    assert!(!speaker(&pool, "S1").await.is_self);
    assert!(speaker(&pool, "S2").await.is_self);
    assert_eq!(count(&pool, "speaker_profiles").await, 0);

    SpeakerRepository::update_meeting_speaker(&pool, MEETING, "S2", None, Some(false))
        .await
        .unwrap();
    assert!(!speaker(&pool, "S2").await.is_self);
}

#[tokio::test]
async fn updating_an_unknown_speaker_is_not_found() {
    let pool = test_pool().await;
    seed_meeting(&pool, MEETING).await;

    let result =
        SpeakerRepository::update_meeting_speaker(&pool, MEETING, "S9", Some("X"), None).await;

    assert!(
        matches!(result, Err(SpeakerRepoError::NotFound(_))),
        "{result:?}"
    );
}

#[tokio::test]
async fn merge_moves_turns_and_labels_and_keeps_the_target_name() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;
    seed_speaker(&pool, "S3", Some("Carol"), 7.0).await;
    seed_turn(&pool, "S3", 15.0, 22.0).await;
    seed_turn(&pool, "S3", 30.0, 35.0).await;
    seed_transcript(&pool, "t3", Some((31.0, 34.0))).await;
    seed_label(&pool, "t3", "S3").await;

    SpeakerRepository::merge_meeting_speakers(&pool, MEETING, "S3", "S2")
        .await
        .unwrap();

    let merged = speaker(&pool, "S2").await;
    assert_eq!(merged.display_name.as_deref(), Some("Bob"));
    assert_eq!(merged.segment_count, 2);
    // S2 10-20 and S3 15-22 overlap: 10-22 plus 30-35.
    assert!(
        (merged.talk_time_seconds - 17.0).abs() < 1e-9,
        "{}",
        merged.talk_time_seconds
    );
    assert_eq!(label_of(&pool, "t3").await.as_deref(), Some("S2"));
    let remaining: Vec<String> = SpeakerRepository::get_meeting_speakers(&pool, MEETING)
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.speaker_key)
        .collect();
    assert_eq!(remaining, ["S1", "S2"]);
    let s3_turns: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM speaker_turns WHERE speaker_key = 'S3'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(s3_turns, 0);
}

#[tokio::test]
async fn merge_into_an_unnamed_speaker_adopts_the_merged_name() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;

    SpeakerRepository::merge_meeting_speakers(&pool, MEETING, "S2", "S1")
        .await
        .unwrap();

    assert_eq!(
        speaker(&pool, "S1").await.display_name.as_deref(),
        Some("Bob")
    );
}

#[tokio::test]
async fn merging_a_speaker_into_itself_is_rejected() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;

    let result = SpeakerRepository::merge_meeting_speakers(&pool, MEETING, "S1", "S1").await;

    assert!(
        matches!(result, Err(SpeakerRepoError::Invalid(_))),
        "{result:?}"
    );
}

#[tokio::test]
async fn only_one_self_profile_can_exist() {
    let pool = test_pool().await;

    seed_self_profile(&pool, "p1", &unit_vector(0))
        .await
        .unwrap();
    let second = seed_self_profile(&pool, "p2", &unit_vector(1)).await;

    assert!(second.is_err());
    assert_eq!(count(&pool, "speaker_profiles").await, 1);
}

#[tokio::test]
async fn forgetting_the_voiceprint_removes_profile_and_suggestions() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;
    seed_self_profile(&pool, "p1", &unit_vector(0))
        .await
        .unwrap();
    sqlx::query("UPDATE meeting_speakers SET suggested_self = 1 WHERE speaker_key = 'S1'")
        .execute(&pool)
        .await
        .unwrap();

    SpeakerRepository::delete_self_voiceprint(&pool)
        .await
        .unwrap();

    assert_eq!(count(&pool, "speaker_profiles").await, 0);
    assert!(!speaker(&pool, "S1").await.suggested_self);
}

#[tokio::test]
async fn job_status_transitions_are_recorded() {
    let pool = test_pool().await;
    seed_meeting(&pool, MEETING).await;

    SpeakerRepository::set_job_status(&pool, MEETING, JobStatus::Queued, None)
        .await
        .unwrap();
    assert_eq!(
        SpeakerRepository::get_job(&pool, MEETING)
            .await
            .unwrap()
            .unwrap()
            .status,
        "queued"
    );

    SpeakerRepository::set_job_status(&pool, MEETING, JobStatus::Failed, Some("boom"))
        .await
        .unwrap();
    let job = SpeakerRepository::get_job(&pool, MEETING)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(job.status, "failed");
    assert_eq!(job.error.as_deref(), Some("boom"));

    let missing = SpeakerRepository::set_job_status(&pool, "gone", JobStatus::Queued, None).await;
    assert!(
        matches!(missing, Err(SpeakerRepoError::MeetingDeleted)),
        "{missing:?}"
    );
}

#[test]
fn union_length_counts_overlap_once() {
    assert_eq!(union_length(&[]), 0.0);
    assert_eq!(
        union_length(&[(0.0, 10.0), (5.0, 12.0), (20.0, 25.0)]),
        17.0
    );
    assert_eq!(union_length(&[(0.0, 10.0), (2.0, 3.0)]), 10.0);
}

// The tests below call the engine's relabel/assign/timeline/voiceprint functions.

#[tokio::test]
#[ignore = "needs engine (track A)"]
async fn save_writes_turns_speakers_and_labels() {
    let pool = test_pool().await;
    seed_meeting(&pool, MEETING).await;
    seed_transcript(&pool, "t1", Some((1.0, 4.0))).await;
    seed_transcript(&pool, "t2", Some((12.0, 18.0))).await;
    seed_transcript(&pool, "t3", None).await;

    let saved = SpeakerRepository::save_result(
        &pool,
        MEETING,
        &output(vec![turn(0.0, 10.0, 0), turn(10.0, 20.0, 1)], 2),
        TranscriptClock::File,
    )
    .await
    .unwrap();

    assert_eq!(
        saved,
        SavedIdentification {
            speaker_count: 2,
            labeled_segments: 2
        }
    );
    assert_eq!(label_of(&pool, "t1").await.as_deref(), Some("S1"));
    assert_eq!(label_of(&pool, "t2").await.as_deref(), Some("S2"));
    assert_eq!(label_of(&pool, "t3").await, None);
    let s2 = speaker(&pool, "S2").await;
    assert_eq!(s2.color_index, 1);
    assert_eq!(s2.talk_time_seconds, 20.0);
    assert_eq!(count(&pool, "speaker_turns").await, 2);
}

#[tokio::test]
#[ignore = "needs engine (track A)"]
async fn rerun_keeps_names_through_stable_keys() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;

    // The new run numbers the speakers the other way round.
    SpeakerRepository::save_result(
        &pool,
        MEETING,
        &output(vec![turn(0.0, 10.0, 1), turn(10.0, 20.0, 0)], 2),
        TranscriptClock::File,
    )
    .await
    .unwrap();

    assert_eq!(
        speaker(&pool, "S2").await.display_name.as_deref(),
        Some("Bob")
    );
    assert_eq!(label_of(&pool, "t2").await.as_deref(), Some("S2"));
    assert_eq!(label_of(&pool, "t1").await.as_deref(), Some("S1"));
}

#[tokio::test]
#[ignore = "needs engine (track A)"]
async fn realign_labels_the_rows_written_by_retranscription() {
    let pool = test_pool().await;
    seed_previous_result(&pool).await;
    sqlx::query("DELETE FROM transcripts WHERE meeting_id = ?")
        .bind(MEETING)
        .execute(&pool)
        .await
        .unwrap();
    seed_transcript(&pool, "new-1", Some((2.0, 5.0))).await;
    seed_transcript(&pool, "new-2", Some((14.0, 16.0))).await;

    let labeled = SpeakerRepository::realign_meeting(&pool, MEETING, TranscriptClock::File)
        .await
        .unwrap();

    assert_eq!(labeled, 2);
    assert_eq!(label_of(&pool, "new-1").await.as_deref(), Some("S1"));
    assert_eq!(label_of(&pool, "new-2").await.as_deref(), Some("S2"));
}

#[tokio::test]
#[ignore = "needs engine (track A)"]
async fn save_labels_me_from_the_stored_voiceprint() {
    let pool = test_pool().await;
    seed_meeting(&pool, MEETING).await;
    seed_self_profile(&pool, "p1", &unit_vector(1))
        .await
        .unwrap();

    SpeakerRepository::save_result(
        &pool,
        MEETING,
        &output(vec![turn(0.0, 10.0, 0), turn(10.0, 20.0, 1)], 2),
        TranscriptClock::File,
    )
    .await
    .unwrap();

    assert!(!speaker(&pool, "S1").await.is_self);
    assert!(speaker(&pool, "S2").await.is_self);
}

#[tokio::test]
#[ignore = "needs engine (track A)"]
async fn marking_me_with_enough_speech_enrolls_the_voiceprint() {
    let pool = test_pool().await;
    seed_meeting(&pool, MEETING).await;
    seed_speaker(&pool, "S1", None, MIN_ENROLL_SECS + 5.0).await;

    SpeakerRepository::update_meeting_speaker(&pool, MEETING, "S1", None, Some(true))
        .await
        .unwrap();

    let (embedding, secs): (Vec<u8>, f64) =
        sqlx::query_as("SELECT embedding, speech_seconds FROM speaker_profiles WHERE is_self = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(voiceprint::from_blob(&embedding).unwrap(), unit_vector(0));
    assert_eq!(secs, MIN_ENROLL_SECS + 5.0);
}
