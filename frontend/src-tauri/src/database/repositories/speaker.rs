//! SQL for speaker identification: speaker turns, per-meeting speakers, per-segment labels,
//! identification job rows and the self voiceprint.
//!
//! Turns are stored in the audio *file* clock; they are mapped to the transcript clock only
//! when segments are (re)labelled, so a retranscription can re-attach labels without
//! re-running the models.

use chrono::Utc;
use meetily_diarization::assign::{assign_segments, SegmentSpan};
use meetily_diarization::relabel::{parse_speaker_key, stable_keys, PreviousTurn};
use meetily_diarization::timeline::TranscriptClock;
use meetily_diarization::voiceprint::{self, SelfMatch, MIN_ENROLL_SECS};
use meetily_diarization::{DiarizationOutput, SpeakerTurn, MODEL_ID};
use serde::Serialize;
use sqlx::{Sqlite, SqliteConnection, SqlitePool, Transaction};
use tracing::info;

/// Speaker colours cycle through this many palette entries (`color_index = (n - 1) % 8`).
const SPEAKER_PALETTE_SIZE: usize = 8;
const SELF_PROFILE_NAME: &str = "Me";

#[derive(Debug, thiserror::Error)]
pub enum SpeakerRepoError {
    /// A write hit a foreign key: the meeting was deleted while identification ran.
    #[error("meeting was deleted")]
    MeetingDeleted,
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Invalid(String),
    #[error("database error: {0}")]
    Database(sqlx::Error),
}

impl From<sqlx::Error> for SpeakerRepoError {
    fn from(error: sqlx::Error) -> Self {
        // Every speaker table references meetings(id), and transcript rows cannot disappear
        // inside our transaction, so a foreign key failure means the meeting is gone.
        if let sqlx::Error::Database(db_error) = &error {
            if db_error.is_foreign_key_violation() {
                return Self::MeetingDeleted;
            }
        }
        Self::Database(error)
    }
}

/// A diarized speaker of one meeting, as shown to the frontend.
#[derive(Debug, Clone, PartialEq, Serialize, sqlx::FromRow)]
pub struct MeetingSpeaker {
    pub speaker_key: String,
    pub display_name: Option<String>,
    pub is_self: bool,
    /// The voice resembles the stored self voiceprint, but not closely enough to label it
    /// "Me" automatically; the UI offers "Is this you?".
    pub suggested_self: bool,
    pub color_index: i64,
    pub segment_count: i64,
    pub talk_time_seconds: f64,
    pub voiceprint: VoiceprintState,
}

/// Whether marking a speaker "Me" puts its voice in the self voiceprint (the same conditions
/// `rebuild_self_voiceprint` enrolls by), so the UI can say when it does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, sqlx::Type)]
#[serde(rename_all = "snake_case")]
#[sqlx(rename_all = "snake_case")]
pub enum VoiceprintState {
    /// Enrollable; for a "Me" speaker the self voiceprint exists.
    Ready,
    /// Less than [`MIN_ENROLL_SECS`] of speech.
    TooShort,
    /// No embedding from the current model ("Forget my voice" drops the "Me" ones); running
    /// identification again restores it.
    VoiceMissing,
    /// Marked "Me" and enrollable, but no self voiceprint exists: it was forgotten and the
    /// voice came back from a later identification run.
    NotSaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// The last stored identification run of a meeting.
#[derive(Debug, Clone, PartialEq, sqlx::FromRow)]
pub struct SpeakerIdJobRow {
    pub status: String,
    pub speaker_count: Option<i64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SavedIdentification {
    pub speaker_count: usize,
    pub labeled_segments: usize,
}

const SELECT_MEETING_SPEAKERS: &str = "
    SELECT ms.speaker_key, ms.display_name, ms.is_self, ms.suggested_self, ms.color_index,
           (SELECT COUNT(*) FROM transcript_speakers ts
             WHERE ts.meeting_id = ms.meeting_id AND ts.speaker_key = ms.speaker_key) AS segment_count,
           ms.speech_seconds AS talk_time_seconds,
           CASE
               WHEN ms.speech_seconds < ? THEN 'too_short'
               WHEN ms.embedding IS NULL OR ms.embedding_model IS NOT ? THEN 'voice_missing'
               WHEN ms.is_self = 1
                    AND NOT EXISTS (SELECT 1 FROM speaker_profiles WHERE is_self = 1)
                   THEN 'not_saved'
               ELSE 'ready'
           END AS voiceprint
    FROM meeting_speakers ms
    WHERE ms.meeting_id = ?";

/// Bind the parameters of [`SELECT_MEETING_SPEAKERS`] that come before `meeting_id`.
fn bind_voiceprint_params<'q>(
    query: sqlx::query::QueryAs<'q, Sqlite, MeetingSpeaker, sqlx::sqlite::SqliteArguments<'q>>,
) -> sqlx::query::QueryAs<'q, Sqlite, MeetingSpeaker, sqlx::sqlite::SqliteArguments<'q>> {
    query.bind(MIN_ENROLL_SECS).bind(MODEL_ID)
}

pub struct SpeakerRepository;

impl SpeakerRepository {
    /// Speakers of a meeting, ordered S1, S2, …
    pub async fn get_meeting_speakers(
        pool: &SqlitePool,
        meeting_id: &str,
    ) -> Result<Vec<MeetingSpeaker>, SpeakerRepoError> {
        let query = format!(
            "{SELECT_MEETING_SPEAKERS} ORDER BY CAST(SUBSTR(ms.speaker_key, 2) AS INTEGER)"
        );
        Ok(bind_voiceprint_params(sqlx::query_as(&query))
            .bind(meeting_id)
            .fetch_all(pool)
            .await?)
    }

    /// Store a diarization result in one transaction: stable keys against the previous run,
    /// turns, speakers (keeping names and "Me" of keys that carry over), the self-voiceprint
    /// match, segment labels, and a `completed` job row.
    pub async fn save_result(
        pool: &SqlitePool,
        meeting_id: &str,
        output: &DiarizationOutput,
        clock: TranscriptClock,
    ) -> Result<SavedIdentification, SpeakerRepoError> {
        let mut tx = begin_write(pool).await?;

        let previous: Vec<PreviousTurn> = sqlx::query_as::<_, (String, f64, f64)>(
            "SELECT speaker_key, start_time, end_time FROM speaker_turns
             WHERE meeting_id = ? ORDER BY start_time",
        )
        .bind(meeting_id)
        .fetch_all(&mut *tx)
        .await?
        .into_iter()
        .map(|(speaker_key, start, end)| PreviousTurn {
            speaker_key,
            start,
            end,
        })
        .collect();

        let speaker_count = output.centroids.len();
        let keys = if speaker_count == 0 {
            Vec::new()
        } else {
            stable_keys(&output.turns, speaker_count, &previous)
        };

        replace_turns(&mut tx, meeting_id, &output.turns, &keys).await?;
        upsert_speakers(&mut tx, meeting_id, output, &keys).await?;
        apply_self_match(&mut tx, meeting_id, output, &keys).await?;
        let labeled_segments = label_transcripts(&mut tx, meeting_id, clock).await?;
        record_job(
            &mut *tx,
            meeting_id,
            JobStatus::Completed,
            Some(output.model_id),
            Some(speaker_count as i64),
            None,
        )
        .await?;

        tx.commit().await?;
        info!(
            "Saved speaker identification for {}: {} speakers, {} labelled segments",
            meeting_id, speaker_count, labeled_segments
        );
        Ok(SavedIdentification {
            speaker_count,
            labeled_segments,
        })
    }

    /// Re-attach segment labels from the stored turns, e.g. after retranscription replaced
    /// every transcript row. Does nothing when the meeting has no turns.
    pub async fn realign_meeting(
        pool: &SqlitePool,
        meeting_id: &str,
        clock: TranscriptClock,
    ) -> Result<usize, SpeakerRepoError> {
        let mut tx = begin_write(pool).await?;
        let has_turns = sqlx::query("SELECT 1 FROM speaker_turns WHERE meeting_id = ? LIMIT 1")
            .bind(meeting_id)
            .fetch_optional(&mut *tx)
            .await?
            .is_some();
        if !has_turns {
            return Ok(0);
        }
        let labeled = label_transcripts(&mut tx, meeting_id, clock).await?;
        tx.commit().await?;
        info!(
            "Realigned {} transcript segments to speakers for {}",
            labeled, meeting_id
        );
        Ok(labeled)
    }

    /// Rename a speaker (`Some("")` resets to the default label) and/or mark it as the local
    /// user. Marking "Me" clears it from the meeting's other speakers. Any change of "Me"
    /// rebuilds the self voiceprint, so un-marking or moving "Me" also takes that voice out.
    /// Marking a speaker that is already "Me" rebuilds it too, which saves a voiceprint again
    /// after "Forget my voice" once identification has restored the voice.
    pub async fn update_meeting_speaker(
        pool: &SqlitePool,
        meeting_id: &str,
        speaker_key: &str,
        display_name: Option<&str>,
        is_self: Option<bool>,
    ) -> Result<MeetingSpeaker, SpeakerRepoError> {
        let mut tx = begin_write(pool).await?;
        let current = fetch_speaker(&mut tx, meeting_id, speaker_key).await?;
        let now = Utc::now();

        if let Some(name) = display_name {
            let name = name.trim();
            sqlx::query(
                "UPDATE meeting_speakers SET display_name = ?, updated_at = ?
                 WHERE meeting_id = ? AND speaker_key = ?",
            )
            .bind((!name.is_empty()).then_some(name))
            .bind(now)
            .bind(meeting_id)
            .bind(speaker_key)
            .execute(&mut *tx)
            .await?;
        }

        match is_self {
            Some(true) if !current.is_self => {
                sqlx::query(
                    "UPDATE meeting_speakers
                     SET is_self = (speaker_key = ?), suggested_self = 0, updated_at = ?
                     WHERE meeting_id = ?",
                )
                .bind(speaker_key)
                .bind(now)
                .bind(meeting_id)
                .execute(&mut *tx)
                .await?;
                rebuild_self_voiceprint(&mut tx).await?;
            }
            Some(true) => rebuild_self_voiceprint(&mut tx).await?,
            Some(false) if current.is_self => {
                sqlx::query(
                    "UPDATE meeting_speakers SET is_self = 0, updated_at = ?
                     WHERE meeting_id = ? AND speaker_key = ?",
                )
                .bind(now)
                .bind(meeting_id)
                .bind(speaker_key)
                .execute(&mut *tx)
                .await?;
                rebuild_self_voiceprint(&mut tx).await?;
            }
            _ => {}
        }

        let updated = fetch_speaker(&mut tx, meeting_id, speaker_key).await?;
        tx.commit().await?;
        Ok(updated)
    }

    /// Fold `from_key` into `into_key`: its turns and segment labels move over, talk time is
    /// recomputed from the merged turns, and `into_key` keeps its name (adopting `from_key`'s
    /// only when it has none).
    pub async fn merge_meeting_speakers(
        pool: &SqlitePool,
        meeting_id: &str,
        from_key: &str,
        into_key: &str,
    ) -> Result<(), SpeakerRepoError> {
        if from_key == into_key {
            return Err(SpeakerRepoError::Invalid(
                "Cannot merge a speaker into itself".to_string(),
            ));
        }
        let mut tx = begin_write(pool).await?;
        let from = fetch_speaker(&mut tx, meeting_id, from_key).await?;
        let into = fetch_speaker(&mut tx, meeting_id, into_key).await?;

        for table in ["speaker_turns", "transcript_speakers"] {
            sqlx::query(&format!(
                "UPDATE {table} SET speaker_key = ? WHERE meeting_id = ? AND speaker_key = ?"
            ))
            .bind(into_key)
            .bind(meeting_id)
            .bind(from_key)
            .execute(&mut *tx)
            .await?;
        }

        let merged_turns: Vec<(f64, f64)> = sqlx::query_as(
            "SELECT start_time, end_time FROM speaker_turns
             WHERE meeting_id = ? AND speaker_key = ? ORDER BY start_time",
        )
        .bind(meeting_id)
        .bind(into_key)
        .fetch_all(&mut *tx)
        .await?;

        sqlx::query(
            "UPDATE meeting_speakers
             SET display_name = COALESCE(display_name, ?),
                 is_self = MAX(is_self, ?),
                 suggested_self = MAX(suggested_self, ?),
                 speech_seconds = ?,
                 updated_at = ?
             WHERE meeting_id = ? AND speaker_key = ?",
        )
        .bind(from.display_name)
        .bind(from.is_self)
        .bind(from.suggested_self)
        .bind(union_length(&merged_turns))
        .bind(Utc::now())
        .bind(meeting_id)
        .bind(into_key)
        .execute(&mut *tx)
        .await?;

        sqlx::query("DELETE FROM meeting_speakers WHERE meeting_id = ? AND speaker_key = ?")
            .bind(meeting_id)
            .bind(from_key)
            .execute(&mut *tx)
            .await?;
        if from.is_self || into.is_self {
            rebuild_self_voiceprint(&mut tx).await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Delete the self voiceprint, the "Is this you?" suggestions derived from it, and the
    /// voice embeddings of every speaker marked "Me" (the voiceprint is rebuilt from those).
    /// Speakers already marked "Me" keep that label.
    pub async fn delete_self_voiceprint(pool: &SqlitePool) -> Result<(), SpeakerRepoError> {
        let mut tx = pool.begin().await?;
        sqlx::query("DELETE FROM speaker_profiles WHERE is_self = 1")
            .execute(&mut *tx)
            .await?;
        sqlx::query("UPDATE meeting_speakers SET suggested_self = 0 WHERE suggested_self = 1")
            .execute(&mut *tx)
            .await?;
        sqlx::query("UPDATE meeting_speakers SET embedding = NULL WHERE is_self = 1")
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn get_job(
        pool: &SqlitePool,
        meeting_id: &str,
    ) -> Result<Option<SpeakerIdJobRow>, SpeakerRepoError> {
        Ok(sqlx::query_as::<_, SpeakerIdJobRow>(
            "SELECT status, speaker_count, error FROM speaker_identification_jobs WHERE meeting_id = ?",
        )
        .bind(meeting_id)
        .fetch_optional(pool)
        .await?)
    }

    /// Whether the meeting's completed summary predates its speaker labels: the summary started
    /// (`start_time`, else `created_at`) before the last completed identification or the last
    /// speaker edit (rename, "Me", merge). `false` without a completed summary or speakers.
    /// Timestamps are compared as `julianday` values rather than text, so rows written with a
    /// different timestamp format or fractional precision still order correctly.
    pub async fn speakers_changed_since_summary(
        pool: &SqlitePool,
        meeting_id: &str,
    ) -> Result<bool, SpeakerRepoError> {
        Ok(sqlx::query_scalar(
            "SELECT EXISTS (
                 SELECT 1 FROM summary_processes sp
                 WHERE sp.meeting_id = ?1
                   AND LOWER(sp.status) = 'completed'
                   AND EXISTS (SELECT 1 FROM meeting_speakers WHERE meeting_id = ?1)
                   AND julianday(COALESCE(sp.start_time, sp.created_at)) < MAX(
                       COALESCE((SELECT julianday(completed_at) FROM speaker_identification_jobs
                                  WHERE meeting_id = ?1 AND status = 'completed'), 0),
                       COALESCE((SELECT MAX(julianday(updated_at)) FROM meeting_speakers
                                  WHERE meeting_id = ?1), 0)))",
        )
        .bind(meeting_id)
        .fetch_one(pool)
        .await?)
    }

    /// Record a non-completed job state (`completed` is written by [`Self::save_result`]).
    pub async fn set_job_status(
        pool: &SqlitePool,
        meeting_id: &str,
        status: JobStatus,
        error: Option<&str>,
    ) -> Result<(), SpeakerRepoError> {
        record_job(pool, meeting_id, status, None, None, error).await
    }
}

/// Begin a transaction that holds the write lock from the start. These transactions read and
/// then write; a deferred one would fail with `SQLITE_BUSY` instead of waiting when another
/// connection commits between its read and its first write (WAL mode).
async fn begin_write(pool: &SqlitePool) -> Result<Transaction<'static, Sqlite>, sqlx::Error> {
    pool.begin_with("BEGIN IMMEDIATE").await
}

async fn record_job<'e, E>(
    executor: E,
    meeting_id: &str,
    status: JobStatus,
    engine: Option<&str>,
    speaker_count: Option<i64>,
    error: Option<&str>,
) -> Result<(), SpeakerRepoError>
where
    E: sqlx::Executor<'e, Database = Sqlite>,
{
    let now = Utc::now();
    sqlx::query(
        "INSERT INTO speaker_identification_jobs
             (meeting_id, status, engine, speaker_count, error, started_at, completed_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(meeting_id) DO UPDATE SET
             status = excluded.status,
             engine = COALESCE(excluded.engine, engine),
             speaker_count = excluded.speaker_count,
             error = excluded.error,
             started_at = COALESCE(excluded.started_at, started_at),
             completed_at = excluded.completed_at,
             updated_at = excluded.updated_at",
    )
    .bind(meeting_id)
    .bind(status.as_str())
    .bind(engine)
    .bind(speaker_count)
    .bind(error)
    .bind((status == JobStatus::Running).then_some(now))
    .bind(status.is_terminal().then_some(now))
    .bind(now)
    .execute(executor)
    .await?;
    Ok(())
}

async fn fetch_speaker(
    conn: &mut SqliteConnection,
    meeting_id: &str,
    speaker_key: &str,
) -> Result<MeetingSpeaker, SpeakerRepoError> {
    let query = format!("{SELECT_MEETING_SPEAKERS} AND ms.speaker_key = ?");
    bind_voiceprint_params(sqlx::query_as(&query))
        .bind(meeting_id)
        .bind(speaker_key)
        .fetch_optional(conn)
        .await?
        .ok_or_else(|| {
            SpeakerRepoError::NotFound(format!(
                "Speaker {speaker_key} not found in meeting {meeting_id}"
            ))
        })
}

async fn replace_turns(
    conn: &mut SqliteConnection,
    meeting_id: &str,
    turns: &[SpeakerTurn],
    keys: &[String],
) -> Result<(), SpeakerRepoError> {
    sqlx::query("DELETE FROM speaker_turns WHERE meeting_id = ?")
        .bind(meeting_id)
        .execute(&mut *conn)
        .await?;
    for turn in turns {
        sqlx::query(
            "INSERT INTO speaker_turns (meeting_id, start_time, end_time, speaker_key)
             VALUES (?, ?, ?, ?)",
        )
        .bind(meeting_id)
        .bind(turn.start)
        .bind(turn.end)
        .bind(&keys[turn.speaker])
        .execute(&mut *conn)
        .await?;
    }
    Ok(())
}

/// Insert or refresh one row per speaker, keeping `display_name` and `is_self` of keys that
/// carried over from the previous run, and drop speakers that no longer appear.
async fn upsert_speakers(
    conn: &mut SqliteConnection,
    meeting_id: &str,
    output: &DiarizationOutput,
    keys: &[String],
) -> Result<(), SpeakerRepoError> {
    let now = Utc::now();
    for (index, key) in keys.iter().enumerate() {
        let color_index = parse_speaker_key(key).map_or(0, |n| n % SPEAKER_PALETTE_SIZE) as i64;
        sqlx::query(
            "INSERT INTO meeting_speakers
                 (meeting_id, speaker_key, color_index, speech_seconds, embedding,
                  embedding_model, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(meeting_id, speaker_key) DO UPDATE SET
                 color_index = excluded.color_index,
                 speech_seconds = excluded.speech_seconds,
                 embedding = excluded.embedding,
                 embedding_model = excluded.embedding_model,
                 suggested_self = 0,
                 updated_at = excluded.updated_at",
        )
        .bind(meeting_id)
        .bind(key)
        .bind(color_index)
        .bind(output.speech_secs[index])
        .bind(voiceprint::to_blob(&output.centroids[index]))
        .bind(output.model_id)
        .bind(now)
        .bind(now)
        .execute(&mut *conn)
        .await?;
    }

    let existing: Vec<String> =
        sqlx::query_scalar("SELECT speaker_key FROM meeting_speakers WHERE meeting_id = ?")
            .bind(meeting_id)
            .fetch_all(&mut *conn)
            .await?;
    for stale in existing.iter().filter(|key| !keys.contains(key)) {
        sqlx::query("DELETE FROM meeting_speakers WHERE meeting_id = ? AND speaker_key = ?")
            .bind(meeting_id)
            .bind(stale)
            .execute(&mut *conn)
            .await?;
    }
    Ok(())
}

/// When no speaker of the meeting is marked "Me" yet, compare the speakers with the stored
/// self voiceprint: a close match is labelled "Me", a weaker one becomes a suggestion.
async fn apply_self_match(
    conn: &mut SqliteConnection,
    meeting_id: &str,
    output: &DiarizationOutput,
    keys: &[String],
) -> Result<(), SpeakerRepoError> {
    if keys.is_empty() {
        return Ok(());
    }
    let has_self =
        sqlx::query("SELECT 1 FROM meeting_speakers WHERE meeting_id = ? AND is_self = 1")
            .bind(meeting_id)
            .fetch_optional(&mut *conn)
            .await?
            .is_some();
    if has_self {
        return Ok(());
    }
    let profile: Option<Vec<u8>> = sqlx::query_scalar(
        "SELECT embedding FROM speaker_profiles WHERE is_self = 1 AND embedding_model = ?",
    )
    .bind(output.model_id)
    .fetch_optional(&mut *conn)
    .await?;
    let Some(profile) = profile.as_deref().and_then(voiceprint::from_blob) else {
        return Ok(());
    };

    let (column, index) = match voiceprint::match_self(&output.centroids, &profile) {
        SelfMatch::Auto(index) => ("is_self", index),
        SelfMatch::Suggest(index) => ("suggested_self", index),
        SelfMatch::None => return Ok(()),
    };
    sqlx::query(&format!(
        "UPDATE meeting_speakers SET {column} = 1 WHERE meeting_id = ? AND speaker_key = ?"
    ))
    .bind(meeting_id)
    .bind(&keys[index])
    .execute(&mut *conn)
    .await?;
    Ok(())
}

/// Replace the meeting's segment labels from its stored turns. Turns are mapped from the
/// file clock to `clock` first; segments without both times stay unlabelled.
async fn label_transcripts(
    conn: &mut SqliteConnection,
    meeting_id: &str,
    clock: TranscriptClock,
) -> Result<usize, SpeakerRepoError> {
    sqlx::query("DELETE FROM transcript_speakers WHERE meeting_id = ?")
        .bind(meeting_id)
        .execute(&mut *conn)
        .await?;

    let rows: Vec<(String, f64, f64)> = sqlx::query_as(
        "SELECT speaker_key, start_time, end_time FROM speaker_turns
         WHERE meeting_id = ? ORDER BY start_time",
    )
    .bind(meeting_id)
    .fetch_all(&mut *conn)
    .await?;
    if rows.is_empty() {
        return Ok(0);
    }

    // assign_segments works on speaker indices; index into the keys seen in the turns.
    let mut keys: Vec<String> = Vec::new();
    let turns: Vec<SpeakerTurn> = rows
        .into_iter()
        .map(|(key, start, end)| {
            let speaker = keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                keys.push(key);
                keys.len() - 1
            });
            SpeakerTurn {
                start,
                end,
                speaker,
            }
        })
        .collect();
    let turns = clock.map_turns(&turns);

    let segments: Vec<SegmentSpan> = sqlx::query_as::<_, (String, f64, f64)>(
        "SELECT id, audio_start_time, audio_end_time FROM transcripts
         WHERE meeting_id = ? AND audio_start_time IS NOT NULL AND audio_end_time IS NOT NULL",
    )
    .bind(meeting_id)
    .fetch_all(&mut *conn)
    .await?
    .into_iter()
    .map(|(transcript_id, start, end)| SegmentSpan {
        transcript_id,
        start,
        end,
    })
    .collect();

    let assignments = assign_segments(&segments, &turns);
    for assignment in &assignments {
        sqlx::query(
            "INSERT INTO transcript_speakers (transcript_id, meeting_id, speaker_key, overlap)
             VALUES (?, ?, ?, ?)",
        )
        .bind(&assignment.transcript_id)
        .bind(meeting_id)
        .bind(&keys[assignment.speaker])
        .bind(assignment.overlap)
        .execute(&mut *conn)
        .await?;
    }
    Ok(assignments.len())
}

/// Rebuild the self voiceprint from every speaker marked "Me" whose embedding comes from the
/// current model and who has at least [`MIN_ENROLL_SECS`] of speech (one profile at most).
/// Without such a speaker there is no voiceprint.
async fn rebuild_self_voiceprint(conn: &mut SqliteConnection) -> Result<(), SpeakerRepoError> {
    let rows: Vec<(Vec<u8>, f64)> = sqlx::query_as(
        "SELECT embedding, speech_seconds FROM meeting_speakers
         WHERE is_self = 1 AND embedding IS NOT NULL AND embedding_model = ?
           AND speech_seconds >= ?",
    )
    .bind(MODEL_ID)
    .bind(MIN_ENROLL_SECS)
    .fetch_all(&mut *conn)
    .await?;
    let sources: Vec<(Vec<f32>, f64)> = rows
        .iter()
        .filter_map(|(blob, secs)| voiceprint::from_blob(blob).map(|v| (v, *secs)))
        .collect();

    let Some((profile, total_secs)) = voiceprint::build_profile(&sources) else {
        sqlx::query("DELETE FROM speaker_profiles WHERE is_self = 1")
            .execute(&mut *conn)
            .await?;
        info!("No speaker marked \"Me\" can be enrolled; self voiceprint cleared");
        return Ok(());
    };

    let existing: Option<String> =
        sqlx::query_scalar("SELECT id FROM speaker_profiles WHERE is_self = 1")
            .fetch_optional(&mut *conn)
            .await?;
    let now = Utc::now();
    match existing {
        Some(id) => {
            sqlx::query(
                "UPDATE speaker_profiles
                 SET embedding = ?, embedding_model = ?, speech_seconds = ?, updated_at = ?
                 WHERE id = ?",
            )
            .bind(voiceprint::to_blob(&profile))
            .bind(MODEL_ID)
            .bind(total_secs)
            .bind(now)
            .bind(id)
            .execute(&mut *conn)
            .await?;
        }
        None => {
            sqlx::query(
                "INSERT INTO speaker_profiles
                     (id, display_name, is_self, embedding, embedding_model, speech_seconds,
                      created_at, updated_at)
                 VALUES (?, ?, 1, ?, ?, ?, ?, ?)",
            )
            .bind(uuid::Uuid::new_v4().to_string())
            .bind(SELF_PROFILE_NAME)
            .bind(voiceprint::to_blob(&profile))
            .bind(MODEL_ID)
            .bind(total_secs)
            .bind(now)
            .bind(now)
            .execute(&mut *conn)
            .await?;
        }
    }
    info!(
        "Rebuilt the self voiceprint from {} speakers ({:.1}s of speech)",
        sources.len(),
        total_secs
    );
    Ok(())
}

/// Total length covered by `(start, end)` intervals sorted by start; overlaps count once.
fn union_length(sorted: &[(f64, f64)]) -> f64 {
    let mut total = 0.0;
    let mut current: Option<(f64, f64)> = None;
    for &(start, end) in sorted {
        current = match current {
            Some((s, e)) if start <= e => Some((s, e.max(end))),
            Some((s, e)) => {
                total += e - s;
                Some((start, end))
            }
            None => Some((start, end)),
        };
    }
    total + current.map_or(0.0, |(s, e)| e - s)
}

#[cfg(test)]
#[path = "speaker_tests.rs"]
mod tests;
