use crate::api::{TranscriptSearchResult, TranscriptSegment};
use chrono::Utc;
use sqlx::{Connection, Error as SqlxError, SqlitePool};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{error, info};
use uuid::Uuid;

static MEETINGS_SAVED: AtomicU64 = AtomicU64::new(0);

/// Meetings saved with `save_transcript` since launch. Tray Quit watches it to
/// know when the frontend has stored the recording it just stopped.
pub fn saved_meeting_count() -> u64 {
    MEETINGS_SAVED.load(Ordering::SeqCst)
}

pub struct TranscriptsRepository;

impl TranscriptsRepository {
    /// Saves a new meeting and its associated transcript segments.
    /// This function uses a transaction to ensure that either both the meeting
    /// and all its transcripts are saved, or none of them are.
    pub async fn save_transcript(
        pool: &SqlitePool,
        meeting_title: &str,
        transcripts: &[TranscriptSegment],
        folder_path: Option<String>,
    ) -> Result<String, SqlxError> {
        let meeting_id = format!("meeting-{}", Uuid::new_v4());

        let mut conn = pool.acquire().await?;
        let mut transaction = conn.begin().await?;

        let now = Utc::now();

        // 1. Create the new meeting
        let result = sqlx::query(
            "INSERT INTO meetings (id, title, created_at, updated_at, folder_path) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&meeting_id)
        .bind(meeting_title)
        .bind(now)
        .bind(now)
        .bind(&folder_path)
        .execute(&mut *transaction)
        .await;

        if let Err(e) = result {
            error!("Failed to create meeting '{}': {}", meeting_title, e);
            transaction.rollback().await?;
            return Err(e);
        }

        info!("Successfully created meeting with id: {}", meeting_id);

        // 2. Save each transcript segment with audio timing fields
        for segment in transcripts {
            let transcript_id = format!("transcript-{}", Uuid::new_v4());
            let result = sqlx::query(
                "INSERT INTO transcripts (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time, duration)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&transcript_id)
            .bind(&meeting_id)
            .bind(&segment.text)
            .bind(&segment.timestamp)
            .bind(segment.audio_start_time)
            .bind(segment.audio_end_time)
            .bind(segment.duration)
            .execute(&mut *transaction)
            .await;

            if let Err(e) = result {
                error!(
                    "Failed to save transcript segment for meeting {}: {}",
                    meeting_id, e
                );
                transaction.rollback().await?;
                return Err(e);
            }
        }

        info!(
            "Successfully saved {} transcript segments for meeting {}",
            transcripts.len(),
            meeting_id
        );

        // Commit the transaction
        transaction.commit().await?;
        MEETINGS_SAVED.fetch_add(1, Ordering::SeqCst);

        Ok(meeting_id)
    }

    /// Searches for a query string within the transcripts.
    /// It returns a list of matching transcripts with context.
    pub async fn search_transcripts(
        pool: &SqlitePool,
        query: &str,
    ) -> Result<Vec<TranscriptSearchResult>, SqlxError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let search_query = format!("%{}%", query.to_lowercase());

        let rows = sqlx::query_as::<_, (String, String, String, String)>(
            "SELECT m.id, m.title, t.transcript, t.timestamp
             FROM meetings m
             JOIN transcripts t ON m.id = t.meeting_id
             WHERE LOWER(t.transcript) LIKE ?",
        )
        .bind(&search_query)
        .fetch_all(pool)
        .await?;

        let results = rows
            .into_iter()
            .map(|(id, title, transcript, timestamp)| {
                let match_context = Self::get_match_context(&transcript, query);
                TranscriptSearchResult {
                    id,
                    title,
                    match_context,
                    timestamp,
                }
            })
            .collect();

        Ok(results)
    }

    /// Helper function to extract a snippet of text around the first match of a query.
    ///
    /// Works in chars, not bytes: lowercasing can change a char's byte length
    /// (and even its char count), so offsets found in a lowercased copy are
    /// mapped back to the original text through a per-char index.
    fn get_match_context(transcript: &str, query: &str) -> String {
        const CONTEXT_CHARS: usize = 100;

        let chars: Vec<char> = transcript.chars().collect();
        let Some((match_start, match_end)) = find_case_insensitive(&chars, query) else {
            return chars.iter().take(200).collect(); // Fallback to the start of the transcript
        };

        let start = match_start.saturating_sub(CONTEXT_CHARS);
        let end = (match_end + CONTEXT_CHARS).min(chars.len());

        let mut context = String::new();
        if start > 0 {
            context.push_str("...");
        }
        context.extend(&chars[start..end]);
        if end < chars.len() {
            context.push_str("...");
        }
        context
    }
}

/// Char range `[start, end)` in `chars` of the first case-insensitive match of `query`.
fn find_case_insensitive(chars: &[char], query: &str) -> Option<(usize, usize)> {
    let needle: Vec<char> = query.chars().flat_map(char::to_lowercase).collect();
    if needle.is_empty() {
        return None;
    }

    // Lowercased haystack plus, for each lowercased char, the original char it came from.
    let mut haystack = Vec::with_capacity(chars.len());
    let mut origin = Vec::with_capacity(chars.len());
    for (index, c) in chars.iter().enumerate() {
        for lower in c.to_lowercase() {
            haystack.push(lower);
            origin.push(index);
        }
    }

    let first = haystack.windows(needle.len()).position(|window| window == needle.as_slice())?;
    let last = first + needle.len() - 1;
    Some((origin[first], origin[last] + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_context_handles_multibyte_text_around_the_match() {
        let prefix = "é".repeat(150);
        let suffix = "日本語".repeat(60);
        let transcript = format!("{prefix}Budget Review{suffix}");

        let context = TranscriptsRepository::get_match_context(&transcript, "budget review");

        assert!(context.starts_with("..."));
        assert!(context.ends_with("..."));
        assert!(context.contains("Budget Review"));
        assert_eq!(context.chars().filter(|&c| c == 'é').count(), 100);
    }

    #[test]
    fn match_context_survives_case_folding_that_changes_byte_length() {
        // 'İ' lowercases to two chars ("i̇"), so lowercased byte offsets differ from the original.
        let transcript = format!("{}İstanbul meeting notes", "İ".repeat(120));

        let context = TranscriptsRepository::get_match_context(&transcript, "meeting");

        assert!(context.contains("meeting notes"));
        assert!(context.starts_with("..."));
    }

    #[test]
    fn match_context_falls_back_to_the_first_200_chars_without_a_match() {
        let transcript = "ü".repeat(300);

        let context = TranscriptsRepository::get_match_context(&transcript, "absent");

        assert_eq!(context.chars().count(), 200);
    }
}
