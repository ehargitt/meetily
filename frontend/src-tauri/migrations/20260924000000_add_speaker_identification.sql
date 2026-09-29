-- Speaker identification (diarization) results.
-- Deliberately no new column on `transcripts`: upstream work claims `transcripts.speaker`
-- and `transcripts.speaker_id`. Per-segment labels live in `transcript_speakers`, which the
-- existing transcript DELETE (retranscription) clears through its foreign key cascade.

-- One row per diarized speaker of a meeting. `speaker_key` is "S<n>" and survives re-runs.
-- `suggested_self` marks the speaker whose voice resembles the stored self voiceprint but
-- not closely enough to be labelled automatically.
CREATE TABLE IF NOT EXISTS meeting_speakers (
    meeting_id TEXT NOT NULL,
    speaker_key TEXT NOT NULL,
    display_name TEXT,
    is_self INTEGER NOT NULL DEFAULT 0,
    suggested_self INTEGER NOT NULL DEFAULT 0,
    color_index INTEGER NOT NULL DEFAULT 0,
    speech_seconds REAL NOT NULL DEFAULT 0,
    embedding BLOB,
    embedding_model TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (meeting_id, speaker_key),
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);

-- Speaker turns in the audio file clock (seconds).
CREATE TABLE IF NOT EXISTS speaker_turns (
    meeting_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    speaker_key TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_speaker_turns_meeting ON speaker_turns(meeting_id, start_time);

-- The speaker assigned to each transcript segment; `overlap` is the covered fraction (0..1).
CREATE TABLE IF NOT EXISTS transcript_speakers (
    transcript_id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    speaker_key TEXT NOT NULL,
    overlap REAL NOT NULL,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcript_speakers_meeting ON transcript_speakers(meeting_id);

-- Last identification run per meeting. A 'queued'/'running' row with no live job means the
-- app quit mid-run ("interrupted").
CREATE TABLE IF NOT EXISTS speaker_identification_jobs (
    meeting_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    engine TEXT,
    speaker_count INTEGER,
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);

-- Voiceprints. At most one self profile (enforced by the partial unique index).
CREATE TABLE IF NOT EXISTS speaker_profiles (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    is_self INTEGER NOT NULL DEFAULT 0,
    embedding BLOB NOT NULL,
    embedding_model TEXT NOT NULL,
    speech_seconds REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_speaker_profiles_single_self ON speaker_profiles(is_self) WHERE is_self = 1;
