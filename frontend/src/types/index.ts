export interface Message {
  id: string;
  content: string;
  timestamp: string;
}

export interface Transcript {
  id: string;
  text: string;
  timestamp: string; // Wall-clock time (e.g., "14:30:05")
  sequence_id?: number;
  chunk_start_time?: number; // Legacy field
  is_partial?: boolean;
  confidence?: number;
  // NEW: Recording-relative timestamps for playback sync
  audio_start_time?: number; // Seconds from recording start (e.g., 125.3)
  audio_end_time?: number;   // Seconds from recording start (e.g., 128.6)
  duration?: number;          // Segment duration in seconds (e.g., 3.3)
  // Speaker identification (absent until a meeting has been diarized)
  speaker_key?: string | null;     // "S1", "S2", ...
  speaker_overlap?: number | null; // Fraction of the segment covered by that speaker (0-1)
}

export interface TranscriptUpdate {
  text: string;
  timestamp: string; // Wall-clock time for reference
  source: string;
  sequence_id: number;
  chunk_start_time: number; // Legacy field
  is_partial: boolean;
  confidence: number;
  // NEW: Recording-relative timestamps for playback sync
  audio_start_time: number; // Seconds from recording start
  audio_end_time: number;   // Seconds from recording start
  duration: number;          // Segment duration in seconds
}

export interface Block {
  id: string;
  type: string;
  content: string;
  color: string;
}

export interface Section {
  title: string;
  blocks: Block[];
}

export interface Summary {
  [key: string]: Section;
}

export interface ApiResponse {
  message: string;
  num_chunks: number;
  data: any[];
}

export interface SummaryResponse {
  status: string;
  summary: Summary;
  raw_summary?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// BlockNote-specific types
export type SummaryFormat = 'legacy' | 'markdown' | 'blocknote';

export interface BlockNoteBlock {
  id: string;
  type: string;
  props?: Record<string, any>;
  content?: any[];
  children?: BlockNoteBlock[];
}

export interface SummaryDataResponse {
  markdown?: string;
  summary_json?: BlockNoteBlock[];
  reasoning_stripped?: boolean;
  normalization_fallback?: boolean;
  // Legacy format fields
  MeetingName?: string;
  _section_order?: string[];
  [key: string]: any; // For legacy section data
}

export type MeetingSummary = Summary | SummaryDataResponse;

export type SummaryProcessStatus =
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'error'
  | 'idle';

export interface ProcessTranscriptResponse {
  message: string;
  process_id: string;
}

export interface CancelSummaryResponse {
  cancelled: boolean;
  message: string;
  meeting_id: string;
}

export interface SummaryProcessResponse {
  status: SummaryProcessStatus;
  meetingName: string | null;
  meeting_id: string;
  start: string | null;
  end: string | null;
  data: unknown | null;
  error: string | null;
}

// Pagination types for optimized transcript loading
export interface MeetingMetadata {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  folder_path?: string;
}

export interface PaginatedTranscriptsResponse {
  transcripts: Transcript[];
  total_count: number;
  has_more: boolean;
}

// Transcript segment data for virtualized display
export interface TranscriptSegmentData {
  id: string;
  timestamp: number; // audio_start_time in seconds
  endTime?: number; // audio_end_time in seconds
  text: string;
  confidence?: number;
  speakerKey?: string | null;
  speakerOverlap?: number | null;
}

// Speaker identification (diarization) contracts shared with the Rust commands and events
export interface MeetingSpeaker {
  speaker_key: string;
  display_name: string | null;
  is_self: boolean;
  color_index: number;
  segment_count: number;
  talk_time_seconds: number;
  /** Voice resembles the saved "Me" voiceprint, but not closely enough to label automatically. */
  suggested_self?: boolean;
}

export type SpeakerIdJobStatus =
  | 'none'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'skipped'
  | 'interrupted';

export interface SpeakerIdStatus {
  meeting_id: string;
  status: SpeakerIdJobStatus;
  stage?: string | null;
  progress_percentage?: number | null;
  speaker_count?: number | null;
  error?: string | null;
  audio_available: boolean;
  models_installed: boolean;
  /** Length of the recording, when known. */
  audio_duration_seconds?: number | null;
  /** The saved summary was started before the latest speaker identification or speaker edit. */
  speakers_changed_since_summary?: boolean;
}

export interface SpeakerIdProgress {
  meeting_id: string;
  stage: string;
  progress_percentage: number;
  message: string;
}

export interface SpeakerIdComplete {
  meeting_id: string;
  speaker_count: number;
  labeled_segments: number;
}

export type SpeakerIdErrorCode =
  | 'no_audio'
  | 'models_missing'
  | 'decode_failed'
  | 'cancelled'
  | 'meeting_deleted'
  | 'busy'
  | 'internal';

export interface SpeakerIdError {
  meeting_id: string;
  code: SpeakerIdErrorCode;
  error: string;
}

export interface SpeakerIdOptions {
  num_speakers?: number | null;
  quality?: 'fast' | 'accurate' | null;
}

export interface StartSpeakerIdResult {
  status: 'started' | 'queued' | 'skipped';
  reason?: string | null;
}

export interface DiarizationModelsStatus {
  installed: boolean;
  missing: string[];
  total_bytes: number;
  models_dir: string;
  /** A download is running, possibly started from another screen. */
  download_in_progress: boolean;
}

export interface DiarizationModelsDownloadProgress {
  file: string;
  downloaded_bytes: number;
  total_bytes: number;
}

export interface DiarizationModelsDownloadError {
  error: string;
  cancelled?: boolean;
}
