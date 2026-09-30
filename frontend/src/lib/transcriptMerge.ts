import type { Transcript, TranscriptUpdate } from '@/types';
import type { StoredTranscript } from '@/services/indexedDBService';

/**
 * A recovery record as indexedDBService.saveTranscript writes it: the live
 * TranscriptUpdate spread into the record, so its order key is `sequence_id`
 * (StoredTranscript's declared `sequenceId` is never set).
 */
export type StoredLiveSegment = StoredTranscript & Partial<TranscriptUpdate>;

/** Display order: recording time, then sequence_id for segments that start together. */
export function compareTranscripts(a: Transcript, b: Transcript): number {
  const chunkTimeDiff = (a.chunk_start_time || 0) - (b.chunk_start_time || 0);
  if (chunkTimeDiff !== 0) return chunkTimeDiff;
  return (a.sequence_id || 0) - (b.sequence_id || 0);
}

/**
 * Adds the segments of `extra` whose sequence_id `kept` lacks, in display order.
 * Each source can hold segments the other missed, so neither replaces the other.
 */
export function mergeTranscripts(kept: Transcript[], extra: Transcript[]): Transcript[] {
  const keptSequenceIds = new Set(kept.map(t => t.sequence_id));
  const missing = extra.filter(t => !keptSequenceIds.has(t.sequence_id));
  return [...kept, ...missing].sort(compareTranscripts);
}

/** Converts a recovery record back to the transcript shape the save takes. */
export function storedToTranscript(stored: StoredLiveSegment): Transcript {
  return {
    id: `stored-${stored.id ?? stored.sequence_id}`,
    text: stored.text,
    timestamp: stored.timestamp,
    sequence_id: stored.sequence_id,
    chunk_start_time: stored.chunk_start_time ?? stored.audio_start_time,
    is_partial: stored.is_partial ?? false,
    confidence: stored.confidence,
    audio_start_time: stored.audio_start_time,
    audio_end_time: stored.audio_end_time,
    duration: stored.duration,
  };
}
