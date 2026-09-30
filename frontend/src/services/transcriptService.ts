/**
 * Transcript Service
 *
 * Handles all transcription-related Tauri backend calls and events.
 * Pure 1-to-1 wrapper - no error handling changes, exact same behavior as direct invoke/listen calls.
 */

import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { TranscriptUpdate, Transcript } from '@/types';

export interface TranscriptionStatus {
  chunks_in_queue: number;
  is_processing: boolean;
  last_activity_ms: number;
}

export interface TranscriptionErrorPayload {
  error: string;
  userMessage: string;
  actionable: boolean;
  phase: 'startup' | 'active';
}

export interface TranscriptChunkLossPayload {
  chunks_queued: number;
  chunks_completed: number;
  chunks_lost: number;
  message: string;
}

/** Sent by the live transcription worker each time it finishes a chunk. */
export interface TranscriptionProgressPayload {
  worker_id: number;
  chunks_completed: number;
  chunks_processed: number;  // transcribed, failed or skipped
  chunks_queued: number;
  progress_percentage: number;
  message: string;
}

export interface ModelDownloadCompletePayload {
  modelName: string;
}

/**
 * Transcript Service
 * Singleton service for managing transcription operations and transcript history
 */
export class TranscriptService {
  /**
   * Get transcript history from backend (for reload sync)
   * @returns Promise<Transcript[]>
   */
  async getTranscriptHistory(): Promise<Transcript[]> {
    return invoke<Transcript[]>('get_transcript_history');
  }

  /**
   * Get current transcription queue status
   * @returns Promise with transcription status
   */
  async getTranscriptionStatus(): Promise<TranscriptionStatus> {
    return invoke<TranscriptionStatus>('get_transcription_status');
  }

  // Event Listeners

  /**
   * Listen for real-time transcript updates
   * @param callback - Function to call when new transcript segment arrives
   * @returns Promise that resolves to unlisten function
   */
  async onTranscriptUpdate(callback: (update: TranscriptUpdate) => void): Promise<UnlistenFn> {
    return listen<TranscriptUpdate>('transcript-update', (event) => {
      callback(event.payload);
    });
  }

  /**
   * Listen for transcription-complete event
   * @param callback - Function to call when transcription processing is complete
   * @returns Promise that resolves to unlisten function
   */
  async onTranscriptionComplete(callback: () => void): Promise<UnlistenFn> {
    return listen('transcription-complete', callback);
  }

  /**
   * Listen for transcription-error event (structured errors)
   * @param callback - Function to call when transcription error occurs
   * @returns Promise that resolves to unlisten function
   */
  async onTranscriptionError(callback: (error: TranscriptionErrorPayload) => void): Promise<UnlistenFn> {
    return listen<TranscriptionErrorPayload>('transcription-error', (event) => {
      callback(event.payload);
    });
  }

  /**
   * Listen for transcript-error event (legacy error format)
   * @param callback - Function to call when transcript error occurs
   * @returns Promise that resolves to unlisten function
   */
  async onTranscriptError(callback: (error: string) => void): Promise<UnlistenFn> {
    return listen<string>('transcript-error', (event) => {
      callback(event.payload);
    });
  }

  /**
   * Listen for transcription-warning event (a single chunk failed to transcribe)
   * @param callback - Function to call with the warning message
   * @returns Promise that resolves to unlisten function
   */
  async onTranscriptionWarning(callback: (message: string) => void): Promise<UnlistenFn> {
    return listen<string>('transcription-warning', (event) => {
      callback(event.payload);
    });
  }

  /**
   * Listen for transcript-chunk-loss-detected event (chunks were not transcribed)
   * @param callback - Function to call with the loss counts
   * @returns Promise that resolves to unlisten function
   */
  async onTranscriptChunkLossDetected(callback: (payload: TranscriptChunkLossPayload) => void): Promise<UnlistenFn> {
    return listen<TranscriptChunkLossPayload>('transcript-chunk-loss-detected', (event) => {
      callback(event.payload);
    });
  }

  /**
   * Listen for transcription-progress event (a live chunk was handled)
   * @param callback - Function to call with the queue counts
   * @returns Promise that resolves to unlisten function
   */
  async onTranscriptionProgress(callback: (payload: TranscriptionProgressPayload) => void): Promise<UnlistenFn> {
    return listen<TranscriptionProgressPayload>('transcription-progress', (event) => {
      callback(event.payload);
    });
  }

  /**
   * Listen for Whisper model download complete event
   * @param callback - Function to call when Whisper model download completes
   * @returns Promise that resolves to unlisten function
   */
  async onModelDownloadComplete(callback: (modelName: string) => void): Promise<UnlistenFn> {
    return listen<ModelDownloadCompletePayload>('model-download-complete', (event) => {
      callback(event.payload.modelName);
    });
  }

  /**
   * Listen for Parakeet model download complete event
   * @param callback - Function to call when Parakeet model download completes
   * @returns Promise that resolves to unlisten function
   */
  async onParakeetModelDownloadComplete(callback: (modelName: string) => void): Promise<UnlistenFn> {
    return listen<ModelDownloadCompletePayload>('parakeet-model-download-complete', (event) => {
      callback(event.payload.modelName);
    });
  }
}

// Export singleton instance
export const transcriptService = new TranscriptService();
