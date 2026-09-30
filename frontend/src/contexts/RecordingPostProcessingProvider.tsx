'use client';

import React, { useEffect, useRef } from 'react';
import { listen } from '@tauri-apps/api/event';
import { toast } from 'sonner';
import { useRecordingStop } from '@/hooks/useRecordingStop';
import { isPostStopInProgress } from '@/lib/postStopFlow';
import { useRecordingState, RecordingStatus, STOP_FLOW_STATUSES } from '@/contexts/RecordingStateContext';
import { recordingService } from '@/services/recordingService';
import { stopBackendRecording, isStillRecordingAfterFailedStop } from '@/lib/stopBackendRecording';

// No-op setters: the global RecordingStateContext already tracks recording
// state; the hook only needs these for page-local state. Module-level so their
// identity is stable across renders.
const setIsRecording = () => { };
const setIsRecordingDisabled = () => { };

const RECORDING_ERROR_TOAST_ID = 'recording-error';

/**
 * RecordingPostProcessingProvider
 *
 * This provider handles post-processing when recording stops from any source:
 * - Tray menu stop
 * - Global keyboard shortcut
 * - Overlay stop button
 * - Main UI stop button
 *
 * It listens for the 'recording-stop-complete' event from Rust backend
 * and triggers the full post-processing flow (save to database, navigate, analytics)
 * regardless of which page the user is currently on.
 *
 * It also handles 'recording-error' (no audio can be captured any more): the
 * backend leaves the session open, so this runs the same stop + save as the
 * Stop button and whatever was recorded is kept.
 */
export function RecordingPostProcessingProvider({ children }: { children: React.ReactNode }) {
  const {
    handleRecordingStop,
  } = useRecordingStop(setIsRecording, setIsRecordingDisabled);
  const { status, isRecording, setStatus } = useRecordingState();

  // The listeners below register once; they read the latest values here.
  const latestRef = useRef({ handleRecordingStop, status, isRecording, setStatus });
  useEffect(() => {
    latestRef.current = { handleRecordingStop, status, isRecording, setStatus };
  });

  useEffect(() => {
    // `cancelled` guard prevents leaking a listener when StrictMode/HMR runs
    // cleanup before the async listen(...) registration resolves.
    let cancelled = false;
    let unlistenStopComplete: (() => void) | undefined;
    let unlistenRecordingError: (() => void) | undefined;

    const stopAfterRecordingError = async (message: string) => {
      const latest = latestRef.current;
      if (!latest.isRecording || STOP_FLOW_STATUSES.includes(latest.status)) {
        console.log('[RecordingPostProcessing] recording-error: no active recording or a stop is already running');
        return;
      }

      latest.setStatus(RecordingStatus.STOPPING, 'Stopping recording...');
      try {
        const result = await stopBackendRecording();
        if (result === 'in-progress') {
          // Another stop is running; its flow owns the save and the status.
          return;
        }
        if (result === 'not-recording') {
          if (!isPostStopInProgress()) latestRef.current.setStatus(RecordingStatus.IDLE);
          return;
        }
        // The stop emitted recording-stop-complete, whose listener below saves.
        toast.error(message, {
          id: RECORDING_ERROR_TOAST_ID,
          description: 'Recording stopped. Saving what was recorded so far.',
          duration: 15000,
        });
      } catch (error) {
        // Same as the Stop button: a failed backend stop skips the save.
        if (await isStillRecordingAfterFailedStop(error)) {
          latestRef.current.setStatus(RecordingStatus.RECORDING);
          return;
        }
        await latestRef.current.handleRecordingStop(false);
      }
    };

    const setupListeners = async () => {
      try {
        // Listen for recording-stop-complete event from Rust
        const fnStopComplete = await listen<boolean>('recording-stop-complete', (event) => {
          console.log('[RecordingPostProcessing] Received recording-stop-complete event:', event.payload);

          // Call the post-processing handler
          // event.payload is the callApi boolean (true for normal stops)
          latestRef.current.handleRecordingStop(event.payload);
        });
        if (cancelled) { fnStopComplete(); return; }
        unlistenStopComplete = fnStopComplete;

        const fnRecordingError = await recordingService.onRecordingError((message) => {
          console.error('[RecordingPostProcessing] recording-error →', message);
          // The description is added once the stop outcome is known.
          toast.error(message, { id: RECORDING_ERROR_TOAST_ID, duration: 15000 });
          stopAfterRecordingError(message);
        });
        if (cancelled) { fnRecordingError(); return; }
        unlistenRecordingError = fnRecordingError;

        console.log('[RecordingPostProcessing] Event listeners set up successfully');
      } catch (error) {
        console.error('[RecordingPostProcessing] Failed to set up event listeners:', error);
      }
    };

    setupListeners();

    return () => {
      console.log('[RecordingPostProcessing] Cleaning up event listeners');
      cancelled = true;
      unlistenStopComplete?.();
      unlistenRecordingError?.();
    };
  }, []);

  return <>{children}</>;
}
