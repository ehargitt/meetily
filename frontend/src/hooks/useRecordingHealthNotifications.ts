import { MutableRefObject, useEffect } from 'react';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { toast } from 'sonner';
import { recordingService, type AudioStreamType } from '@/services/recordingService';
import { transcriptService } from '@/services/transcriptService';

// A failing engine can reject every chunk; one toast per window is enough to
// tell the user without burying the screen.
const TRANSCRIPTION_WARNING_TOAST_INTERVAL_MS = 30000;

const streamLabel = (deviceType: AudioStreamType) =>
  deviceType === 'microphone' ? 'Microphone' : 'System audio';

/**
 * Surfaces backend recording-health events as toasts on whichever page is open:
 * audio streams that degrade, recover or are unavailable, failed audio saves,
 * and transcription failures or lost chunks. Notification only — none of these
 * events stops the recording.
 *
 * @param isRecordingRef - current recording flag; stream degrade/recover
 *   toasts are dropped once the session has ended (teardown noise).
 */
export function useRecordingHealthNotifications(isRecordingRef: MutableRefObject<boolean>) {
  useEffect(() => {
    // `cancelled` guard prevents leaking a listener when StrictMode/HMR runs
    // cleanup before the async listen(...) registration resolves.
    let cancelled = false;
    const unlisteners: UnlistenFn[] = [];
    let lastWarningToastAt = 0;

    const register = async (subscription: Promise<UnlistenFn>) => {
      const unlisten = await subscription;
      if (cancelled) {
        unlisten();
        return;
      }
      unlisteners.push(unlisten);
    };

    const setup = async () => {
      try {
        await Promise.all([
          register(recordingService.onAudioStreamDegraded(({ device_type, device_name, reason }) => {
            console.warn('[RecordingHealth] audio-stream-degraded →', device_type, device_name, reason);
            if (!isRecordingRef.current) return;
            // Shared id: the matching "restored" toast replaces this one.
            toast.warning(`${streamLabel(device_type)} interrupted — reconnecting`, {
              id: `audio-stream-${device_type}`,
              description: `${device_name}: ${reason}`,
              duration: 10000,
            });
          })),
          register(recordingService.onAudioStreamRecovered(({ device_type, device_name }) => {
            console.log('[RecordingHealth] audio-stream-recovered →', device_type, device_name);
            if (!isRecordingRef.current) return;
            toast.success(`${streamLabel(device_type)} restored`, {
              id: `audio-stream-${device_type}`,
              description: device_name,
              duration: 5000,
            });
          })),
          // Fires at start as well as mid-recording, so not gated on isRecordingRef.
          register(recordingService.onSystemAudioUnavailable(({ device_name, reason }) => {
            console.warn('[RecordingHealth] system-audio-unavailable →', device_name, reason);
            toast.warning("System audio unavailable — participants' audio is not being recorded", {
              id: 'system-audio-unavailable',
              description: device_name ? `${device_name}: ${reason}` : reason,
              duration: 20000,
            });
          })),
          // Can fire while the final audio is written during stop, after
          // isRecording has already dropped, so not gated.
          register(recordingService.onRecordingSaveError(({ message }) => {
            console.error('[RecordingHealth] recording-save-error →', message);
            toast.error('Audio could not be saved to disk', {
              id: 'recording-save-error',
              description: message,
              duration: 10000,
            });
          })),
          register(transcriptService.onTranscriptChunkLossDetected(({ chunks_lost, message }) => {
            console.warn('[RecordingHealth] transcript-chunk-loss-detected →', chunks_lost, message);
            const chunks = chunks_lost === 1 ? '1 audio chunk was' : `${chunks_lost} audio chunks were`;
            toast.warning(`${chunks} not transcribed`, {
              description: message,
              duration: 10000,
            });
          })),
          register(transcriptService.onTranscriptionWarning((message) => {
            console.warn('[RecordingHealth] transcription-warning →', message);
            const now = Date.now();
            if (now - lastWarningToastAt < TRANSCRIPTION_WARNING_TOAST_INTERVAL_MS) return;
            lastWarningToastAt = now;
            toast.warning('Some audio could not be transcribed', {
              description: message,
              duration: 6000,
            });
          })),
          // Startup-phase errors are shown by the start flow (useModalState).
          register(transcriptService.onTranscriptionError(({ phase, userMessage, error }) => {
            if (phase !== 'active') return;
            console.error('[RecordingHealth] transcription-error (active) →', error);
            toast.error('Transcription stopped', {
              id: 'transcription-error-active',
              description: `${userMessage || error} Audio is still being recorded.`,
              duration: 15000,
            });
          })),
          register(transcriptService.onTranscriptError((message) => {
            console.error('[RecordingHealth] transcript-error →', message);
            toast.error('Transcription error', { description: message, duration: 8000 });
          })),
        ]);
      } catch (e) {
        console.error('[RecordingHealth] Failed to set up recording health listeners:', e);
      }
    };

    setup();

    return () => {
      cancelled = true;
      unlisteners.forEach(unlisten => unlisten());
    };
  }, [isRecordingRef]);
}
