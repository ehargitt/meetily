import { appDataDir } from '@tauri-apps/api/path';
import { toast } from 'sonner';
import { recordingService } from '@/services/recordingService';

/**
 * - `stopped`: this call ran the backend stop; the caller runs the post-stop save.
 * - `in-progress`: another stop holds the backend stop guard; its own flow
 *   (`recording-stopped` / `recording-stop-complete`) saves — do nothing.
 * - `not-recording`: there was no recording to stop.
 */
export type BackendStopResult = 'stopped' | 'in-progress' | 'not-recording';

// Exact rejection from a stop_recording call that loses the backend stop guard.
const STOP_IN_PROGRESS = 'STOP_IN_PROGRESS';

// Shared with the recording-stop-failed event toast so one failure shows once.
export const STOP_FAILED_TOAST_ID = 'recording-stop-failed';

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * Stops the backend recording the same way the Stop button does. Only the
 * backend stop — the caller runs the post-stop save (`handleRecordingStop`).
 *
 * @throws the backend error for a stop that failed
 */
export async function stopBackendRecording(): Promise<BackendStopResult> {
  const dataDir = await appDataDir();
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const savePath = `${dataDir}/recording-${timestamp}.wav`;
  console.log('Stopping recording, save path:', savePath);
  try {
    await recordingService.stopRecording(savePath);
    return 'stopped';
  } catch (error) {
    const message = errorText(error);
    if (message === STOP_IN_PROGRESS) {
      console.log('stop_recording lost the stop guard - another stop is already running');
      return 'in-progress';
    }
    if (message.includes('No recording in progress')) {
      return 'not-recording';
    }
    throw error;
  }
}

/**
 * Reports a failed backend stop and says whether the session is still live.
 * When it is, the caller must return the UI to RECORDING instead of running
 * the post-stop flow; otherwise the caller resets the UI as before.
 */
export async function isStillRecordingAfterFailedStop(error: unknown): Promise<boolean> {
  console.error('Failed to stop recording:', error);
  toast.error('Recording could not be stopped', {
    id: STOP_FAILED_TOAST_ID,
    description: errorText(error),
    duration: 10000,
  });
  try {
    return await recordingService.isRecording();
  } catch (statusError) {
    console.error('Failed to check recording state after a failed stop:', statusError);
    return false;
  }
}
