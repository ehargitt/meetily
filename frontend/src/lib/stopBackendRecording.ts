import { appDataDir } from '@tauri-apps/api/path';
import { recordingService } from '@/services/recordingService';

/**
 * Stops the backend recording the same way the Stop button does. Only the
 * backend stop — the caller runs the post-stop save (`handleRecordingStop`).
 *
 * @returns false when the backend reports there was no recording to stop
 * @throws the backend error for any other stop failure
 */
export async function stopBackendRecording(): Promise<boolean> {
  const dataDir = await appDataDir();
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const savePath = `${dataDir}/recording-${timestamp}.wav`;
  console.log('Stopping recording, save path:', savePath);
  try {
    await recordingService.stopRecording(savePath);
    return true;
  } catch (error) {
    // String() covers Error, plain string and object rejections alike.
    if (String(error).includes('No recording in progress')) {
      return false;
    }
    throw error;
  }
}
