import { toast } from 'sonner';
import { RecordingStatus, STOP_FLOW_STATUSES } from '@/contexts/RecordingStateContext';

// Module scope, not per hook instance: the home page and
// RecordingPostProcessingProvider each mount useRecordingStop, and a UI stop
// and a tray stop must not both run the post-stop save.
let stopInProgress = false;

/** Whether a post-stop flow (transcription wait, save, navigation) is running. */
export function isPostStopInProgress() {
  return stopInProgress;
}

/** Claims or releases the post-stop flow; only useRecordingStop calls this. */
export function setPostStopInProgress(value: boolean) {
  stopInProgress = value;
}

/**
 * Whether the previous meeting is still being stopped or saved, so a new
 * recording must not start: its start would clear the transcripts the save
 * is about to store, and the save would then clear the new meeting's IDs.
 * Once COMPLETED only analytics and the cancellable navigation remain.
 */
export function isPreviousMeetingSaving(status: RecordingStatus): boolean {
  return STOP_FLOW_STATUSES.includes(status)
    || (stopInProgress && status !== RecordingStatus.COMPLETED);
}

/** Tells the user why a start was refused by {@link isPreviousMeetingSaving}. */
export function notifyPreviousMeetingSaving() {
  toast.info('Finishing saving the previous meeting…', {
    id: 'previous-meeting-saving',
    description: 'Start the new recording once it has been saved.',
    duration: 5000,
  });
}
