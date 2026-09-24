// Every mock of RecordingStateContext must export the full module shape: bun re-links modules
// already loaded against a mocked module, so a later partial mock breaks an earlier importer of
// RecordingStatus (useRecordingStop) with "Export named 'RecordingStatus' not found".

/** Copy of the RecordingStatus enum; the real module is never loaded in tests. */
export const RecordingStatus = {
  IDLE: 'idle',
  STARTING: 'starting',
  RECORDING: 'recording',
  STOPPING: 'stopping',
  PROCESSING_TRANSCRIPTS: 'processing',
  SAVING: 'saving',
  COMPLETED: 'completed',
  ERROR: 'error',
} as const;

export function recordingStateModule(state: Record<string, unknown> = {}) {
  return {
    RecordingStatus,
    useRecordingState: () => ({ isRecording: false, ...state }),
  };
}
