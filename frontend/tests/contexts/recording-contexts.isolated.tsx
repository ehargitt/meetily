// Runs in its own bun process (see recording-contexts.test.ts): other suites mock
// RecordingStateContext and TranscriptContext for the whole run, and bun cannot
// un-mock a module, so the real providers can only be loaded where no suite has.
import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';

type BackendState = {
  is_recording: boolean; is_paused: boolean; is_active: boolean;
  recording_duration: number | null; active_duration: number | null;
};
const idle: BackendState = { is_recording: false, is_paused: false, is_active: false, recording_duration: null, active_duration: null };
const live: BackendState = { ...idle, is_recording: true, is_active: true, recording_duration: 12, active_duration: 12 };
let backendState: BackendState;
const invoke = mock(async (command: string): Promise<unknown> => {
  if (command === 'get_recording_state') return backendState;
  if (command === 'get_transcript_history') return [];
  if (command === 'get_recording_meeting_name') return null;
  throw new Error(`Unexpected command: ${command}`);
});
mock.module('@tauri-apps/api/core', () => ({ invoke }));

type EventHandler = (event: { payload: unknown }) => void;
const eventHandlers = new Map<string, Set<EventHandler>>();
mock.module('@tauri-apps/api/event', () => ({
  listen: async (name: string, handler: EventHandler) => {
    if (!eventHandlers.has(name)) eventHandlers.set(name, new Set());
    eventHandlers.get(name)!.add(handler);
    return () => { eventHandlers.get(name)?.delete(handler); };
  },
}));
async function emit(name: string, payload: unknown) {
  await act(async () => { eventHandlers.get(name)?.forEach(handler => handler({ payload })); });
}

const toastError = mock<(message: string, options?: Record<string, unknown>) => void>(() => {});
const notify = mock(() => {});
mock.module('sonner', () => ({ toast: { info: notify, error: toastError, success: notify, warning: notify } }));

const saveTranscript = mock<(meetingId: string, update: unknown) => Promise<void>>(async () => {});
mock.module('../../src/services/indexedDBService', () => ({
  indexedDBService: {
    init: async () => {}, saveTranscript, saveMeetingMetadata: async () => {},
    getMeetingMetadata: async () => null, markMeetingSaved: async () => {},
  },
}));

class MemoryStorage {
  private items = new Map<string, string>();
  getItem(key: string) { return this.items.get(key) ?? null; }
  setItem(key: string, value: string) { this.items.set(key, value); }
  removeItem(key: string) { this.items.delete(key); }
}
Object.defineProperty(globalThis, 'sessionStorage', { configurable: true, writable: true, value: new MemoryStorage() });

const { RecordingStateProvider, useRecordingState, RecordingStatus } = await import('../../src/contexts/RecordingStateContext');
const { TranscriptProvider, useTranscripts } = await import('../../src/contexts/TranscriptContext');

let recording: ReturnType<typeof useRecordingState>;
let transcripts: ReturnType<typeof useTranscripts>;
function Probe() {
  recording = useRecordingState();
  transcripts = useTranscripts();
  return null;
}

let renderer: ReactTestRenderer | undefined;
async function mount() {
  await act(async () => {
    renderer = create(<RecordingStateProvider><TranscriptProvider><Probe /></TranscriptProvider></RecordingStateProvider>);
  });
}
async function until(predicate: () => boolean, what: string, timeoutMs = 1500) {
  for (let waited = 0; !predicate(); waited += 10) {
    if (waited > timeoutMs) throw new Error(`timed out waiting for ${what}`);
    await act(() => new Promise(resolve => setTimeout(resolve, 10)));
  }
}
const stateCalls = () => invoke.mock.calls.filter(([command]) => command === 'get_recording_state').length;

beforeEach(() => {
  backendState = idle;
  invoke.mockClear(); toastError.mockClear(); saveTranscript.mockClear();
  sessionStorage.removeItem('indexeddb_current_meeting_id');
});
afterEach(async () => {
  // Unmounting stops the state polling interval.
  if (renderer) await act(async () => renderer!.unmount());
  renderer = undefined;
});

describe('mounting while the backend is already recording (webview reload)', () => {
  test('adopts the live session, resumes polling and restores the meeting id', async () => {
    backendState = live;
    sessionStorage.setItem('indexeddb_current_meeting_id', 'meeting-before-reload');
    await mount();

    await until(() => recording.status === RecordingStatus.RECORDING && recording.isRecording, 'RECORDING');
    await until(() => transcripts.currentMeetingId === 'meeting-before-reload', 'the restored meeting id');
    // Polling runs every 500 ms only while a session is live.
    await until(() => stateCalls() >= 2, 'a polled state sync');

    await emit('transcript-update', {
      text: 'after the reload', timestamp: '00:12', sequence_id: 7, chunk_start_time: 12,
      is_partial: false, confidence: 0.9, audio_start_time: 12, audio_end_time: 13, duration: 1,
    });
    expect(saveTranscript.mock.calls.map(([meetingId]) => meetingId)).toEqual(['meeting-before-reload']);
  });

  test('an idle backend stays idle and a stored id is not adopted', async () => {
    sessionStorage.setItem('indexeddb_current_meeting_id', 'meeting-left-over');
    await mount();
    await until(() => stateCalls() >= 1, 'the mount sync');
    await act(() => new Promise(resolve => setTimeout(resolve, 600)));

    expect(recording.status).toBe(RecordingStatus.IDLE);
    expect(transcripts.currentMeetingId).toBeNull();
    expect(stateCalls()).toBe(1);
  });
});

describe('recording-stopping and recording-stop-failed', () => {
  test('a stop sets STOPPING; its failure returns to RECORDING while the backend still records', async () => {
    backendState = live;
    await mount();
    await until(() => recording.status === RecordingStatus.RECORDING, 'RECORDING');

    await emit('recording-stopping', { source: 'tray' });
    expect(recording.status).toBe(RecordingStatus.STOPPING);

    await emit('recording-stop-failed', { message: 'encoder busy' });
    await until(() => recording.status === RecordingStatus.RECORDING, 'RECORDING after the failure');
    expect(toastError).toHaveBeenCalledWith('Recording could not be stopped',
      expect.objectContaining({ id: 'recording-stop-failed', description: 'encoder busy' }));
  });

  test('a failed stop that left the backend idle returns to IDLE', async () => {
    backendState = live;
    await mount();
    await until(() => recording.status === RecordingStatus.RECORDING, 'RECORDING');

    await emit('recording-stopping', { source: 'ui' });
    backendState = idle;
    await emit('recording-stop-failed', { message: 'save failed' });
    await until(() => recording.status === RecordingStatus.IDLE, 'IDLE after the failure');
    expect(recording.isRecording).toBe(false);
  });
});
