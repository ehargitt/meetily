import { afterAll, afterEach, beforeEach, describe, expect, mock, test } from 'bun:test';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import type { StartSpeakerIdResult } from '../../src/types';
import { RecordingStatus, recordingStateModule } from '../support/recording-state-mock';

// Bun shares module mocks between test files; restore the real modules other suites use.
// TranscriptContext, the recording state, the services and the event module are never loaded
// for real here (the event module would stay linked to core); their mocks stay registered.
const originalCore = { ...await import('@tauri-apps/api/core') };
const originalAnalytics = { ...await import('../../src/lib/analytics') };
const originalPreferences = { ...await import('../../src/lib/summary-language-preferences') };
const originalToast = { ...await import('sonner') };
const originalNavigation = { ...await import('next/navigation') };
afterAll(() => {
  mock.module('@tauri-apps/api/core', () => originalCore);
  mock.module('../../src/lib/analytics', () => originalAnalytics);
  mock.module('../../src/lib/summary-language-preferences', () => originalPreferences);
  mock.module('sonner', () => originalToast);
  mock.module('next/navigation', () => originalNavigation);
});

const calls: string[] = [];
const push = mock((url: string) => { calls.push(`navigate ${url}`); });
mock.module('next/navigation', () => ({ usePathname: () => '/', useRouter: () => ({ push }) }));

const setStatus = mock((status: string) => { calls.push(`status ${status}`); });
mock.module('../../src/contexts/RecordingStateContext', () => recordingStateModule({
  status: RecordingStatus.IDLE, setStatus, isStopping: false, isProcessing: false, isSaving: false,
}));

const transcript = { id: 't1', text: 'Hello there', timestamp: '00:00', audio_start_time: 0, audio_end_time: 2 };
const markMeetingAsSaved = mock(async () => { calls.push('marked saved'); });
mock.module('../../src/contexts/TranscriptContext', () => ({
  useTranscripts: () => ({
    transcriptsRef: { current: [transcript] }, flushBuffer() {}, clearTranscripts() {},
    meetingTitle: 'Standup', markMeetingAsSaved,
  }),
}));

mock.module('../../src/services/transcriptService', () => ({
  transcriptService: { getTranscriptionStatus: async () => ({ is_processing: false, chunks_in_queue: 0, last_activity_ms: 0 }) },
}));
const saveMeeting = mock(async () => { calls.push('saveMeeting'); return { meeting_id: 'meeting-new' }; });
mock.module('../../src/services/storageService', () => ({
  storageService: { saveMeeting, getMeeting: async () => ({ title: 'Standup' }) },
}));

mock.module('../../src/lib/analytics', () => ({ default: {
  trackPageView() {}, trackButtonClick() {}, trackBackendConnection() {},
  // Ends the completion-analytics block early, before it loads the real store plugin.
  getMeetingsCountToday: async () => { throw new Error('analytics are not under test'); },
} }));
mock.module('../../src/lib/summary-language-preferences', () => ({
  applyPinnedSummaryLanguageToMeeting: async () => true,
  detectAndCacheSummaryLanguage: async () => ({ language: 'en' }),
}));

const toastInfo = mock((..._args: unknown[]) => {});
const notify = mock(() => {});
mock.module('sonner', () => ({ toast: { info: toastInfo, error: notify, success: notify, warning: notify } }));

let startIdentification: () => Promise<StartSpeakerIdResult>;
const invoke = mock(async (command: string, args?: Record<string, unknown>): Promise<unknown> => {
  if (command === 'api_get_meetings') return [];
  if (command === 'start_speaker_identification') {
    calls.push(`identify ${JSON.stringify(args)}`);
    return startIdentification();
  }
  throw new Error(`Unexpected command: ${command}`);
});
mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
mock.module('@tauri-apps/api/event', () => ({ listen: async () => () => {} }));

const { SidebarProvider } = await import('../../src/components/Sidebar/SidebarProvider');
const { useRecordingStop, SPEAKER_MODELS_HINT_SHOWN_KEY } = await import('../../src/hooks/useRecordingStop');

// The hook uses browser storage and window; bun provides neither, and other suites may have
// installed their own window, so every global is put back as it was.
class MemoryStorage {
  private items = new Map<string, string>();
  getItem(key: string) { return this.items.get(key) ?? null; }
  setItem(key: string, value: string) { this.items.set(key, value); }
  removeItem(key: string) { this.items.delete(key); }
  clear() { this.items.clear(); }
}
let localStorageImpl: Pick<MemoryStorage, 'getItem' | 'setItem'>;
const browserGlobals: Record<string, PropertyDescriptor> = {
  window: { configurable: true, writable: true, value: {} },
  sessionStorage: { configurable: true, writable: true, value: new MemoryStorage() },
  localStorage: { configurable: true, get: () => localStorageImpl },
};
const previousGlobals = Object.keys(browserGlobals).map(name =>
  [name, Object.getOwnPropertyDescriptor(globalThis, name)] as const);
Object.entries(browserGlobals).forEach(([name, descriptor]) => Object.defineProperty(globalThis, name, descriptor));
afterAll(() => {
  previousGlobals.forEach(([name, descriptor]) => {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete (globalThis as Record<string, unknown>)[name];
  });
});

// The stop flow waits seconds for late transcripts and before navigating; run those waits at once.
const realSetTimeout = globalThis.setTimeout;
beforeEach(() => {
  calls.length = 0;
  invoke.mockClear(); push.mockClear(); setStatus.mockClear(); saveMeeting.mockClear();
  markMeetingAsSaved.mockClear(); toastInfo.mockClear(); notify.mockClear();
  startIdentification = async () => ({ status: 'started' });
  localStorageImpl = new MemoryStorage();
  mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
  globalThis.setTimeout = ((callback: () => void, delay?: number) =>
    realSetTimeout(callback, delay !== undefined && delay >= 500 ? 0 : delay)) as typeof setTimeout;
});
let renderer: ReactTestRenderer | undefined;
afterEach(async () => {
  if (renderer) await act(async () => renderer!.unmount());
  renderer = undefined;
  globalThis.setTimeout = realSetTimeout;
});

let stop: ReturnType<typeof useRecordingStop>;
function Recorder() {
  stop = useRecordingStop(() => {}, () => {});
  return null;
}
async function stopRecording() {
  await act(async () => { renderer = create(<SidebarProvider><Recorder /></SidebarProvider>); });
  await act(async () => { await stop.handleRecordingStop(true); });
  // Let the delayed navigation run.
  await act(() => new Promise(resolve => realSetTimeout(resolve, 10)));
}

describe('automatic speaker identification when a recording stops', () => {
  test('starts identification for the saved meeting, then opens it', async () => {
    await stopRecording();
    const identify = `identify ${JSON.stringify({ meetingId: 'meeting-new', trigger: 'auto' })}`;
    expect(calls).toContain(identify);
    expect(calls.indexOf('saveMeeting')).toBeLessThan(calls.indexOf(identify));
    expect(calls).toContain('navigate /meeting-details?id=meeting-new&source=recording');
    expect(toastInfo).not.toHaveBeenCalled();
  });

  test('a failed start does not interrupt saving and opening the meeting', async () => {
    startIdentification = async () => { throw new Error('command not found'); };
    await stopRecording();
    expect(calls).toContain('marked saved');
    expect(calls).toContain(`status ${RecordingStatus.COMPLETED}`);
    expect(calls).not.toContain(`status ${RecordingStatus.ERROR}`);
    expect(calls).toContain('navigate /meeting-details?id=meeting-new&source=recording');
  });

  test('says once where to download the models when they are missing', async () => {
    startIdentification = async () => ({ status: 'skipped', reason: 'models_missing' });
    await stopRecording();
    expect(toastInfo).toHaveBeenCalledTimes(1);
    expect(toastInfo.mock.calls[0][0]).toBe('Download speaker models in Settings to identify speakers automatically');
    expect(localStorageImpl.getItem(SPEAKER_MODELS_HINT_SHOWN_KEY)).toBe('true');

    await act(async () => renderer!.unmount());
    renderer = undefined;
    await stopRecording();
    expect(toastInfo).toHaveBeenCalledTimes(1);
  });

  test('other skip reasons and unavailable storage do not break the stop flow', async () => {
    startIdentification = async () => ({ status: 'skipped', reason: 'disabled' });
    await stopRecording();
    expect(toastInfo).not.toHaveBeenCalled();

    await act(async () => renderer!.unmount());
    renderer = undefined;
    startIdentification = async () => ({ status: 'skipped', reason: 'models_missing' });
    localStorageImpl = {
      getItem: () => { throw new Error('storage is disabled'); },
      setItem: () => { throw new Error('storage is disabled'); },
    };
    await stopRecording();
    expect(toastInfo).toHaveBeenCalledTimes(1);
    expect(calls).toContain('navigate /meeting-details?id=meeting-new&source=recording');
  });
});
