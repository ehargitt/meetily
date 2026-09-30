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
const originalPath = { ...await import('@tauri-apps/api/path') };
const originalConfig = { ...await import('../../src/contexts/ConfigContext') };
const originalRecordingNotification = { ...await import('../../src/lib/recordingNotification') };
const originalTooltip = { ...await import('../../src/components/ui/tooltip') };
afterAll(() => {
  mock.module('../../src/components/ui/tooltip', () => originalTooltip);
  mock.module('@tauri-apps/api/path', () => originalPath);
  mock.module('../../src/contexts/ConfigContext', () => originalConfig);
  mock.module('../../src/lib/recordingNotification', () => originalRecordingNotification);
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
// Mutable so a test can set the state a component renders with.
const recordingState: Record<string, unknown> = {};
const resetRecordingState = () => Object.assign(recordingState, {
  status: RecordingStatus.IDLE, isRecording: false, setStatus, isStopping: false, isProcessing: false, isSaving: false,
  isPaused: false, isStartingRecording: false,
});
resetRecordingState();
mock.module('../../src/contexts/RecordingStateContext', () => recordingStateModule(recordingState));

const transcript = { id: 't1', text: 'Hello there', timestamp: '00:00', audio_start_time: 0, audio_end_time: 2 };
const markMeetingAsSaved = mock(async () => { calls.push('marked saved'); });
mock.module('../../src/contexts/TranscriptContext', () => ({
  useTranscripts: () => ({
    transcriptsRef: { current: [transcript] }, flushBuffer() {}, clearTranscripts() {},
    meetingTitle: 'Standup', markMeetingAsSaved, setMeetingTitle() {},
  }),
}));

type TranscriptionStatus = { is_processing: boolean; chunks_in_queue: number; last_activity_ms: number };
const transcriptionDone = async (): Promise<TranscriptionStatus> =>
  ({ is_processing: false, chunks_in_queue: 0, last_activity_ms: 0 });
let transcriptionStatus = transcriptionDone;
mock.module('../../src/services/transcriptService', () => ({
  transcriptService: { getTranscriptionStatus: () => transcriptionStatus() },
}));
let saveMeetingResult: () => Promise<{ meeting_id: string }>;
const saveMeeting = mock(async () => { calls.push('saveMeeting'); return saveMeetingResult(); });
mock.module('../../src/services/storageService', () => ({
  storageService: { saveMeeting, getMeeting: async () => ({ title: 'Standup' }) },
}));

mock.module('../../src/lib/analytics', () => ({ default: {
  trackPageView() {}, trackButtonClick() {}, trackBackendConnection() {},
  trackTranscriptionSuccess() {}, trackTranscriptionError() {},
  // Ends the completion-analytics block early, before it loads the real store plugin.
  getMeetingsCountToday: async () => { throw new Error('analytics are not under test'); },
} }));
mock.module('../../src/lib/summary-language-preferences', () => ({
  applyPinnedSummaryLanguageToMeeting: async () => true,
  detectAndCacheSummaryLanguage: async () => ({ language: 'en' }),
}));

const toastInfo = mock((..._args: unknown[]) => {});
type ToastCall = (message: string, options?: Record<string, unknown>) => void;
const toastWarning = mock<ToastCall>(() => {});
const toastError = mock<ToastCall>(() => {});
const notify = mock(() => {});
mock.module('sonner', () => ({ toast: { info: toastInfo, error: toastError, success: notify, warning: toastWarning } }));

let startIdentification: () => Promise<StartSpeakerIdResult>;
let backendRecording: boolean | (() => never);
let transcriptionModelReady: boolean;
let stopRecordingResult: () => Promise<void>;
const invoke = mock(async (command: string, args?: Record<string, unknown>): Promise<unknown> => {
  if (command === 'api_get_meetings') return [];
  if (command === 'is_recording') return typeof backendRecording === 'function' ? backendRecording() : backendRecording;
  if (command === 'api_get_transcript_config') return { provider: 'parakeet' };
  if (command === 'parakeet_init') return null;
  if (command === 'parakeet_has_available_models') return transcriptionModelReady;
  if (command === 'parakeet_get_available_models') return [];
  if (command === 'start_recording_with_devices_and_meeting') {
    calls.push('start_recording');
    return null;
  }
  if (command === 'stop_recording') {
    calls.push('stop_recording');
    return stopRecordingResult();
  }
  if (command === 'start_speaker_identification') {
    calls.push(`identify ${JSON.stringify(args)}`);
    return startIdentification();
  }
  throw new Error(`Unexpected command: ${command}`);
});
mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
// Handlers are kept so a test can emit backend events.
type EventHandler = (event: { payload: unknown }) => void;
const eventHandlers = new Map<string, Set<EventHandler>>();
mock.module('@tauri-apps/api/event', () => ({
  listen: async (name: string, handler: EventHandler) => {
    if (!eventHandlers.has(name)) eventHandlers.set(name, new Set());
    eventHandlers.get(name)!.add(handler);
    return () => { eventHandlers.get(name)?.delete(handler); };
  },
}));
mock.module('@tauri-apps/api/path', () => ({ ...originalPath, appDataDir: async () => '/app-data' }));
mock.module('../../src/contexts/ConfigContext', () => ({
  ...originalConfig,
  useConfig: () => ({ selectedDevices: { micDevice: null, systemDevice: null } }),
}));
mock.module('../../src/lib/recordingNotification', () => ({ showRecordingNotification: async () => {} }));
// Radix tooltips need a DOM, which bun does not have; RecordingControls renders plain children.
const Passthrough = ({ children }: { children?: React.ReactNode }) => <>{children}</>;
mock.module('../../src/components/ui/tooltip', () => ({
  ...originalTooltip, Tooltip: Passthrough, TooltipTrigger: Passthrough, TooltipProvider: Passthrough, TooltipContent: () => null,
}));
async function emit(name: string, payload: unknown) {
  await act(async () => { eventHandlers.get(name)?.forEach(handler => handler({ payload })); });
}

const { SidebarProvider } = await import('../../src/components/Sidebar/SidebarProvider');
const { useRecordingStop, SPEAKER_MODELS_HINT_SHOWN_KEY, cancelPostSaveNavigation } =
  await import('../../src/hooks/useRecordingStop');
const { RecordingPostProcessingProvider } = await import('../../src/contexts/RecordingPostProcessingProvider');
const { stopBackendRecording } = await import('../../src/lib/stopBackendRecording');
const { useRecordingStart } = await import('../../src/hooks/useRecordingStart');
const { RecordingControls } = await import('../../src/components/RecordingControls');
const { Square } = await import('lucide-react');

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
  window: { configurable: true, writable: true, value: { addEventListener() {}, removeEventListener() {} } },
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
  markMeetingAsSaved.mockClear(); toastInfo.mockClear(); toastWarning.mockClear(); toastError.mockClear();
  notify.mockClear();
  startIdentification = async () => ({ status: 'started' });
  transcriptionStatus = transcriptionDone;
  saveMeetingResult = async () => ({ meeting_id: 'meeting-new' });
  backendRecording = true;
  transcriptionModelReady = true;
  stopRecordingResult = async () => {};
  resetRecordingState();
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

/** Waits (real time) until the predicate holds; fails the test after a second. */
async function until(predicate: () => boolean, what: string) {
  for (let waited = 0; !predicate(); waited += 5) {
    if (waited > 1000) throw new Error(`timed out waiting for ${what}; calls: ${calls.join(', ')}`);
    await act(() => new Promise(resolve => realSetTimeout(resolve, 5)));
  }
}
const settle = () => act(() => new Promise(resolve => realSetTimeout(resolve, 50)));
const SAVE_UNFINISHED_WARNING = 'Saving before transcription finished';

describe('saving when the transcription wait does not finish', () => {
  test('a failed status check still saves the meeting, with a warning', async () => {
    transcriptionStatus = async () => { throw new Error('status command failed'); };
    await stopRecording();
    expect(calls).toContain('saveMeeting');
    expect(calls).toContain(`status ${RecordingStatus.COMPLETED}`);
    expect(toastWarning.mock.calls.map(call => call[0])).toContain(SAVE_UNFINISHED_WARNING);
  });

  test('a wait that times out with chunks still queued still saves, with a warning', async () => {
    transcriptionStatus = async () => ({ is_processing: true, chunks_in_queue: 3, last_activity_ms: 0 });
    await stopRecording();
    expect(calls).toContain('saveMeeting');
    expect(toastWarning.mock.calls.map(call => call[0])).toContain(SAVE_UNFINISHED_WARNING);
  });

  test('a finished transcription saves without the warning', async () => {
    await stopRecording();
    expect(calls).toContain('saveMeeting');
    expect(toastWarning.mock.calls.map(call => call[0])).not.toContain(SAVE_UNFINISHED_WARNING);
  });
});

describe('one post-stop save at a time across hook instances', () => {
  let first: ReturnType<typeof useRecordingStop>;
  let second: ReturnType<typeof useRecordingStop>;
  function FirstRecorder() { first = useRecordingStop(() => {}, () => {}); return null; }
  function SecondRecorder() { second = useRecordingStop(() => {}, () => {}); return null; }
  async function renderTwoRecorders() {
    await act(async () => {
      renderer = create(<SidebarProvider><FirstRecorder /><SecondRecorder /></SidebarProvider>);
    });
  }

  test('concurrent stops from two instances save once', async () => {
    await renderTwoRecorders();
    await act(async () => {
      await Promise.all([first.handleRecordingStop(true), second.handleRecordingStop(true)]);
    });
    expect(saveMeeting).toHaveBeenCalledTimes(1);
  });

  test('the guard is released after a stop fails, so the next stop saves', async () => {
    saveMeetingResult = async () => { throw new Error('database is locked'); };
    await renderTwoRecorders();
    await act(async () => { await first.handleRecordingStop(true); });
    expect(calls).toContain(`status ${RecordingStatus.ERROR}`);

    saveMeetingResult = async () => ({ meeting_id: 'meeting-new' });
    await act(async () => { await second.handleRecordingStop(true); });
    expect(saveMeeting).toHaveBeenCalledTimes(2);
    expect(calls).toContain(`status ${RecordingStatus.COMPLETED}`);
  });
});

describe('cancelling the navigation after a save', () => {
  // Keep the post-save delay (2 s) short but real, so there is time to cancel it.
  beforeEach(() => {
    globalThis.setTimeout = ((callback: () => void, delay?: number) =>
      realSetTimeout(callback, delay === 2000 ? 30 : delay !== undefined && delay >= 500 ? 0 : delay)) as typeof setTimeout;
  });
  async function stopWithoutWaiting() {
    await act(async () => { renderer = create(<SidebarProvider><Recorder /></SidebarProvider>); });
    await act(async () => { await stop.handleRecordingStop(true); });
  }

  test('without a cancel the saved meeting opens and status returns to idle', async () => {
    await stopWithoutWaiting();
    await settle();
    expect(calls).toContain('navigate /meeting-details?id=meeting-new&source=recording');
    expect(calls).toContain(`status ${RecordingStatus.IDLE}`);
  });

  test('a new recording cancels the pending navigation and idle reset', async () => {
    await stopWithoutWaiting();
    expect(calls).toContain(`status ${RecordingStatus.COMPLETED}`);
    cancelPostSaveNavigation();
    await settle();
    expect(push).not.toHaveBeenCalled();
    expect(calls).not.toContain(`status ${RecordingStatus.IDLE}`);
  });
});

describe('recording-error runs the Stop button flow', () => {
  async function renderProvider(state: Record<string, unknown>) {
    Object.assign(recordingState, state);
    await act(async () => {
      renderer = create(<SidebarProvider><RecordingPostProcessingProvider>{null}</RecordingPostProcessingProvider></SidebarProvider>);
    });
  }
  const liveRecording = { isRecording: true, status: RecordingStatus.RECORDING };

  test('stops the backend, then saves what was recorded', async () => {
    await renderProvider(liveRecording);
    await emit('recording-error', 'No audio can be captured');
    await until(() => calls.includes('saveMeeting'), 'the save');
    expect(calls.indexOf('stop_recording')).toBeLessThan(calls.indexOf('saveMeeting'));
    expect(toastError.mock.calls.at(-1)).toEqual(['No audio can be captured', expect.objectContaining({
      description: 'Recording stopped. Saving what was recorded so far.',
    })]);
  });

  test('does nothing more while another stop is already running', async () => {
    await renderProvider({ isRecording: true, status: RecordingStatus.STOPPING });
    await emit('recording-error', 'No audio can be captured');
    await settle();
    expect(calls).not.toContain('stop_recording');
    expect(saveMeeting).not.toHaveBeenCalled();
  });

  test('a stop that loses the backend stop guard leaves the save and status to the winner', async () => {
    stopRecordingResult = async () => { throw 'STOP_IN_PROGRESS'; };
    await renderProvider(liveRecording);
    await emit('recording-error', 'No audio can be captured');
    await until(() => calls.includes('stop_recording'), 'the stop');
    await settle();
    expect(saveMeeting).not.toHaveBeenCalled();
    expect(calls.filter(call => call.startsWith('status '))).toEqual([`status ${RecordingStatus.STOPPING}`]);
    // Only the immediate error toast; no "saving" description and no stop-failed toast.
    expect(toastError).toHaveBeenCalledTimes(1);
    expect(toastError.mock.calls[0][1]).not.toHaveProperty('description');
  });

  test('a backend that is not recording is not stopped or saved', async () => {
    backendRecording = false;
    await renderProvider(liveRecording);
    await emit('recording-error', 'No audio can be captured');
    await until(() => calls.includes(`status ${RecordingStatus.IDLE}`), 'the idle reset');
    expect(calls).not.toContain('stop_recording');
    expect(saveMeeting).not.toHaveBeenCalled();
  });

  test('a failed stop with the backend still recording returns to recording without saving', async () => {
    stopRecordingResult = async () => { throw 'Failed to stop recording: encoder busy'; };
    await renderProvider(liveRecording);
    await emit('recording-error', 'No audio can be captured');
    await until(() => calls.includes(`status ${RecordingStatus.RECORDING}`), 'the recording status');
    expect(saveMeeting).not.toHaveBeenCalled();
    expect(toastError.mock.calls.map(call => call[0])).toContain('Recording could not be stopped');
  });
});

describe('stopBackendRecording results', () => {
  test('a completed backend stop is "stopped", with a save path under the app data dir', async () => {
    expect(await stopBackendRecording()).toBe('stopped');
    const stopCall = invoke.mock.calls.find(([command]) => command === 'stop_recording');
    expect((stopCall?.[1] as { args: { save_path: string } }).args.save_path).toStartWith('/app-data/recording-');
  });

  test('losing the backend stop guard is "in-progress", as a string or an Error', async () => {
    stopRecordingResult = async () => { throw 'STOP_IN_PROGRESS'; };
    expect(await stopBackendRecording()).toBe('in-progress');
    stopRecordingResult = async () => { throw new Error('STOP_IN_PROGRESS'); };
    expect(await stopBackendRecording()).toBe('in-progress');
  });

  test('an idle backend is "not-recording" and is not sent a stop', async () => {
    backendRecording = false;
    expect(await stopBackendRecording()).toBe('not-recording');
    expect(calls).not.toContain('stop_recording');
  });

  test('a failed is_recording check falls through to the real stop', async () => {
    backendRecording = () => { throw new Error('state command failed'); };
    expect(await stopBackendRecording()).toBe('stopped');
    expect(calls).toContain('stop_recording');
  });

  test('any other stop failure is rethrown', async () => {
    stopRecordingResult = async () => { throw 'Failed to stop recording: encoder busy'; };
    await expect(stopBackendRecording()).rejects.toBe('Failed to stop recording: encoder busy');
  });
});

describe('a new start and the pending post-save navigation', () => {
  // As above: keep the 2 s post-save delay short but real.
  beforeEach(() => {
    globalThis.setTimeout = ((callback: () => void, delay?: number) =>
      realSetTimeout(callback, delay === 2000 ? 30 : delay !== undefined && delay >= 500 ? 0 : delay)) as typeof setTimeout;
  });
  let start: ReturnType<typeof useRecordingStart>;
  function Starter() {
    start = useRecordingStart(false, () => {}, () => {});
    return null;
  }
  async function saveThenStart() {
    await act(async () => { renderer = create(<SidebarProvider><Recorder /><Starter /></SidebarProvider>); });
    await act(async () => { await stop.handleRecordingStop(true); });
    await act(async () => { await start.handleRecordingStart(); });
    await settle();
  }

  test('a start refused for a missing model keeps the navigation to the saved meeting', async () => {
    transcriptionModelReady = false;
    await saveThenStart();
    expect(calls).not.toContain('start_recording');
    expect(calls).toContain('navigate /meeting-details?id=meeting-new&source=recording');
  });

  test('a start that passes the model check cancels it', async () => {
    await saveThenStart();
    expect(calls).toContain('start_recording');
    expect(push).not.toHaveBeenCalled();
  });
});

describe('the Stop button', () => {
  const onRecordingStop = mock<(callApi?: boolean) => void>(() => {});
  async function clickStop() {
    Object.assign(recordingState, { isRecording: true, status: RecordingStatus.RECORDING });
    await act(async () => {
      renderer = create(
        <RecordingControls
          isRecording barHeights={['4px']} onRecordingStop={onRecordingStop} onRecordingStart={() => {}}
          onTranscriptReceived={() => {}} isRecordingDisabled={false} isParentProcessing={false}
        />,
      );
    });
    const stopButton = renderer!.root.find(node => node.type === 'button' && node.findAllByType(Square).length > 0);
    await act(async () => { stopButton.props.onClick(); });
  }
  beforeEach(() => onRecordingStop.mockClear());

  test('an idle backend is not stopped or saved, and the status returns to idle', async () => {
    backendRecording = false;
    await clickStop();
    await until(() => calls.includes(`status ${RecordingStatus.IDLE}`), 'the idle reset');
    expect(calls).not.toContain('stop_recording');
    expect(onRecordingStop).not.toHaveBeenCalled();
  });

  test('a live backend is stopped and the post-stop save runs', async () => {
    await clickStop();
    await until(() => onRecordingStop.mock.calls.length > 0, 'the post-stop call');
    expect(calls).toContain('stop_recording');
    expect(onRecordingStop).toHaveBeenCalledWith(true);
    expect(calls).not.toContain(`status ${RecordingStatus.IDLE}`);
  });
});
