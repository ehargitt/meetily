import { afterAll, afterEach, beforeEach, describe, expect, mock, test } from 'bun:test';
import type { ReactNode } from 'react';
import { act, create, type ReactTestInstance, type ReactTestRenderer } from 'react-test-renderer';
import type { SpeakerIdJobStatus, SummaryProcessResponse } from '../../src/types';
import { recordingStateModule } from '../support/recording-state-mock';

// The real meeting route, page content, summary panel and hooks. Only IPC, events and the leaves
// that need a browser window are stubbed.

// Bun shares module mocks between test files; restore the real modules after this suite.
// The real event module is never loaded (see speaker-identification-gate.test.tsx); its mock stays registered.
const originalCore = { ...await import('@tauri-apps/api/core') };
const originalToast = { ...await import('sonner') };
const originalNavigation = { ...await import('next/navigation') };
const originalConfig = { ...await import('../../src/contexts/ConfigContext') };
const originalAnalytics = { ...await import('../../src/lib/analytics') };
const originalPreferences = { ...await import('../../src/lib/summary-language-preferences') };
const originalBlockNote = { ...await import('../../src/components/AISummary/BlockNoteSummaryView') };
const originalGeneratorButtons = { ...await import('../../src/components/MeetingDetails/SummaryGeneratorButtonGroup') };
const originalUpdaterButtons = { ...await import('../../src/components/MeetingDetails/SummaryUpdaterButtonGroup') };
const originalTranscriptPanel = { ...await import('../../src/components/MeetingDetails/TranscriptPanel') };
const originalSplitView = { ...await import('../../src/components/MeetingDetails/MeetingDetailsSplitView') };
const originalEmptyState = { ...await import('../../src/components/EmptyStateSummary') };
const originalRecentLanguages = { ...await import('../../src/hooks/useRecentLanguages') };
afterAll(() => {
  mock.module('@tauri-apps/api/core', () => originalCore);
  mock.module('sonner', () => originalToast);
  mock.module('next/navigation', () => originalNavigation);
  mock.module('../../src/contexts/ConfigContext', () => originalConfig);
  mock.module('../../src/lib/analytics', () => originalAnalytics);
  mock.module('../../src/lib/summary-language-preferences', () => originalPreferences);
  mock.module('../../src/components/AISummary/BlockNoteSummaryView', () => originalBlockNote);
  mock.module('../../src/components/MeetingDetails/SummaryGeneratorButtonGroup', () => originalGeneratorButtons);
  mock.module('../../src/components/MeetingDetails/SummaryUpdaterButtonGroup', () => originalUpdaterButtons);
  mock.module('../../src/components/MeetingDetails/TranscriptPanel', () => originalTranscriptPanel);
  mock.module('../../src/components/MeetingDetails/MeetingDetailsSplitView', () => originalSplitView);
  mock.module('../../src/components/EmptyStateSummary', () => originalEmptyState);
  mock.module('../../src/hooks/useRecentLanguages', () => originalRecentLanguages);
  handlers.clear();
});

let params: Record<string, string>;
mock.module('next/navigation', () => ({
  usePathname: () => '/meeting-details', useRouter: () => ({ push() {} }),
  useSearchParams: () => new URLSearchParams(params),
}));
mock.module('../../src/contexts/RecordingStateContext', () => recordingStateModule());
const modelConfig = { provider: 'ollama', model: 'm', whisperModel: 'base' };
mock.module('../../src/contexts/ConfigContext', () => ({
  useConfig: () => ({ isAutoSummary: true, modelConfig, setModelConfig() {}, isModelConfigLoading: false }),
}));
const notify = mock((..._args: unknown[]) => {});
mock.module('sonner', () => ({ toast: { info: notify, error: notify, success: notify, warning: notify } }));
mock.module('../../src/lib/analytics', () => ({ default: {
  trackPageView() {}, trackBackendConnection() {}, trackButtonClick() {}, trackFeatureUsed() {},
  trackSummaryGenerationStarted: async () => {}, trackSummaryGenerationCompleted: async () => {},
  trackCustomPromptUsed: async () => {},
} }));
// Reading the summary language is one of the awaits between asking for a summary and the Rust
// call; the summary panel also reads it once on mount.
let languageGate: Promise<void> | null;
let languageReads: number;
mock.module('../../src/lib/summary-language-preferences', () => ({
  ...originalPreferences,
  readCachedDetectedSummaryLanguage: async () => null,
  detectAndCacheSummaryLanguage: async () => ({ language: 'en' }),
  readMeetingSummaryLanguage: async () => {
    languageReads += 1;
    if (languageGate) await languageGate;
    return { language: 'en', storage: 'metadata' };
  },
}));
mock.module('../../src/components/AISummary/BlockNoteSummaryView', () => ({ BlockNoteSummaryView: () => <article /> }));
mock.module('../../src/components/MeetingDetails/SummaryGeneratorButtonGroup', () => ({ SummaryGeneratorButtonGroup: () => null }));
mock.module('../../src/components/MeetingDetails/SummaryUpdaterButtonGroup', () => ({ SummaryUpdaterButtonGroup: () => null }));
mock.module('../../src/components/MeetingDetails/TranscriptPanel', () => ({ TranscriptPanel: () => null }));
mock.module('../../src/components/MeetingDetails/MeetingDetailsSplitView', () => ({
  MeetingDetailsSplitView: ({ summary }: { summary: ReactNode }) => <main>{summary}</main>,
}));
mock.module('../../src/components/EmptyStateSummary', () => ({ EmptyStateSummary: () => <section /> }));
mock.module('../../src/hooks/useRecentLanguages', () => ({ ...originalRecentLanguages, useRecentLanguages: () => ({ addRecent() {} }) }));

// The database behind the Tauri commands.
interface StoredMeeting {
  job: SpeakerIdJobStatus;
  labelled: boolean;
  speakersChangedSinceSummary: boolean;
  summary: Omit<SummaryProcessResponse, 'meeting_id'>;
}
const db: Record<string, StoredMeeting> = {};
const processes: Array<{ meetingId: string; text: string }> = [];
interface CommandArgs { meetingId?: string; text?: string }
const invoke = mock(async (command: string, args: CommandArgs = {}): Promise<unknown> => {
  const meetingId = args.meetingId ?? '';
  const meeting = db[meetingId];
  switch (command) {
    case 'api_get_meetings': return [];
    case 'api_list_templates': return [];
    case 'api_get_model_config': return { provider: 'ollama', model: 'm' };
    case 'get_ollama_models': return [{ name: 'm' }];
    case 'api_get_summary': return { meeting_id: meetingId, ...meeting.summary };
    case 'api_get_meeting_metadata':
      return { id: meetingId, title: meetingId, created_at: '2026-09-24', updated_at: '2026-09-24' };
    case 'api_get_meeting_transcripts': {
      const transcript = { id: `${meetingId}-t1`, text: 'hello', timestamp: '00:00', audio_start_time: 0, audio_end_time: 2 };
      return { transcripts: [meeting.labelled ? { ...transcript, speaker_key: 'S1' } : transcript], total_count: 1, has_more: false };
    }
    case 'get_speaker_identification_status':
      return {
        meeting_id: meetingId, status: meeting.job, speaker_count: meeting.job === 'completed' ? 1 : null,
        audio_available: true, models_installed: true, speakers_changed_since_summary: meeting.speakersChangedSinceSummary,
      };
    case 'api_get_meeting_speakers':
      return meeting.labelled
        ? [{ speaker_key: 'S1', display_name: 'Alice', is_self: false, color_index: 0, segment_count: 1, talk_time_seconds: 2, voiceprint: 'too_short' }]
        : [];
    case 'api_process_transcript':
      processes.push({ meetingId, text: args.text ?? '' });
      meeting.summary = { status: 'processing', meetingName: null, start: `attempt-${processes.length}`, end: null, data: null, error: null };
      return { process_id: `attempt-${processes.length}` };
    default: throw new Error(`Unexpected command: ${command}`);
  }
});
mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));

type Handler = (event: { payload: unknown }) => void;
const handlers = new Map<string, Set<Handler>>();
mock.module('@tauri-apps/api/event', () => ({
  listen: async (event: string, handler: Handler) => {
    if (!handlers.has(event)) handlers.set(event, new Set());
    handlers.get(event)!.add(handler);
    return () => { handlers.get(event)?.delete(handler); };
  },
}));

const { SidebarProvider } = await import('../../src/components/Sidebar/SidebarProvider');
const { default: MeetingDetails } = await import('../../src/app/meeting-details/page');

const noSummary: StoredMeeting['summary'] = { status: 'idle', meetingName: null, start: null, end: null, data: null, error: null };
const WAITING_TEXT = 'Waiting for speaker identification';
const HINT_TEXT = 'Regenerate to include speaker names';

// Keep the summary status poll from running on a real timer.
const realSetInterval = globalThis.setInterval;
const realClearInterval = globalThis.clearInterval;
let renderer: ReactTestRenderer | undefined;
beforeEach(() => {
  handlers.clear();
  processes.length = 0;
  languageGate = null;
  languageReads = 0;
  db['meeting-a'] = { job: 'running', labelled: false, speakersChangedSinceSummary: false, summary: { ...noSummary } };
  db['meeting-b'] = { job: 'none', labelled: false, speakersChangedSinceSummary: false, summary: { ...noSummary } };
  invoke.mockClear();
  mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
  globalThis.setInterval = (() => 0) as unknown as typeof setInterval;
  globalThis.clearInterval = (() => {}) as typeof clearInterval;
});
afterEach(async () => {
  if (renderer) await act(async () => renderer!.unmount());
  renderer = undefined;
  globalThis.setInterval = realSetInterval;
  globalThis.clearInterval = realClearInterval;
});

const settle = () => act(async () => { for (let i = 0; i < 10; i++) await new Promise(resolve => setTimeout(resolve, 2)); });
async function open(query: Record<string, string>) {
  params = query;
  await act(async () => {
    const view = <SidebarProvider><MeetingDetails /></SidebarProvider>;
    if (renderer) renderer.update(view); else renderer = create(view);
  });
  await settle();
}
async function emit(event: string, payload: unknown) {
  await act(async () => { handlers.get(event)?.forEach(handler => handler({ payload })); });
}
async function finishIdentification(meetingId: string) {
  db[meetingId].job = 'completed';
  db[meetingId].labelled = true;
  await emit('speaker-identification-complete', { meeting_id: meetingId, speaker_count: 1, labeled_segments: 1 });
  await settle();
}
const textOf = (node: ReactTestInstance | string): string =>
  typeof node === 'string' ? node : node.children.map(textOf).join('');
const pageText = () => textOf(renderer!.root);
const generateNowButton = () => renderer!.root.find(node => node.type === 'button' && textOf(node).includes('Generate now'));
const summaryRequests = () => processes.map(process => process.meetingId);

describe('meeting page summary with speaker identification', () => {
  test('shows the waiting pane during identification, and "Generate now" starts the summary', async () => {
    await open({ id: 'meeting-a', source: 'recording' });
    expect(pageText()).toContain(WAITING_TEXT);
    expect(processes).toHaveLength(0);

    await act(async () => { generateNowButton().props.onClick(); });
    await settle();
    expect(summaryRequests()).toEqual(['meeting-a']);
    expect(pageText()).not.toContain(WAITING_TEXT);
  });

  test('leaving right after "Generate now" still starts the summary once', async () => {
    await open({ id: 'meeting-a', source: 'recording' });
    const panelLanguageReads = languageReads;
    let releaseLanguage!: () => void;
    languageGate = new Promise<void>(resolve => { releaseLanguage = resolve; });
    await act(async () => { generateNowButton().props.onClick(); });
    await settle();
    expect(languageReads).toBe(panelLanguageReads + 1);

    await open({ id: 'meeting-b' });
    await act(async () => { releaseLanguage(); });
    await settle();
    expect(summaryRequests()).toEqual(['meeting-a']);
  });

  test('leaving right after identification starts the auto-summary still starts it once', async () => {
    await open({ id: 'meeting-a', source: 'recording' });
    const panelLanguageReads = languageReads;
    let releaseLanguage!: () => void;
    languageGate = new Promise<void>(resolve => { releaseLanguage = resolve; });
    await finishIdentification('meeting-a');
    expect(languageReads).toBe(panelLanguageReads + 1);

    await open({ id: 'meeting-b' });
    await act(async () => { releaseLanguage(); });
    await settle();
    expect(summaryRequests()).toEqual(['meeting-a']);
    expect(processes[0].text).toContain('Alice');
  });

  test('reopening a meeting whose summary predates its speaker labels offers to regenerate', async () => {
    db['meeting-a'] = {
      job: 'completed', labelled: true, speakersChangedSinceSummary: true,
      summary: { status: 'completed', meetingName: null, start: 'attempt-0', end: 'attempt-0', data: { markdown: 'Summary without names' }, error: null },
    };
    await open({ id: 'meeting-a' });
    expect(pageText()).toContain(HINT_TEXT);
  });

  test('reopening a meeting whose summary is newer than its speaker labels shows no hint', async () => {
    db['meeting-a'] = {
      job: 'completed', labelled: true, speakersChangedSinceSummary: false,
      summary: { status: 'completed', meetingName: null, start: 'attempt-0', end: 'attempt-0', data: { markdown: 'Summary with names' }, error: null },
    };
    await open({ id: 'meeting-a' });
    expect(invoke.mock.calls.some(([command]) => command === 'get_speaker_identification_status')).toBe(true);
    expect(pageText()).not.toContain(HINT_TEXT);
  });
});
