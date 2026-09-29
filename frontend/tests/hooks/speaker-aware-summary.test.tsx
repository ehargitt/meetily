import { afterAll, afterEach, beforeEach, describe, expect, mock, test } from 'bun:test';
import { useState } from 'react';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import type { MeetingSpeaker, MeetingSummary, SpeakerIdJobStatus, SpeakerIdStatus, Transcript, VoiceprintState } from '../../src/types';

// Bun shares module mocks between test files; restore the real modules after this suite.
// The real event module is never loaded (see speaker-identification-gate.test.tsx); its mock stays registered.
const originalCore = { ...await import('@tauri-apps/api/core') };
const originalToast = { ...await import('sonner') };
afterAll(() => {
  mock.module('@tauri-apps/api/core', () => originalCore);
  mock.module('sonner', () => originalToast);
  handlers.clear();
});

const toastInfo = mock((..._args: unknown[]) => {});
const toastError = mock((..._args: unknown[]) => {});
const notify = mock(() => {});
mock.module('sonner', () => ({ toast: { info: toastInfo, error: toastError, success: notify, warning: notify } }));

// The database behind the Tauri commands.
let jobStatus: SpeakerIdJobStatus;
let speakerCount: number | null;
let labeled: boolean;
let speakers: MeetingSpeaker[];
let failSpeakerReads: boolean;
let speakersChangedSinceSummary: boolean;
const statusReads: string[] = [];
const invoke = mock(async (command: string, args?: Record<string, unknown>): Promise<unknown> => {
  if (command === 'get_speaker_identification_status') {
    statusReads.push(jobStatus);
    const status: SpeakerIdStatus = {
      meeting_id: 'meeting-a', status: jobStatus, speaker_count: speakerCount,
      audio_available: true, models_installed: true, speakers_changed_since_summary: speakersChangedSinceSummary,
    };
    return status;
  }
  if (command === 'api_get_meeting_speakers') {
    if (failSpeakerReads) throw new Error('database is locked');
    return speakers;
  }
  if (command === 'api_update_meeting_speaker') return speakers.find(s => s.speaker_key === args?.speakerKey);
  throw new Error(`Unexpected command: ${command}`);
});
mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));

type Handler = (event: { payload: unknown }) => void;
const handlers = new Map<string, Set<Handler>>();
// Registration resolves only when the test says so, to model the IPC round trip of listen().
let pendingRegistrations: Array<() => void> = [];
let holdRegistrations = false;
const listen = mock((event: string, handler: Handler) => {
  const registerNow = () => {
    if (!handlers.has(event)) handlers.set(event, new Set());
    handlers.get(event)!.add(handler);
    return () => { handlers.get(event)?.delete(handler); };
  };
  if (!holdRegistrations) return Promise.resolve(registerNow());
  return new Promise<() => void>(resolve => { pendingRegistrations.push(() => resolve(registerNow())); });
});
mock.module('@tauri-apps/api/event', () => ({ listen }));

const { useSpeakerAwareSummary } = await import('../../src/hooks/meeting-details/useSpeakerAwareSummary');
const { MISSING_VOICE_MESSAGE, SHORT_VOICEPRINT_MESSAGE } = await import('../../src/hooks/meeting-details/useMeetingSpeakers');

type SummaryStatus = Parameters<typeof useSpeakerAwareSummary>[0]['summaryStatus'];

const row = (id: string, speakerKey: string | null): Transcript => ({
  id, text: `line ${id}`, timestamp: '00:00', audio_start_time: 0, speaker_key: speakerKey,
});
const readRows = () => [row('1', labeled ? 'S1' : null), row('2', labeled ? 'S2' : null)];
const speaker = (key: string, talkTime: number, voiceprint: VoiceprintState = talkTime < 10 ? 'too_short' : 'ready'): MeetingSpeaker => ({
  speaker_key: key, display_name: null, is_self: false, color_index: 0, segment_count: 3, talk_time_seconds: talkTime, voiceprint,
});

const generate = mock(async (..._args: unknown[]) => {});
let refetches = 0;
let summaryStatus: SummaryStatus;
let aiSummary: MeetingSummary | null;
let state: ReturnType<typeof useSpeakerAwareSummary>;

// Stands in for page.tsx and page-content: rows come from the "database" and reload on refetch.
function MeetingPage({ autoGenerate }: { autoGenerate: boolean }) {
  const [rows, setRows] = useState<Transcript[]>(readRows);
  const [shouldAutoGenerate, setShouldAutoGenerate] = useState(autoGenerate);
  state = useSpeakerAwareSummary({
    meetingId: 'meeting-a',
    shouldAutoGenerate,
    isModelConfigLoading: false,
    transcripts: rows,
    aiSummary,
    summaryStatus,
    generateSummary: generate,
    onAutoGenerateStarted: () => setShouldAutoGenerate(false),
    onRefetchTranscripts: async () => {
      refetches += 1;
      setRows(readRows());
    },
  });
  return <output>{rows.map(transcript => transcript.speaker_key ?? '-').join(',')}</output>;
}

let renderer: ReactTestRenderer | undefined;
beforeEach(() => {
  handlers.clear();
  pendingRegistrations = [];
  holdRegistrations = false;
  statusReads.length = 0;
  jobStatus = 'running';
  speakerCount = null;
  labeled = false;
  speakers = [];
  failSpeakerReads = false;
  speakersChangedSinceSummary = false;
  refetches = 0;
  summaryStatus = 'idle';
  aiSummary = null;
  invoke.mockClear(); listen.mockClear(); generate.mockClear();
  toastInfo.mockClear(); toastError.mockClear(); notify.mockClear();
  mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
  mock.module('@tauri-apps/api/event', () => ({ listen }));
});
afterEach(async () => {
  if (renderer) await act(async () => renderer!.unmount());
  renderer = undefined;
});

async function show(autoGenerate = true) {
  await act(async () => {
    const view = <MeetingPage autoGenerate={autoGenerate} />;
    if (renderer) renderer.update(view); else renderer = create(view);
  });
}
const rerender = () => act(async () => { renderer!.update(<MeetingPage autoGenerate />); });
const labels = () => (renderer!.toJSON() as { children: string[] }).children.join('');
async function emit(event: string, payload: unknown) {
  await act(async () => { handlers.get(event)?.forEach(handler => handler({ payload })); });
}
// The job finishes: its rows and speakers are saved, then the completion event goes out.
async function finishJob() {
  jobStatus = 'completed';
  speakerCount = 2;
  labeled = true;
  speakers = [speaker('S1', 30), speaker('S2', 20)];
  await emit('speaker-identification-complete', { meeting_id: 'meeting-a', speaker_count: 2, labeled_segments: 2 });
}

describe('meeting page wiring for speaker identification', () => {
  test('holds the auto-summary while identification runs, then labels rows and summarizes once', async () => {
    await show();
    expect(state.isWaitingForSpeakers).toBe(true);
    expect(generate).not.toHaveBeenCalled();

    await finishJob();
    expect(labels()).toBe('S1,S2');
    expect(state.meetingSpeakers.speakers.map(s => s.speaker_key)).toEqual(['S1', 'S2']);
    expect(state.isWaitingForSpeakers).toBe(false);
    expect(generate).toHaveBeenCalledTimes(1);
    expect(generate.mock.calls[0][0]).toBe('');
  });

  test('a completion emitted before the listeners exist still labels rows and releases the summary', async () => {
    holdRegistrations = true;
    await show();
    // The job finishes while listen() is still registering: nobody hears the event.
    await finishJob();
    expect(statusReads).toEqual([]);
    expect(labels()).toBe('-,-');

    await act(async () => { pendingRegistrations.forEach(register => register()); });
    expect(statusReads).toEqual(['completed']);
    expect(labels()).toBe('S1,S2');
    expect(refetches).toBe(1);
    expect(state.speakerIdentification.isSummaryHeld).toBe(false);
    expect(generate).toHaveBeenCalledTimes(1);
  });

  test('an already-completed run does not reload rows that carry labels', async () => {
    jobStatus = 'completed';
    labeled = true;
    await show(false);
    expect(statusReads).toEqual(['completed']);
    expect(refetches).toBe(0);
  });

  test('an already-completed run that found no speech does not reload rows', async () => {
    jobStatus = 'completed';
    speakerCount = 0;
    await show(false);
    expect(refetches).toBe(0);
  });

  test('leaving the page during the hold starts the summary so it outlives the page', async () => {
    await show();
    await act(async () => renderer!.unmount());
    renderer = undefined;
    expect(generate).toHaveBeenCalledTimes(1);
    expect(generate.mock.calls[0]).toEqual(['', { survivesUnmount: true }]);
  });

  test('leaving the page after the summary started does not start another', async () => {
    await show();
    await finishJob();
    summaryStatus = 'processing';
    await rerender();
    await act(async () => renderer!.unmount());
    renderer = undefined;
    expect(generate).toHaveBeenCalledTimes(1);
  });

  test('"Generate now" ends the wait and the auto-summary does not run again', async () => {
    await show();
    await act(async () => { state.generateNow('Focus on decisions'); });
    expect(generate).toHaveBeenCalledTimes(1);
    expect(generate.mock.calls[0][0]).toBe('Focus on decisions');
    expect(state.isWaitingForSpeakers).toBe(false);

    await finishJob();
    await act(async () => renderer!.unmount());
    renderer = undefined;
    expect(generate).toHaveBeenCalledTimes(1);
  });

  test('flags the summary for regeneration when identification finishes while it is being generated', async () => {
    await show(false);
    summaryStatus = 'processing';
    await rerender();
    await finishJob();
    expect(state.showSpeakerNamesHint).toBe(false);

    summaryStatus = 'completed';
    aiSummary = { markdown: 'Summary without names' } as MeetingSummary;
    await rerender();
    expect(state.showSpeakerNamesHint).toBe(true);
  });

  test('shows the regenerate hint on load when the saved summary predates the speaker labels', async () => {
    jobStatus = 'completed';
    labeled = true;
    speakersChangedSinceSummary = true;
    summaryStatus = 'completed';
    aiSummary = { markdown: 'Summary without names' } as MeetingSummary;
    await show(false);
    expect(state.showSpeakerNamesHint).toBe(true);
  });

  test('shows no hint on load when the saved summary is newer than the speaker labels', async () => {
    jobStatus = 'completed';
    labeled = true;
    summaryStatus = 'completed';
    aiSummary = { markdown: 'Summary with names' } as MeetingSummary;
    await show(false);
    expect(statusReads).toEqual(['completed']);
    expect(state.showSpeakerNamesHint).toBe(false);
  });

  test('shows the regenerate hint when a status refresh reports the summary predates the speakers', async () => {
    jobStatus = 'completed';
    labeled = true;
    summaryStatus = 'completed';
    aiSummary = { markdown: 'Summary without names' } as MeetingSummary;
    await show(false);
    expect(state.showSpeakerNamesHint).toBe(false);

    speakersChangedSinceSummary = true;
    await act(async () => { await state.speakerIdentification.refreshStatus(); });
    expect(state.showSpeakerNamesHint).toBe(true);
  });

  test('regenerating clears the persisted hint and a later status with the same flag does not raise it again', async () => {
    jobStatus = 'completed';
    labeled = true;
    speakersChangedSinceSummary = true;
    summaryStatus = 'completed';
    aiSummary = { markdown: 'Summary without names' } as MeetingSummary;
    await show(false);
    expect(state.showSpeakerNamesHint).toBe(true);

    summaryStatus = 'regenerating';
    await rerender();
    expect(state.showSpeakerNamesHint).toBe(false);
    summaryStatus = 'completed';
    await rerender();
    await act(async () => { await state.speakerIdentification.refreshStatus(); });
    expect(state.showSpeakerNamesHint).toBe(false);
  });

  test('a speaker rename still flags the summary when the saved status says it is current', async () => {
    jobStatus = 'completed';
    labeled = true;
    speakers = [speaker('S1', 30)];
    summaryStatus = 'completed';
    aiSummary = { markdown: 'Summary with names' } as MeetingSummary;
    await show(false);
    expect(state.showSpeakerNamesHint).toBe(false);

    await act(async () => { await state.meetingSpeakers.renameSpeaker('S1', 'Alice'); });
    expect(state.showSpeakerNamesHint).toBe(true);
  });

  test('keeps the speakers on screen when a later read fails', async () => {
    speakers = [speaker('S1', 30)];
    await show(false);
    expect(state.meetingSpeakers.speakers).toHaveLength(1);

    failSpeakerReads = true;
    await act(async () => { await state.meetingSpeakers.refresh(); });
    expect(state.meetingSpeakers.speakers).toHaveLength(1);
    expect(toastError).toHaveBeenCalledTimes(1);
  });

  test('"This is me" on a speaker with under 10 s of speech says no voiceprint was saved', async () => {
    speakers = [speaker('S1', 4), speaker('S2', 30)];
    await show(false);
    await act(async () => { await state.meetingSpeakers.setSelf('S2', true); });
    expect(toastInfo).not.toHaveBeenCalled();

    await act(async () => { await state.meetingSpeakers.setSelf('S1', true); });
    expect(toastInfo).toHaveBeenCalledTimes(1);
    expect(toastInfo.mock.calls[0][1]).toEqual({ description: SHORT_VOICEPRINT_MESSAGE });
  });

  test('"This is me" after "Forget my voice" says to run Identify again', async () => {
    speakers = [speaker('S1', 121, 'voice_missing')];
    await show(false);
    await act(async () => { await state.meetingSpeakers.setSelf('S1', true); });
    expect(toastInfo).toHaveBeenCalledTimes(1);
    expect(toastInfo.mock.calls[0][1]).toEqual({ description: MISSING_VOICE_MESSAGE });
  });

  test('saving the voiceprint of the speaker that is already "Me" does not flag the summary', async () => {
    jobStatus = 'completed';
    labeled = true;
    speakers = [{ ...speaker('S1', 121, 'not_saved'), is_self: true }];
    summaryStatus = 'completed';
    aiSummary = { markdown: 'Summary with names' } as MeetingSummary;
    await show(false);
    expect(state.showSpeakerNamesHint).toBe(false);

    await act(async () => { await state.meetingSpeakers.setSelf('S1', true); });
    expect(invoke.mock.calls.some(([command]) => command === 'api_update_meeting_speaker')).toBe(true);
    expect(state.showSpeakerNamesHint).toBe(false);

    await act(async () => { await state.meetingSpeakers.setSelf('S1', false); });
    expect(state.showSpeakerNamesHint).toBe(true);
  });
});
