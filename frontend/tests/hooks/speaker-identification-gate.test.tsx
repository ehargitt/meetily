import { afterAll, afterEach, beforeEach, describe, expect, mock, test } from 'bun:test';
import { useEffect } from 'react';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import type { SpeakerIdJobStatus, SpeakerIdStatus } from '../../src/types';

// Bun shares module mocks between test files; restore the real modules after this suite.
// The real event module is deliberately never loaded: once loaded it stays linked to core and
// breaks later suites that replace core with an invoke-only mock. Its mock is left registered.
const originalCore = { ...await import('@tauri-apps/api/core') };
const originalToast = { ...await import('sonner') };
afterAll(() => {
  mock.module('@tauri-apps/api/core', () => originalCore);
  mock.module('sonner', () => originalToast);
  handlers.clear();
});

const notify = mock(() => {});
mock.module('sonner', () => ({ toast: { info: notify, error: notify, success: notify, warning: notify } }));

let readStatus: () => Promise<SpeakerIdStatus>;
const invoke = mock(async (command: string): Promise<unknown> => {
  if (command === 'get_speaker_identification_status') return readStatus();
  throw new Error(`Unexpected command: ${command}`);
});
mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));

type Handler = (event: { payload: unknown }) => void;
const handlers = new Map<string, Set<Handler>>();
const listen = mock(async (event: string, handler: Handler) => {
  if (!handlers.has(event)) handlers.set(event, new Set());
  handlers.get(event)!.add(handler);
  return () => { handlers.get(event)?.delete(handler); };
});
mock.module('@tauri-apps/api/event', () => ({ listen }));

const { useSpeakerIdentification, SUMMARY_HOLD_TIMEOUT_MS } =
  await import('../../src/hooks/meeting-details/useSpeakerIdentification');

const status = (value: SpeakerIdJobStatus, meetingId = 'meeting-a'): SpeakerIdStatus => ({
  meeting_id: meetingId, status: value, audio_available: true, models_installed: true,
});

// Mirrors the page's auto-summary effect: generate once, only when not held.
const generate = mock(() => {});
const completed = mock(() => {});
let state: ReturnType<typeof useSpeakerIdentification>;
function AutoSummary({ holdSummary }: { holdSummary: boolean }) {
  state = useSpeakerIdentification({ meetingId: 'meeting-a', holdSummary, onComplete: completed });
  useEffect(() => {
    if (holdSummary && !state.isSummaryHeld) generate();
  }, [holdSummary, state.isSummaryHeld]);
  return <output>{String(state.isSummaryHeld)}</output>;
}

// Capture only the hold timer; React's own timers keep running normally.
const realSetTimeout = globalThis.setTimeout;
const realClearTimeout = globalThis.clearTimeout;
const holdTimers = new Map<number, () => void>();
let nextHoldTimer = 0;
let renderer: ReactTestRenderer | undefined;
beforeEach(() => {
  handlers.clear(); holdTimers.clear();
  invoke.mockClear(); listen.mockClear(); generate.mockClear(); completed.mockClear(); notify.mockClear();
  readStatus = async () => status('running');
  mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
  mock.module('@tauri-apps/api/event', () => ({ listen }));
  globalThis.setTimeout = ((callback: () => void, delay?: number, ...args: unknown[]) => {
    if (delay !== SUMMARY_HOLD_TIMEOUT_MS) return realSetTimeout(callback, delay, ...args);
    const id = -(++nextHoldTimer);
    holdTimers.set(id, callback);
    return id;
  }) as typeof setTimeout;
  globalThis.clearTimeout = ((id: number) => {
    if (holdTimers.delete(id)) return;
    realClearTimeout(id);
  }) as typeof clearTimeout;
});
afterEach(async () => {
  if (renderer) await act(async () => renderer!.unmount());
  renderer = undefined;
  globalThis.setTimeout = realSetTimeout;
  globalThis.clearTimeout = realClearTimeout;
});

async function show(holdSummary = true) {
  await act(async () => {
    const view = <AutoSummary holdSummary={holdSummary} />;
    if (renderer) renderer.update(view); else renderer = create(view);
  });
}
async function emit(event: string, payload: unknown) {
  await act(async () => { handlers.get(event)?.forEach(handler => handler({ payload })); });
}

describe('auto-summary gate for speaker identification', () => {
  test('holds while the status is loading and while identification runs', async () => {
    let resolveStatus!: (value: SpeakerIdStatus) => void;
    readStatus = () => new Promise(resolve => { resolveStatus = resolve; });
    await show();
    expect(state.isSummaryHeld).toBe(true);
    await act(async () => { resolveStatus(status('running')); });
    expect(state.isSummaryHeld).toBe(true);
    await emit('speaker-identification-progress', {
      meeting_id: 'meeting-a', stage: 'embedding', progress_percentage: 50, message: 'Analysing voices',
    });
    expect(state.isSummaryHeld).toBe(true);
    expect(generate).not.toHaveBeenCalled();
  });

  test('releases on completion and reports the result', async () => {
    await show();
    readStatus = async () => status('completed');
    await emit('speaker-identification-complete', { meeting_id: 'meeting-a', speaker_count: 2, labeled_segments: 7 });
    expect(state.isSummaryHeld).toBe(false);
    expect(generate).toHaveBeenCalledTimes(1);
    expect(completed).toHaveBeenCalledWith({ meeting_id: 'meeting-a', speaker_count: 2, labeled_segments: 7 });
  });

  test('releases on an identification error', async () => {
    await show();
    readStatus = async () => status('failed');
    await emit('speaker-identification-error', { meeting_id: 'meeting-a', code: 'decode_failed', error: 'bad audio' });
    expect(state.isSummaryHeld).toBe(false);
    expect(state.error?.code).toBe('decode_failed');
    expect(generate).toHaveBeenCalledTimes(1);
  });

  test('releases after the timeout while identification is still running', async () => {
    await show();
    expect(holdTimers.size).toBe(1);
    await act(async () => { holdTimers.forEach(callback => callback()); });
    expect(state.isSummaryHeld).toBe(false);
    expect(generate).toHaveBeenCalledTimes(1);
    expect(holdTimers.size).toBe(0);
  });

  test('ignores events for another meeting', async () => {
    await show();
    await emit('speaker-identification-complete', { meeting_id: 'meeting-b', speaker_count: 2, labeled_segments: 7 });
    await emit('speaker-identification-error', { meeting_id: 'meeting-b', code: 'internal', error: 'x' });
    expect(state.isSummaryHeld).toBe(true);
    expect(generate).not.toHaveBeenCalled();
  });

  test.each(['none', 'skipped', 'completed', 'failed', 'cancelled', 'interrupted'] as SpeakerIdJobStatus[])(
    'does not hold when the job status is %s',
    async (value) => {
      readStatus = async () => status(value);
      await show();
      expect(state.isSummaryHeld).toBe(false);
      expect(generate).toHaveBeenCalledTimes(1);
    },
  );

  test('holds while queued and stays released once released', async () => {
    readStatus = async () => status('queued');
    await show();
    expect(state.isSummaryHeld).toBe(true);
    await emit('speaker-identification-error', { meeting_id: 'meeting-a', code: 'cancelled', error: 'cancelled' });
    expect(state.isSummaryHeld).toBe(false);
    expect(state.error).toBeNull();
    readStatus = async () => status('running');
    await act(async () => { await state.refreshStatus(); });
    expect(state.isSummaryHeld).toBe(false);
    expect(generate).toHaveBeenCalledTimes(1);
  });

  test('releases when the status cannot be read', async () => {
    readStatus = async () => { throw new Error('command not found'); };
    await show();
    expect(state.isSummaryHeld).toBe(false);
    expect(generate).toHaveBeenCalledTimes(1);
  });

  test('never holds when the page is not waiting to auto-summarise', async () => {
    await show(false);
    expect(state.isSummaryHeld).toBe(false);
    expect(holdTimers.size).toBe(0);
  });
});
