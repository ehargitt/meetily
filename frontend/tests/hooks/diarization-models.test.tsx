import { afterAll, afterEach, beforeEach, describe, expect, mock, test } from 'bun:test';
import type { ReactNode } from 'react';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import type { DiarizationModelsStatus, SpeakerIdStatus } from '../../src/types';
import type { UseSpeakerIdentificationReturn } from '../../src/hooks/meeting-details/useSpeakerIdentification';

// Bun shares module mocks between test files; restore the real modules after this suite.
// The real event module is never loaded (see speaker-identification-gate.test.tsx); its mock stays registered.
const originalCore = { ...await import('@tauri-apps/api/core') };
const originalDialog = { ...await import('../../src/components/ui/dialog') };
const originalSelect = { ...await import('../../src/components/ui/select') };
afterAll(() => {
  mock.module('@tauri-apps/api/core', () => originalCore);
  mock.module('../../src/components/ui/dialog', () => originalDialog);
  mock.module('../../src/components/ui/select', () => originalSelect);
});

// Radix renders dialogs and select menus through DOM portals, which the test renderer lacks.
const Pass = ({ children }: { children?: ReactNode }) => <div>{children}</div>;
mock.module('../../src/components/ui/dialog', () => ({
  Dialog: ({ open, children }: { open: boolean; children?: ReactNode }) => (open ? <div>{children}</div> : null),
  DialogContent: Pass, DialogDescription: Pass, DialogFooter: Pass, DialogHeader: Pass, DialogTitle: Pass,
}));
mock.module('../../src/components/ui/select', () => ({
  Select: Pass, SelectContent: Pass, SelectItem: Pass, SelectTrigger: Pass, SelectValue: () => null,
}));

let modelsStatus: DiarizationModelsStatus;
let finishDownload: (error?: string) => void;
const invoke = mock(async (command: string): Promise<unknown> => {
  if (command === 'get_diarization_models_status') return modelsStatus;
  if (command === 'download_diarization_models') {
    modelsStatus = { ...modelsStatus, download_in_progress: true };
    return new Promise<void>((resolve, reject) => {
      finishDownload = error => {
        modelsStatus = { ...modelsStatus, download_in_progress: false };
        if (error) reject(error); else resolve();
      };
    });
  }
  if (command === 'cancel_diarization_models_download') return undefined;
  throw new Error(`Unexpected command: ${command}`);
});
mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
mock.module('@tauri-apps/api/event', () => ({ listen: async () => () => {} }));

const { useDiarizationModels } = await import('../../src/hooks/useDiarizationModels');
const { IdentifySpeakersDialog, LONG_RECORDING_WARNING } = await import('../../src/components/Speakers/IdentifySpeakersDialog');

// The hook refreshes on window focus; bun has no window, and other suites may have installed one.
const previousWindow = Object.getOwnPropertyDescriptor(globalThis, 'window');
const focusTarget = new EventTarget();
Object.defineProperty(globalThis, 'window', { configurable: true, writable: true, value: focusTarget });
afterAll(() => {
  if (previousWindow) Object.defineProperty(globalThis, 'window', previousWindow);
  else delete (globalThis as Record<string, unknown>).window;
});

const modelsReady = (overrides: Partial<DiarizationModelsStatus> = {}): DiarizationModelsStatus => ({
  installed: false, missing: ['segmentation.onnx'], total_bytes: 32_523_463, models_dir: '/models',
  download_in_progress: false, ...overrides,
});

let renderer: ReactTestRenderer | undefined;
beforeEach(() => {
  invoke.mockClear();
  modelsStatus = modelsReady();
  mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
});
afterEach(async () => {
  if (renderer) await act(async () => renderer!.unmount());
  renderer = undefined;
});

let models: ReturnType<typeof useDiarizationModels>;
function ModelsPanel() {
  models = useDiarizationModels();
  return null;
}
async function showModels() {
  await act(async () => { renderer = create(<ModelsPanel />); });
}

describe('speaker model download state', () => {
  test('shows a download that is already running when the screen opens', async () => {
    modelsStatus = modelsReady({ download_in_progress: true });
    await showModels();
    expect(models.isDownloading).toBe(true);
  });

  test('picks up a download started elsewhere when the window regains focus', async () => {
    await showModels();
    expect(models.isDownloading).toBe(false);
    modelsStatus = modelsReady({ download_in_progress: true });
    await act(async () => { focusTarget.dispatchEvent(new Event('focus')); });
    expect(models.isDownloading).toBe(true);
  });

  test('cancel keeps the download shown until Rust stops it, then a retry starts a new download', async () => {
    await showModels();
    await act(async () => { void models.download(); });
    expect(models.isDownloading).toBe(true);

    await act(async () => { await models.cancelDownload(); });
    expect(models.isCancelling).toBe(true);
    expect(models.isDownloading).toBe(true);

    await act(async () => { finishDownload('Download cancelled'); });
    expect(models.isDownloading).toBe(false);
    expect(models.isCancelling).toBe(false);
    expect(models.downloadError).toBeNull();

    await act(async () => { void models.download(); });
    expect(models.isDownloading).toBe(true);
    expect(models.downloadError).toBeNull();
    expect(invoke.mock.calls.filter(([command]) => command === 'download_diarization_models')).toHaveLength(2);
  });

  test('a failed download reports its error', async () => {
    await showModels();
    await act(async () => { void models.download(); });
    await act(async () => { finishDownload('connection reset'); });
    expect(models.isDownloading).toBe(false);
    expect(models.downloadError).toBe('connection reset');
  });
});

describe('identify speakers dialog', () => {
  const identification = (audioDuration: number | null): UseSpeakerIdentificationReturn => {
    const status: SpeakerIdStatus = {
      meeting_id: 'meeting-a', status: 'none', audio_available: true, models_installed: true,
      audio_duration_seconds: audioDuration,
    };
    return {
      status, progress: null, error: null, isActive: false, isSummaryHeld: false,
      start: async () => null, cancel: async () => {}, refreshStatus: async () => {},
    };
  };
  const dialogText = () => JSON.stringify(renderer!.toJSON());

  test.each([
    [3 * 60 * 60 + 1, true],
    [3 * 60 * 60, false],
    [null, false],
  ])('recording of %p seconds shows the long-recording warning: %p', async (duration, warned) => {
    modelsStatus = modelsReady({ installed: true, missing: [] });
    await act(async () => {
      renderer = create(<IdentifySpeakersDialog open onOpenChange={() => {}} identification={identification(duration)} />);
    });
    expect(dialogText()).toContain('Identify speakers');
    expect(dialogText().includes(LONG_RECORDING_WARNING)).toBe(warned);
  });
});
