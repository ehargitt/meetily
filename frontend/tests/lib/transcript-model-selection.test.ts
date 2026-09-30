import { afterAll, beforeEach, describe, expect, mock, test } from 'bun:test';

// Bun shares module mocks between test files; restore the real modules other suites use.
const originalCore = { ...await import('@tauri-apps/api/core') };
const originalToast = { ...await import('sonner') };
afterAll(() => {
  mock.module('@tauri-apps/api/core', () => originalCore);
  mock.module('sonner', () => originalToast);
});

let saveResult: () => Promise<void>;
const invoke = mock<(command: string, args?: Record<string, unknown>) => Promise<unknown>>(async (command) => {
  if (command === 'api_save_transcript_config') return saveResult();
  throw new Error(`Unexpected command: ${command}`);
});
mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));

type ToastCall = (message: string, options?: Record<string, unknown>) => void;
const toastError = mock<ToastCall>(() => {});
const toastSuccess = mock<ToastCall>(() => {});
const notify = mock(() => {});
mock.module('sonner', () => ({ toast: { info: notify, error: toastError, success: toastSuccess, warning: notify } }));

const { selectTranscriptModel } = await import('../../src/lib/transcriptModelSelection');

const onModelSelect = mock<(modelName: string) => void>(() => {});
const select = (autoSave: boolean) => selectTranscriptModel({
  provider: 'localWhisper', modelName: 'large-v3', autoSave, onModelSelect, successMessage: 'Switched to Large v3',
});

beforeEach(() => {
  saveResult = async () => {};
  invoke.mockClear(); toastError.mockClear(); toastSuccess.mockClear(); onModelSelect.mockClear();
  mock.module('@tauri-apps/api/core', () => ({ ...originalCore, invoke }));
});

describe('selecting a transcription model', () => {
  test('a switch refused during a recording keeps the previous selection', async () => {
    saveResult = async () => {
      throw 'Cannot change the transcription model while a recording is in progress or its transcription is still finishing.';
    };
    expect(await select(true)).toBe(false);
    expect(onModelSelect).not.toHaveBeenCalled();
    expect(toastSuccess).not.toHaveBeenCalled();
    expect(toastError).toHaveBeenCalledTimes(1);
    expect(toastError.mock.calls[0][1]?.description).toStartWith('Cannot change the transcription model');
  });

  test('an accepted switch is saved, then applied and announced', async () => {
    expect(await select(true)).toBe(true);
    expect(invoke).toHaveBeenCalledWith('api_save_transcript_config',
      { provider: 'localWhisper', model: 'large-v3', apiKey: null });
    expect(onModelSelect).toHaveBeenCalledWith('large-v3');
    expect(toastSuccess).toHaveBeenCalledWith('Switched to Large v3', { duration: 3000 });
    expect(toastError).not.toHaveBeenCalled();
  });

  test('without auto-save the selection is applied without saving', async () => {
    expect(await select(false)).toBe(true);
    expect(invoke).not.toHaveBeenCalled();
    expect(onModelSelect).toHaveBeenCalledWith('large-v3');
  });
});
