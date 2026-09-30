import { afterAll, afterEach, beforeEach, describe, expect, mock, test } from 'bun:test';
import { act, create, type ReactTestInstance, type ReactTestRenderer } from 'react-test-renderer';

// Which @tauri-apps/api/core a module sees depends on which suites loaded it
// first, so these tests stub the managers' own dependencies instead: the model
// APIs and selectTranscriptModel (its save/refusal logic is tested in
// tests/lib/transcript-model-selection.test.ts). What is under test is that the
// managers leave applying the selection to it. Restored for other suites.
const originalWhisper = { ...await import('../../src/lib/whisper') };
const originalParakeet = { ...await import('../../src/lib/parakeet') };
const originalSelection = { ...await import('../../src/lib/transcriptModelSelection') };
const originalMotion = { ...await import('framer-motion') };
afterAll(() => {
  mock.module('../../src/lib/whisper', () => originalWhisper);
  mock.module('../../src/lib/parakeet', () => originalParakeet);
  mock.module('../../src/lib/transcriptModelSelection', () => originalSelection);
  mock.module('framer-motion', () => originalMotion);
});

const availableModel = (name: string) => ({
  name, path: `/models/${name}`, size_mb: 100, accuracy: 'Good', speed: 'Fast',
  status: 'Available', quantization: 'Int8',
});
const modelApi = (modelName: string) => ({
  init: async () => {},
  getAvailableModels: async () => [availableModel(modelName)],
});
mock.module('../../src/lib/whisper', () => ({ ...originalWhisper, WhisperAPI: modelApi('small') }));
mock.module('../../src/lib/parakeet', () => ({
  ...originalParakeet, ParakeetAPI: modelApi('parakeet-tdt-0.6b-v3-int8'),
}));
mock.module('@tauri-apps/api/event', () => ({ listen: async () => () => {} }));

type Selection = Parameters<typeof originalSelection.selectTranscriptModel>[0];
let backendAccepts: boolean;
// Stands in for the real helper: applies the selection only when the save is accepted.
const selectTranscriptModel = mock(async (selection: Selection) => {
  if (!backendAccepts) return false;
  selection.onModelSelect?.(selection.modelName);
  return true;
});
mock.module('../../src/lib/transcriptModelSelection', () => ({ ...originalSelection, selectTranscriptModel }));

// framer-motion needs browser APIs; animations are not under test.
const Passthrough = ({ children }: { children?: React.ReactNode }) => <>{children}</>;
mock.module('framer-motion', () => ({
  ...originalMotion,
  AnimatePresence: Passthrough,
  motion: new Proxy({}, { get: () => Passthrough }),
}));

const { ModelManager: WhisperModelManager } = await import('../../src/components/WhisperModelManager');
const { ParakeetModelManager } = await import('../../src/components/ParakeetModelManager');

const onModelSelect = mock<(modelName: string) => void>(() => {});
let renderer: ReactTestRenderer | undefined;
beforeEach(() => {
  backendAccepts = true;
  selectTranscriptModel.mockClear(); onModelSelect.mockClear();
});
afterEach(async () => {
  if (renderer) await act(async () => renderer!.unmount());
  renderer = undefined;
});

/** Renders a manager with auto-save and picks its one available model. */
async function pickModel(manager: React.ReactElement) {
  await act(async () => { renderer = create(manager); });
  // Let the mount-time model list load.
  await act(() => new Promise(resolve => setTimeout(resolve, 20)));
  const card = renderer!.root.find((node: ReactTestInstance) =>
    typeof node.props.onSelect === 'function' && node.props.model?.status === 'Available');
  await act(async () => { await card.props.onSelect(); });
}

describe.each([
  ['Whisper', 'localWhisper', () => <WhisperModelManager autoSave onModelSelect={onModelSelect} />, 'small'],
  ['Parakeet', 'parakeet', () => <ParakeetModelManager autoSave onModelSelect={onModelSelect} />, 'parakeet-tdt-0.6b-v3-int8'],
])('%s model manager selection', (_name, provider, manager, modelName) => {
  test('a refused switch leaves the previous selection', async () => {
    backendAccepts = false;
    await pickModel(manager());
    expect(selectTranscriptModel).toHaveBeenCalledTimes(1);
    expect(selectTranscriptModel.mock.calls[0][0]).toMatchObject({ provider, modelName, autoSave: true });
    expect(onModelSelect).not.toHaveBeenCalled();
  });

  test('an accepted switch is applied and announced through the helper', async () => {
    await pickModel(manager());
    expect(onModelSelect).toHaveBeenCalledTimes(1);
    expect(selectTranscriptModel.mock.calls[0][0].successMessage).toStartWith('Switched to ');
  });
});
