import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';

export type LocalTranscriptProvider = 'localWhisper' | 'parakeet';

interface TranscriptModelSelection {
  provider: LocalTranscriptProvider;
  modelName: string;
  /** Persist the choice; it is then applied only if the backend accepts it. */
  autoSave: boolean;
  onModelSelect?: (modelName: string) => void;
  /** Success toast shown once the selection is applied. */
  successMessage?: string;
}

/**
 * Saves the transcription model choice.
 * @returns false when the backend refused it — it does while a recording or
 *   its transcription drain uses the engine — after showing the reason.
 */
export async function saveTranscriptModelSelection(
  provider: LocalTranscriptProvider,
  modelName: string,
): Promise<boolean> {
  try {
    await invoke('api_save_transcript_config', { provider, model: modelName, apiKey: null });
    return true;
  } catch (error) {
    console.error('Failed to save model selection:', error);
    toast.error('Could not change the transcription model', {
      description: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

/**
 * Makes `modelName` the selected transcription model. With `autoSave` the
 * choice is saved first, so a refused save leaves the previous selection in
 * the UI instead of showing a model the backend will not use.
 * @returns whether the selection was applied
 */
export async function selectTranscriptModel({
  provider, modelName, autoSave, onModelSelect, successMessage,
}: TranscriptModelSelection): Promise<boolean> {
  if (autoSave && !(await saveTranscriptModelSelection(provider, modelName))) {
    return false;
  }
  onModelSelect?.(modelName);
  if (successMessage) {
    toast.success(successMessage, { duration: 3000 });
  }
  return true;
}
