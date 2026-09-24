import { invoke } from '@tauri-apps/api/core';
import { cn } from '@/lib/utils';

// Combined size of the two pinned model files, shown before the status command has answered.
const FALLBACK_MODEL_BYTES = 32_523_463;

export function formatModelSize(totalBytes: number | undefined): string {
  return `${((totalBytes || FALLBACK_MODEL_BYTES) / 1_000_000).toFixed(1)} MB`;
}

function ExternalLink({ url, children }: { url: string; children: string }) {
  return (
    <button
      type="button"
      className="underline hover:text-gray-700"
      onClick={() => {
        invoke('open_external_url', { url }).catch(error => console.error('Failed to open link:', error));
      }}
    >
      {children}
    </button>
  );
}

/** Credits the speaker identification models; the CC BY 4.0 licence requires this in the UI. */
export function ModelAttribution({ className }: { className?: string }) {
  return (
    <p className={cn('text-xs text-gray-500 leading-relaxed', className)}>
      Speaker identification uses pyannote segmentation-3.0 (© 2022 CNRS,{' '}
      <ExternalLink url="https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE">MIT License</ExternalLink>)
      and WeSpeaker ResNet34-LM trained on VoxCeleb (WeSpeaker,{' '}
      <ExternalLink url="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</ExternalLink>),
      both as ONNX exports from the sherpa-onnx project.
    </p>
  );
}
