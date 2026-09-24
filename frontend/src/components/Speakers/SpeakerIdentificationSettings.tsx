import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { CheckCircle2, Download, X } from 'lucide-react';
import { toast } from 'sonner';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useDiarizationModels } from '@/hooks/useDiarizationModels';
import { ModelAttribution, formatModelSize } from './ModelAttribution';

interface SpeakerIdentificationSettingsProps {
  autoIdentify: boolean;
  onAutoIdentifyChange: (enabled: boolean) => void;
  disabled?: boolean;
}

export function SpeakerIdentificationSettings({
  autoIdentify,
  onAutoIdentifyChange,
  disabled = false,
}: SpeakerIdentificationSettingsProps) {
  const models = useDiarizationModels();
  const [isForgetting, setIsForgetting] = useState(false);
  const progress = models.downloadProgress;

  const handleForgetVoice = async () => {
    setIsForgetting(true);
    try {
      await invoke('forget_self_voiceprint');
      toast.success('Your voiceprint was deleted', {
        description: 'New meetings will no longer label you as "Me" automatically.',
      });
    } catch (error) {
      console.error('Failed to delete voiceprint:', error);
      toast.error('Failed to delete your voiceprint', { description: String(error) });
    } finally {
      setIsForgetting(false);
    }
  };

  return (
    <div className="space-y-4 border-t pt-6">
      <h4 className="text-base font-medium text-gray-900">Speaker Identification</h4>

      <div className="flex items-center justify-between p-4 border rounded-lg">
        <div className="flex-1">
          <div className="font-medium">Automatically identify speakers</div>
          <div className="text-sm text-gray-600">
            After a recording stops, label who said what. The summary waits for it (up to 10 minutes).
          </div>
        </div>
        <Switch checked={autoIdentify} onCheckedChange={onAutoIdentifyChange} disabled={disabled} />
      </div>

      <div className="p-4 border rounded-lg bg-gray-50 space-y-3">
        {models.status?.installed ? (
          <div className="flex items-center gap-2 text-sm text-green-700">
            <CheckCircle2 className="w-4 h-4" />
            Speaker models installed ({formatModelSize(models.status.total_bytes)})
          </div>
        ) : (
          <>
            <div className="text-sm text-gray-700">
              Speaker identification needs two models (about {formatModelSize(models.status?.total_bytes)}),
              downloaded once from Hugging Face. They run on this computer; your audio never leaves it.
            </div>
            {models.isDownloading ? (
              <div className="space-y-2">
                <Progress
                  value={progress && progress.total_bytes > 0 ? (progress.downloaded_bytes / progress.total_bytes) * 100 : 0}
                  className="h-2"
                />
                <div className="flex items-center justify-between text-xs text-gray-600">
                  <span>{progress ? `Downloading ${progress.file}` : 'Starting download...'}</span>
                  <Button variant="ghost" size="sm" className="h-7 px-2" onClick={() => void models.cancelDownload()}>
                    <X className="w-3 h-3" />
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <Button variant="outline" size="sm" onClick={() => void models.download()}>
                <Download className="w-4 h-4" />
                {models.downloadError ? 'Retry download' : 'Download speaker models'}
              </Button>
            )}
            {models.downloadError && (
              <div className="text-sm text-red-700">Download failed: {models.downloadError}</div>
            )}
          </>
        )}
        <ModelAttribution />
      </div>

      <div className="flex items-center justify-between p-4 border rounded-lg">
        <div className="flex-1">
          <div className="font-medium">Forget my voice</div>
          <div className="text-sm text-gray-600">
            Delete the voiceprint saved when you marked a speaker as &quot;This is me&quot;.
          </div>
        </div>
        <Button variant="outline" size="sm" onClick={handleForgetVoice} disabled={isForgetting}>
          Forget my voice
        </Button>
      </div>
    </div>
  );
}
