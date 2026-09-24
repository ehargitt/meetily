import { useEffect, useRef, useState } from 'react';
import { AlertCircle, Download, Loader2, Users, X } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useDiarizationModels } from '@/hooks/useDiarizationModels';
import type { UseSpeakerIdentificationReturn } from '@/hooks/meeting-details/useSpeakerIdentification';
import { ModelAttribution, formatModelSize } from './ModelAttribution';

interface IdentifySpeakersDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  identification: UseSpeakerIdentificationReturn;
}

type Quality = 'fast' | 'accurate';

const SPEAKER_COUNT_OPTIONS = ['auto', '2', '3', '4', '5', '6', '7', '8', '9', '10'];

const STAGE_LABELS: Record<string, string> = {
  loading_models: 'Loading models',
  decoding: 'Reading audio',
  segmenting: 'Finding speech',
  embedding: 'Analysing voices',
  clustering: 'Grouping speakers',
  saving: 'Saving',
};

export function IdentifySpeakersDialog({ open, onOpenChange, identification }: IdentifySpeakersDialogProps) {
  const models = useDiarizationModels();
  const [speakerCount, setSpeakerCount] = useState('auto');
  const [quality, setQuality] = useState<Quality>('fast');
  const [startError, setStartError] = useState<string | null>(null);
  const { status, progress, error, isActive } = identification;

  // Close once a run watched by this dialog finishes cleanly; the hook announces the result.
  const wasActiveRef = useRef(false);
  useEffect(() => {
    if (open && wasActiveRef.current && !isActive && !error) onOpenChange(false);
    wasActiveRef.current = isActive;
  }, [open, isActive, error, onOpenChange]);

  const refreshModels = models.refresh;
  useEffect(() => {
    if (open) {
      setStartError(null);
      void refreshModels();
    }
  }, [open, refreshModels]);

  const handleStart = async () => {
    setStartError(null);
    const result = await identification.start({
      num_speakers: speakerCount === 'auto' ? null : Number(speakerCount),
      quality,
    });
    if (result?.status === 'skipped') {
      setStartError(skipReasonMessage(result.reason));
    }
  };

  const modelsInstalled = models.status?.installed ?? status?.models_installed ?? false;
  const failure = startError ?? error?.error ?? null;
  const percentage = progress?.progress_percentage ?? status?.progress_percentage ?? 0;
  const stage = progress?.stage ?? status?.stage ?? null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[460px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {isActive ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
                Identifying speakers...
              </>
            ) : failure ? (
              <>
                <AlertCircle className="h-5 w-5 text-red-600" />
                Speaker identification failed
              </>
            ) : (
              <>
                <Users className="h-5 w-5 text-blue-600" />
                Identify speakers
              </>
            )}
          </DialogTitle>
          <DialogDescription>
            {isActive
              ? 'This runs in the background. You can close this window.'
              : 'Label who said what in this meeting. Everything runs on this computer.'}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {isActive && (
            status?.status === 'queued' && !progress ? (
              <p className="text-sm text-muted-foreground">Waiting for another identification to finish...</p>
            ) : (
              <div className="space-y-2">
                <Progress value={Math.min(percentage, 100)} className="h-3" />
                <div className="flex justify-between text-xs text-gray-600">
                  <span>{stage ? STAGE_LABELS[stage] ?? stage : 'Starting'}</span>
                  <span>{Math.round(percentage)}%</span>
                </div>
                {progress?.message && (
                  <p className="text-sm text-muted-foreground text-center">{progress.message}</p>
                )}
              </div>
            )
          )}

          {!isActive && !modelsInstalled && (
            <div className="space-y-3">
              <p className="text-sm text-gray-700">
                Speaker identification needs two models (about {formatModelSize(models.status?.total_bytes)}),
                downloaded once from Hugging Face. Your audio never leaves this computer.
              </p>
              <ModelAttribution />
              {models.isDownloading && (
                <div className="space-y-1">
                  <Progress value={downloadPercentage(models.downloadProgress)} className="h-2" />
                  <p className="text-xs text-gray-600">
                    {models.downloadProgress ? `Downloading ${models.downloadProgress.file}` : 'Starting download...'}
                  </p>
                </div>
              )}
              {models.downloadError && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <p className="text-sm text-red-800">Download failed: {models.downloadError}</p>
                </div>
              )}
            </div>
          )}

          {!isActive && modelsInstalled && !failure && (
            <div className="space-y-4">
              {status?.status === 'interrupted' && (
                <p className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-md p-2">
                  The last run was interrupted before it finished. Run it again to label this meeting.
                </p>
              )}
              <div className="space-y-2">
                <span className="text-sm font-medium">Number of speakers</span>
                <Select value={speakerCount} onValueChange={setSpeakerCount}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SPEAKER_COUNT_OPTIONS.map(option => (
                      <SelectItem key={option} value={option}>
                        {option === 'auto' ? 'Detect automatically' : option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <span className="text-sm font-medium">Quality</span>
                <Select value={quality} onValueChange={value => setQuality(value as Quality)}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fast">Fast (recommended)</SelectItem>
                    <SelectItem value="accurate">Accurate (about 3x slower)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <p className="text-xs text-muted-foreground">
                Running again keeps the names you gave to voices it finds again.
              </p>
            </div>
          )}

          {!isActive && failure && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-sm text-red-800">{failure}</p>
            </div>
          )}
        </div>

        <DialogFooter>
          {isActive && (
            <Button variant="outline" onClick={() => void identification.cancel()}>
              <X className="h-4 w-4 mr-2" />
              Cancel identification
            </Button>
          )}

          {!isActive && !modelsInstalled && (
            models.isDownloading ? (
              <Button variant="outline" onClick={() => void models.cancelDownload()}>
                <X className="h-4 w-4 mr-2" />
                Cancel download
              </Button>
            ) : (
              <>
                <Button variant="outline" onClick={() => onOpenChange(false)}>
                  Not now
                </Button>
                <Button onClick={() => void models.download()} className="bg-blue-600 hover:bg-blue-700">
                  <Download className="h-4 w-4 mr-2" />
                  {models.downloadError ? 'Retry download' : 'Download models'}
                </Button>
              </>
            )
          )}

          {!isActive && modelsInstalled && !failure && (
            <>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button onClick={handleStart} className="bg-blue-600 hover:bg-blue-700">
                <Users className="h-4 w-4 mr-2" />
                Identify speakers
              </Button>
            </>
          )}

          {!isActive && modelsInstalled && failure && (
            <>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Close
              </Button>
              <Button variant="outline" onClick={handleStart}>
                Try again
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function downloadPercentage(progress: { downloaded_bytes: number; total_bytes: number } | null): number {
  if (!progress || progress.total_bytes <= 0) return 0;
  return Math.min(100, (progress.downloaded_bytes / progress.total_bytes) * 100);
}

function skipReasonMessage(reason: string | null | undefined): string {
  switch (reason) {
    case 'no_audio':
      return 'This meeting has no saved recording to analyse.';
    case 'models_missing':
      return 'The speaker identification models are not installed.';
    case 'already_running':
      return 'Speaker identification is already running for this meeting.';
    case 'retranscription_running':
      return 'Wait for the transcript enhancement to finish, then try again.';
    default:
      return reason ? `Speaker identification was skipped: ${reason}` : 'Speaker identification was skipped.';
  }
}
