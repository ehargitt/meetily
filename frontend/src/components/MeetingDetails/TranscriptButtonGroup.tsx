"use client";

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { ButtonGroup } from '@/components/ui/button-group';
import { AudioLines, Copy, FolderOpen, Loader2, RefreshCw, Users } from 'lucide-react';
import Analytics from '@/lib/analytics';
import { RetranscribeDialog } from './RetranscribeDialog';
import { useConfig } from '@/contexts/ConfigContext';
import { SpeakersPopover } from '@/components/Speakers/SpeakersPopover';
import { IdentifySpeakersDialog } from '@/components/Speakers/IdentifySpeakersDialog';
import type { UseMeetingSpeakersReturn } from '@/hooks/meeting-details/useMeetingSpeakers';
import type { UseSpeakerIdentificationReturn } from '@/hooks/meeting-details/useSpeakerIdentification';


interface TranscriptButtonGroupProps {
  transcriptCount: number;
  onCopyTranscript: () => void;
  onOpenMeetingFolder: () => Promise<void>;
  meetingId?: string;
  meetingFolderPath?: string | null;
  onRefetchTranscripts?: () => Promise<void>;
  meetingSpeakers?: UseMeetingSpeakersReturn;
  speakerIdentification?: UseSpeakerIdentificationReturn;
}


export function TranscriptButtonGroup({
  transcriptCount,
  onCopyTranscript,
  onOpenMeetingFolder,
  meetingId,
  meetingFolderPath,
  onRefetchTranscripts,
  meetingSpeakers,
  speakerIdentification,
}: TranscriptButtonGroupProps) {
  const { betaFeatures } = useConfig();
  const [showRetranscribeDialog, setShowRetranscribeDialog] = useState(false);
  const [showIdentifyDialog, setShowIdentifyDialog] = useState(false);
  // Identification needs the saved recording; naming speakers only needs speakers to exist.
  const canIdentify = Boolean(meetingFolderPath && speakerIdentification?.status?.audio_available);
  const hasSpeakers = (meetingSpeakers?.speakers.length ?? 0) > 0;
  const identifyProgress = speakerIdentification?.progress?.progress_percentage
    ?? speakerIdentification?.status?.progress_percentage;

  const handleRetranscribeComplete = useCallback(async () => {
    // Refetch transcripts to show the updated data
    if (onRefetchTranscripts) {
      await onRefetchTranscripts();
    }
  }, [onRefetchTranscripts]);

  return (
    <div className="flex items-center justify-center w-full gap-2">
      <ButtonGroup>
        <Button
          variant="outline"
          size="sm"
          className="px-2 @[22rem]:px-3"
          onClick={() => {
            Analytics.trackButtonClick('copy_transcript', 'meeting_details');
            onCopyTranscript();
          }}
          disabled={transcriptCount === 0}
          title={transcriptCount === 0 ? 'No transcript available' : 'Copy Transcript'}
        >
          <Copy />
          <span className="hidden @[22rem]:inline">Copy</span>
        </Button>

        <Button
          size="sm"
          variant="outline"
          className="px-2 @[22rem]:px-4"
          onClick={() => {
            Analytics.trackButtonClick('open_recording_folder', 'meeting_details');
            onOpenMeetingFolder();
          }}
          title="Open Recording Folder"
        >
          <FolderOpen className="@[22rem]:mr-2" size={18} />
          <span className="hidden @[22rem]:inline">Recording</span>
        </Button>

        {betaFeatures.importAndRetranscribe && meetingId && meetingFolderPath && (
          <Button
            size="sm"
            variant="outline"
            className="bg-gradient-to-r from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 border-blue-200 px-2 @[22rem]:px-4"
            onClick={() => {
              Analytics.trackButtonClick('enhance_transcript', 'meeting_details');
              setShowRetranscribeDialog(true);
            }}
            title="Retranscribe to enhance your recorded audio"
          >
            <RefreshCw className="@[22rem]:mr-2" size={18} />
            <span className="hidden @[22rem]:inline">Enhance</span>
          </Button>
        )}

        {meetingSpeakers && hasSpeakers && (
          <SpeakersPopover
            speakers={meetingSpeakers.speakers}
            onRename={meetingSpeakers.renameSpeaker}
            onSetSelf={meetingSpeakers.setSelf}
            onMerge={meetingSpeakers.mergeSpeakers}
          >
            <Button
              size="sm"
              variant="outline"
              className="px-2 @[30rem]:px-4"
              onClick={() => Analytics.trackButtonClick('speakers', 'meeting_details')}
              title="Name the speakers in this meeting"
            >
              <Users className="@[30rem]:mr-2" size={18} />
              <span className="hidden @[30rem]:inline">Speakers</span>
            </Button>
          </SpeakersPopover>
        )}

        {speakerIdentification && canIdentify && (
          <Button
            size="sm"
            variant="outline"
            className="px-2 @[30rem]:px-4"
            onClick={() => {
              Analytics.trackButtonClick('identify_speakers', 'meeting_details');
              setShowIdentifyDialog(true);
            }}
            title={speakerIdentification.isActive ? 'Identifying speakers...' : 'Identify who is speaking'}
          >
            {speakerIdentification.isActive ? (
              <Loader2 className="animate-spin @[30rem]:mr-2" size={18} />
            ) : (
              <AudioLines className="@[30rem]:mr-2" size={18} />
            )}
            <span className="hidden @[30rem]:inline">
              {speakerIdentification.isActive
                ? (identifyProgress != null ? `${Math.round(identifyProgress)}%` : 'Identifying')
                : 'Identify'}
            </span>
          </Button>
        )}
      </ButtonGroup>

      {betaFeatures.importAndRetranscribe && meetingId && meetingFolderPath && (
        <RetranscribeDialog
          open={showRetranscribeDialog}
          onOpenChange={setShowRetranscribeDialog}
          meetingId={meetingId}
          meetingFolderPath={meetingFolderPath}
          onComplete={handleRetranscribeComplete}
        />
      )}

      {speakerIdentification && canIdentify && (
        <IdentifySpeakersDialog
          open={showIdentifyDialog}
          onOpenChange={setShowIdentifyDialog}
          identification={speakerIdentification}
        />
      )}
    </div>
  );
}
