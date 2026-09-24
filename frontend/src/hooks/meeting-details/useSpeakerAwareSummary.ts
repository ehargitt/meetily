import { useCallback, useEffect, useRef, useState } from 'react';
import { MeetingSummary, Transcript } from '@/types';
import { hasVisibleSummaryContent } from '@/lib/summary-content';
import { useMeetingSpeakers, SpeakerEdit, UseMeetingSpeakersReturn } from './useMeetingSpeakers';
import { useSpeakerIdentification, UseSpeakerIdentificationReturn } from './useSpeakerIdentification';
import type { GenerateSummaryOptions } from './useSummaryGeneration';

type SummaryStatus = 'idle' | 'processing' | 'summarizing' | 'regenerating' | 'completed' | 'error';

const GENERATING_STATUSES: readonly SummaryStatus[] = ['processing', 'summarizing', 'regenerating'];

interface UseSpeakerAwareSummaryProps {
  meetingId: string;
  /** The page was opened right after a recording with auto-summary on. */
  shouldAutoGenerate: boolean;
  isModelConfigLoading: boolean;
  /** Transcript rows loaded on the page. */
  transcripts: Transcript[];
  aiSummary: MeetingSummary | null;
  summaryStatus: SummaryStatus;
  generateSummary: (customPrompt: string, options?: GenerateSummaryOptions) => Promise<void>;
  onAutoGenerateStarted?: () => void;
  onRefetchTranscripts?: () => Promise<void>;
}

export interface UseSpeakerAwareSummaryReturn {
  meetingSpeakers: UseMeetingSpeakersReturn;
  speakerIdentification: UseSpeakerIdentificationReturn;
  /** The auto-summary is held until speaker identification finishes. */
  isWaitingForSpeakers: boolean;
  /** Stop waiting for speaker identification and summarize now. */
  generateNow: (customPrompt: string) => void;
  /** The summary on screen was written before the latest speaker changes. */
  showSpeakerNamesHint: boolean;
  dismissSpeakerNamesHint: () => void;
}

/**
 * Keeps a meeting page's transcript and summary in step with speaker identification: reloads
 * rows when labels change, holds the post-recording auto-summary until identification ends,
 * and flags a summary written without the latest speaker names.
 */
export function useSpeakerAwareSummary({
  meetingId,
  shouldAutoGenerate,
  isModelConfigLoading,
  transcripts,
  aiSummary,
  summaryStatus,
  generateSummary,
  onAutoGenerateStarted,
  onRefetchTranscripts,
}: UseSpeakerAwareSummaryProps): UseSpeakerAwareSummaryReturn {
  const [showSpeakerNamesHint, setShowSpeakerNamesHint] = useState(false);
  const autoSummaryStartedForRef = useRef<string | null>(null);
  const speakersChangedDuringSummaryRef = useRef(false);
  const isSummaryGenerating = GENERATING_STATUSES.includes(summaryStatus);

  // There is no automatic regeneration; a summary being generated is flagged once it lands.
  const flagSummaryWithoutSpeakerNames = () => {
    if (isSummaryGenerating) speakersChangedDuringSummaryRef.current = true;
    else if (hasVisibleSummaryContent(aiSummary)) setShowSpeakerNamesHint(true);
  };

  const meetingSpeakers = useMeetingSpeakers({
    meetingId,
    onSpeakersEdited: (edit: SpeakerEdit) => {
      if (edit === 'merge') void onRefetchTranscripts?.();
      flagSummaryWithoutSpeakerNames();
    },
  });

  const speakerIdentification = useSpeakerIdentification({
    meetingId,
    holdSummary: shouldAutoGenerate,
    onComplete: async () => {
      await Promise.all([meetingSpeakers.refresh(), onRefetchTranscripts?.()]);
      flagSummaryWithoutSpeakerNames();
    },
    onAlreadyCompleted: async status => {
      if (status.speaker_count === 0 || transcripts.some(transcript => transcript.speaker_key)) return;
      await Promise.all([meetingSpeakers.refresh(), onRefetchTranscripts?.()]);
    },
  });

  useEffect(() => {
    if (isSummaryGenerating) {
      setShowSpeakerNamesHint(false);
      return;
    }
    if (!speakersChangedDuringSummaryRef.current) return;
    speakersChangedDuringSummaryRef.current = false;
    if (summaryStatus === 'completed') setShowSpeakerNamesHint(true);
  }, [isSummaryGenerating, summaryStatus]);

  const dismissSpeakerNamesHint = useCallback(() => setShowSpeakerNamesHint(false), []);

  const startAutoSummary = useCallback((customPrompt: string, options?: GenerateSummaryOptions) => {
    autoSummaryStartedForRef.current = meetingId;
    onAutoGenerateStarted?.();
    void generateSummary(customPrompt, options);
  }, [meetingId, onAutoGenerateStarted, generateSummary]);

  const isWaitingForSpeakers = shouldAutoGenerate
    && speakerIdentification.isSummaryHeld
    && summaryStatus === 'idle';
  const canAutoGenerate = shouldAutoGenerate
    && summaryStatus === 'idle'
    && !isModelConfigLoading
    && transcripts.length > 0;

  // Auto-generate only after the model configuration has settled.
  useEffect(() => {
    if (!canAutoGenerate || speakerIdentification.isSummaryHeld || autoSummaryStartedForRef.current === meetingId) {
      return;
    }
    console.log('🤖 Auto-generating summary...');
    startAutoSummary('');
  }, [canAutoGenerate, speakerIdentification.isSummaryHeld, meetingId, startAutoSummary]);

  // Leaving the page ends the hold. The summary runs in Rust, so start it now rather than lose it.
  const leavePageRef = useRef<() => void>(() => {});
  leavePageRef.current = () => {
    if (canAutoGenerate && isWaitingForSpeakers && autoSummaryStartedForRef.current !== meetingId) {
      console.log('Leaving the meeting before speaker identification finished; generating the summary now');
      startAutoSummary('', { survivesUnmount: true });
    }
  };
  useEffect(() => () => leavePageRef.current(), []);

  const generateNow = useCallback((customPrompt: string) => startAutoSummary(customPrompt), [startAutoSummary]);

  return {
    meetingSpeakers,
    speakerIdentification,
    isWaitingForSpeakers,
    generateNow,
    showSpeakerNamesHint,
    dismissSpeakerNamesHint,
  };
}
