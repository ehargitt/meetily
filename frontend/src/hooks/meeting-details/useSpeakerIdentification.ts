import { useState, useCallback, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { toast } from 'sonner';
import {
  SpeakerIdComplete,
  SpeakerIdError,
  SpeakerIdJobStatus,
  SpeakerIdOptions,
  SpeakerIdProgress,
  SpeakerIdStatus,
  StartSpeakerIdResult,
} from '@/types';

/** Longest the auto-summary waits for speaker identification before running without names. */
export const SUMMARY_HOLD_TIMEOUT_MS = 10 * 60 * 1000;

const ACTIVE_STATUSES: readonly SpeakerIdJobStatus[] = ['queued', 'running'];

// Codes that need no user-facing message: the user asked for them or the meeting is gone.
const SILENT_ERROR_CODES: readonly SpeakerIdError['code'][] = ['cancelled', 'meeting_deleted'];

interface UseSpeakerIdentificationProps {
  meetingId: string;
  /** Hold the auto-summary while this meeting's identification is queued or running. */
  holdSummary: boolean;
  onComplete?: (result: SpeakerIdComplete) => void | Promise<void>;
  /**
   * The first status read found a run that had already completed. Its completion event may
   * have fired before this page was listening, so data loaded earlier can lack the labels.
   */
  onAlreadyCompleted?: (status: SpeakerIdStatus) => void | Promise<void>;
}

export interface UseSpeakerIdentificationReturn {
  /** Null until the first status read finishes. */
  status: SpeakerIdStatus | null;
  progress: SpeakerIdProgress | null;
  error: SpeakerIdError | null;
  isActive: boolean;
  /**
   * True while the auto-summary should wait. Released for good on completion, failure, a
   * non-running status, a failed status read, or after SUMMARY_HOLD_TIMEOUT_MS.
   */
  isSummaryHeld: boolean;
  start: (options: SpeakerIdOptions) => Promise<StartSpeakerIdResult | null>;
  cancel: () => Promise<void>;
  refreshStatus: () => Promise<void>;
}

export function useSpeakerIdentification({
  meetingId,
  holdSummary,
  onComplete,
  onAlreadyCompleted,
}: UseSpeakerIdentificationProps): UseSpeakerIdentificationReturn {
  const [status, setStatus] = useState<SpeakerIdStatus | null>(null);
  const [progress, setProgress] = useState<SpeakerIdProgress | null>(null);
  const [error, setError] = useState<SpeakerIdError | null>(null);
  const [summaryReleased, setSummaryReleased] = useState(false);
  const statusRequestRef = useRef(0);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const onAlreadyCompletedRef = useRef(onAlreadyCompleted);
  onAlreadyCompletedRef.current = onAlreadyCompleted;

  /** Resolves to the status it stored, or null when the read failed or a newer read superseded it. */
  const readStatus = useCallback(async (): Promise<SpeakerIdStatus | null> => {
    const requestId = ++statusRequestRef.current;
    try {
      const next = await invoke<SpeakerIdStatus>('get_speaker_identification_status', { meetingId });
      if (requestId !== statusRequestRef.current) return null;
      setStatus(next);
      return next;
    } catch (err) {
      console.warn('Failed to read speaker identification status:', err);
      // Without a status the summary cannot know what it is waiting for.
      if (requestId === statusRequestRef.current) setSummaryReleased(true);
      return null;
    }
  }, [meetingId]);

  const refreshStatus = useCallback(async () => { await readStatus(); }, [readStatus]);

  useEffect(() => {
    const unlisteners: UnlistenFn[] = [];
    let cleanedUp = false;

    const register = async <T extends { meeting_id: string }>(event: string, handler: (payload: T) => void) => {
      const unlisten = await listen<T>(event, ({ payload }) => {
        if (payload.meeting_id === meetingId) handler(payload);
      });
      if (cleanedUp) unlisten();
      else unlisteners.push(unlisten);
    };

    const registrations = [
      register<SpeakerIdProgress>('speaker-identification-progress', payload => {
        setProgress(payload);
        setError(null);
        setStatus(prev => prev && {
          ...prev, status: 'running', stage: payload.stage, progress_percentage: payload.progress_percentage,
        });
      }),

      register<SpeakerIdComplete>('speaker-identification-complete', payload => {
        setProgress(null);
        setError(null);
        setSummaryReleased(true);
        void readStatus();
        if (payload.speaker_count === 0) {
          toast.info('No speech detected', { description: 'Speaker identification found nobody talking.' });
        } else {
          toast.success(`Identified ${payload.speaker_count} speaker${payload.speaker_count === 1 ? '' : 's'}`);
        }
        void onCompleteRef.current?.(payload);
      }),

      register<SpeakerIdError>('speaker-identification-error', payload => {
        setProgress(null);
        setSummaryReleased(true);
        void readStatus();
        const silent = SILENT_ERROR_CODES.includes(payload.code);
        setError(silent ? null : payload);
        if (!silent) toast.error('Speaker identification failed', { description: payload.error });
      }),
    ];

    // Read the status only once every listener is in place: an event emitted in between
    // would otherwise be lost, leaving a stale "running" status or unlabelled rows.
    void (async () => {
      const results = await Promise.allSettled(registrations);
      results.forEach(result => {
        if (result.status === 'rejected') {
          console.warn('Failed to listen for speaker identification events:', result.reason);
        }
      });
      if (cleanedUp) return;
      // A completion event handled meanwhile makes this read stale; readStatus then yields null.
      const initial = await readStatus();
      if (!cleanedUp && initial?.status === 'completed') void onAlreadyCompletedRef.current?.(initial);
    })();

    return () => {
      cleanedUp = true;
      statusRequestRef.current += 1;
      unlisteners.forEach(unlisten => unlisten());
    };
  }, [meetingId, readStatus]);

  const isActive = status !== null && ACTIVE_STATUSES.includes(status.status);

  useEffect(() => {
    if (holdSummary && status !== null && !isActive) setSummaryReleased(true);
  }, [holdSummary, status, isActive]);

  useEffect(() => {
    if (!holdSummary || summaryReleased) return;
    const timer = setTimeout(() => {
      console.warn('Speaker identification is taking too long; generating the summary without speaker names');
      setSummaryReleased(true);
    }, SUMMARY_HOLD_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [holdSummary, summaryReleased]);

  const start = useCallback(async (options: SpeakerIdOptions) => {
    setError(null);
    setProgress(null);
    try {
      const result = await invoke<StartSpeakerIdResult>('start_speaker_identification', {
        meetingId,
        trigger: 'manual',
        options,
      });
      await refreshStatus();
      return result;
    } catch (err) {
      console.error('Failed to start speaker identification:', err);
      setError({ meeting_id: meetingId, code: 'internal', error: String(err) });
      return null;
    }
  }, [meetingId, refreshStatus]);

  const cancel = useCallback(async () => {
    try {
      await invoke<boolean>('cancel_speaker_identification', { meetingId });
    } catch (err) {
      console.error('Failed to cancel speaker identification:', err);
      toast.error('Failed to cancel speaker identification');
    }
  }, [meetingId]);

  return {
    status,
    progress,
    error,
    isActive,
    isSummaryHeld: holdSummary && !summaryReleased && (status === null || isActive),
    start,
    cancel,
    refreshStatus,
  };
}
