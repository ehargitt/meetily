import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import { MeetingSpeaker, VoiceprintState } from '@/types';
import { speakersByKey } from '@/lib/speaker-label';

/**
 * Speakers identified for a meeting. Callers that only decorate text (summary, copy) treat a
 * failed read as "no speakers" so identification problems never block those features.
 */
export async function fetchMeetingSpeakers(meetingId: string): Promise<MeetingSpeaker[]> {
  try {
    return await invoke<MeetingSpeaker[]>('api_get_meeting_speakers', { meetingId });
  } catch (error) {
    console.warn('Failed to load meeting speakers; continuing without speaker labels:', error);
    return [];
  }
}

export const SHORT_VOICEPRINT_MESSAGE = 'Needs at least 10 s of speech to save a voiceprint';
export const MISSING_VOICE_MESSAGE = 'Run Identify again to save a voiceprint for this speaker';

/** Why marking this speaker "This is me" saves no voiceprint, or null when it does. */
export function voiceprintProblem(state: VoiceprintState): string | null {
  if (state === 'too_short') return SHORT_VOICEPRINT_MESSAGE;
  if (state === 'voice_missing') return MISSING_VOICE_MESSAGE;
  return null;
}

export type SpeakerEdit = 'rename' | 'self' | 'merge';

interface UseMeetingSpeakersProps {
  meetingId: string;
  /** Called after an edit has been saved; a merge also changes which speaker each transcript row has. */
  onSpeakersEdited?: (edit: SpeakerEdit) => void;
}

export interface UseMeetingSpeakersReturn {
  speakers: MeetingSpeaker[];
  speakerMap: Map<string, MeetingSpeaker>;
  refresh: () => Promise<void>;
  /** An empty name resets the speaker to its default label. */
  renameSpeaker: (speakerKey: string, displayName: string) => Promise<void>;
  setSelf: (speakerKey: string, isSelf: boolean) => Promise<void>;
  mergeSpeakers: (fromKey: string, intoKey: string) => Promise<void>;
}

export function useMeetingSpeakers({ meetingId, onSpeakersEdited }: UseMeetingSpeakersProps): UseMeetingSpeakersReturn {
  const [speakers, setSpeakers] = useState<MeetingSpeaker[]>([]);
  const requestIdRef = useRef(0);
  const onSpeakersEditedRef = useRef(onSpeakersEdited);
  onSpeakersEditedRef.current = onSpeakersEdited;

  // A failed read keeps the speakers on screen: clearing them would hide every name.
  const refresh = useCallback(async () => {
    const requestId = ++requestIdRef.current;
    try {
      const result = await invoke<MeetingSpeaker[]>('api_get_meeting_speakers', { meetingId });
      if (requestId === requestIdRef.current) setSpeakers(result);
    } catch (error) {
      if (requestId !== requestIdRef.current) return;
      console.error('Failed to load meeting speakers:', error);
      toast.error('Failed to load speakers', { description: String(error) });
    }
  }, [meetingId]);

  useEffect(() => {
    setSpeakers([]);
    void refresh();
    return () => { requestIdRef.current += 1; };
  }, [refresh]);

  /** Resolves to whether the edit was saved. A null edit changed no speaker label, so it is not reported. */
  const saveEdit = useCallback(async (edit: SpeakerEdit | null, action: () => Promise<unknown>, failureMessage: string) => {
    try {
      await action();
    } catch (error) {
      console.error(`${failureMessage}:`, error);
      toast.error(failureMessage, { description: String(error) });
      return false;
    }
    await refresh();
    if (edit) onSpeakersEditedRef.current?.(edit);
    return true;
  }, [refresh]);

  const renameSpeaker = useCallback(async (speakerKey: string, displayName: string) => {
    await saveEdit('rename', () => invoke<MeetingSpeaker>('api_update_meeting_speaker', {
      meetingId, speakerKey, displayName: displayName.trim(), isSelf: null,
    }), 'Failed to rename speaker');
  }, [meetingId, saveEdit]);

  const setSelf = useCallback(async (speakerKey: string, isSelf: boolean) => {
    // Marking the speaker that is already "Me" only saves its voiceprint again; no label changes.
    const labelChanges = speakers.find(candidate => candidate.speaker_key === speakerKey)?.is_self !== isSelf;
    let updated: MeetingSpeaker | undefined;
    const saved = await saveEdit(labelChanges ? 'self' : null, async () => {
      updated = await invoke<MeetingSpeaker>('api_update_meeting_speaker', {
        meetingId, speakerKey, displayName: null, isSelf,
      });
    }, 'Failed to update speaker');
    const problem = saved && isSelf && updated ? voiceprintProblem(updated.voiceprint) : null;
    if (problem) toast.info('Marked as you, but no voiceprint was saved', { description: problem });
  }, [meetingId, saveEdit, speakers]);

  const mergeSpeakers = useCallback(async (fromKey: string, intoKey: string) => {
    await saveEdit('merge', () => invoke('api_merge_meeting_speakers', { meetingId, fromKey, intoKey }),
      'Failed to merge speakers');
  }, [meetingId, saveEdit]);

  const speakerMap = useMemo(() => speakersByKey(speakers), [speakers]);

  return { speakers, speakerMap, refresh, renameSpeaker, setSelf, mergeSpeakers };
}
