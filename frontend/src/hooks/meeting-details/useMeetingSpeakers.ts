import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import { MeetingSpeaker } from '@/types';
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

  const refresh = useCallback(async () => {
    const requestId = ++requestIdRef.current;
    const result = await fetchMeetingSpeakers(meetingId);
    if (requestId === requestIdRef.current) setSpeakers(result);
  }, [meetingId]);

  useEffect(() => {
    setSpeakers([]);
    void refresh();
    return () => { requestIdRef.current += 1; };
  }, [refresh]);

  const saveEdit = useCallback(async (edit: SpeakerEdit, action: () => Promise<unknown>, failureMessage: string) => {
    try {
      await action();
    } catch (error) {
      console.error(`${failureMessage}:`, error);
      toast.error(failureMessage, { description: String(error) });
      return;
    }
    await refresh();
    onSpeakersEditedRef.current?.(edit);
  }, [refresh]);

  const renameSpeaker = useCallback((speakerKey: string, displayName: string) =>
    saveEdit('rename', () => invoke<MeetingSpeaker>('api_update_meeting_speaker', {
      meetingId, speakerKey, displayName: displayName.trim(), isSelf: null,
    }), 'Failed to rename speaker'),
  [meetingId, saveEdit]);

  const setSelf = useCallback((speakerKey: string, isSelf: boolean) =>
    saveEdit('self', () => invoke<MeetingSpeaker>('api_update_meeting_speaker', {
      meetingId, speakerKey, displayName: null, isSelf,
    }), 'Failed to update speaker'),
  [meetingId, saveEdit]);

  const mergeSpeakers = useCallback((fromKey: string, intoKey: string) =>
    saveEdit('merge', () => invoke('api_merge_meeting_speakers', { meetingId, fromKey, intoKey }),
      'Failed to merge speakers'),
  [meetingId, saveEdit]);

  const speakerMap = useMemo(() => speakersByKey(speakers), [speakers]);

  return { speakers, speakerMap, refresh, renameSpeaker, setSelf, mergeSpeakers };
}
