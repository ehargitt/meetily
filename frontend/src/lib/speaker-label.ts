import type { MeetingSpeaker } from '@/types';

/** Number of distinct speaker colours; keys beyond this reuse colours in order. */
export const SPEAKER_COLOR_COUNT = 8;

const SPEAKER_KEY_PATTERN = /^S(\d+)$/;

type LabelSource = Pick<MeetingSpeaker, 'display_name' | 'is_self'>;

/** The 1-based speaker number in a key such as "S3", or null when the key has another shape. */
export function speakerNumber(speakerKey: string): number | null {
  const match = SPEAKER_KEY_PATTERN.exec(speakerKey);
  return match ? Number(match[1]) : null;
}

/**
 * The name shown for a speaker: its custom name, else "Me" for the person who recorded,
 * else "Speaker N" derived from the key. A key with no speaker row still gets "Speaker N".
 */
export function speakerLabel(speakerKey: string, speaker?: LabelSource | null): string {
  const name = speaker?.display_name?.trim();
  if (name) return name;
  if (speaker?.is_self) return 'Me';
  const n = speakerNumber(speakerKey);
  return n === null ? speakerKey : `Speaker ${n}`;
}

/** Colour slot for a speaker, falling back to the key-derived slot the backend would assign. */
export function speakerColorIndex(speakerKey: string, speaker?: Pick<MeetingSpeaker, 'color_index'> | null): number {
  if (speaker) return speaker.color_index % SPEAKER_COLOR_COUNT;
  const n = speakerNumber(speakerKey);
  return n === null ? 0 : (n - 1) % SPEAKER_COLOR_COUNT;
}

export function speakersByKey(speakers: readonly MeetingSpeaker[]): Map<string, MeetingSpeaker> {
  return new Map(speakers.map(speaker => [speaker.speaker_key, speaker]));
}
