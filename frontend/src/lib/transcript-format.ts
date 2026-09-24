import type { MeetingSpeaker, Transcript } from '@/types';
import { speakerLabel, speakersByKey } from './speaker-label';

export type FormattableTranscript = Pick<Transcript, 'text' | 'timestamp' | 'audio_start_time' | 'speaker_key'>;

// Summary chunking splits long input at newlines, so a merged speaker run is capped to keep
// every chunk starting on a line that carries its speaker label.
const MAX_SUMMARY_LINE_CHARS = 1000;

/** `[MM:SS]` from recording-relative seconds; legacy rows without audio time keep their wall-clock timestamp. */
export function formatTranscriptTime(seconds: number | undefined, fallbackTimestamp: string): string {
  if (seconds === undefined) {
    return fallbackTimestamp;
  }
  const totalSecs = Math.floor(seconds);
  const mins = Math.floor(totalSecs / 60);
  const secs = totalSecs % 60;
  return `[${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}]`;
}

/**
 * Labels are only worth adding when they tell the reader something: several speakers, or a
 * single speaker who has a name or is the person who recorded.
 */
function resolveLabels(
  rows: readonly FormattableTranscript[],
  speakers: readonly MeetingSpeaker[],
): ((speakerKey: string) => string) | null {
  const keys = new Set(rows.flatMap(row => (row.speaker_key ? [row.speaker_key] : [])));
  if (keys.size === 0) return null;

  const byKey = speakersByKey(speakers);
  if (keys.size === 1) {
    const only = byKey.get([...keys][0]);
    if (!only?.is_self && !only?.display_name?.trim()) return null;
  }
  return speakerKey => speakerLabel(speakerKey, byKey.get(speakerKey));
}

/**
 * Transcript text sent to the summary model. Consecutive segments from one speaker become one
 * `[MM:SS] Label: text` line; without useful labels the output is the plain `[MM:SS] text` lines.
 */
export function formatTranscriptForSummary(
  rows: readonly FormattableTranscript[],
  speakers: readonly MeetingSpeaker[],
): string {
  const labelFor = resolveLabels(rows, speakers);
  if (!labelFor) {
    return rows
      .map(row => `${formatTranscriptTime(row.audio_start_time, row.timestamp)} ${row.text}`)
      .join('\n');
  }

  const lines: string[] = [];
  let run: { key: string; prefix: string; texts: string[]; length: number } | null = null;
  const flush = () => {
    if (run) lines.push(`${run.prefix}${run.texts.join(' ')}`);
    run = null;
  };

  for (const row of rows) {
    const time = formatTranscriptTime(row.audio_start_time, row.timestamp);
    const text = row.text.trim();
    if (!row.speaker_key) {
      flush();
      lines.push(`${time} ${row.text}`);
      continue;
    }
    if (run && (run.key !== row.speaker_key || (run.length > 0 && run.length + text.length > MAX_SUMMARY_LINE_CHARS))) {
      flush();
    }
    if (!run) {
      run = { key: row.speaker_key, prefix: `${time} ${labelFor(row.speaker_key)}: `, texts: [], length: 0 };
    }
    if (text) {
      run.texts.push(text);
      run.length += text.length + 1;
    }
  }
  flush();
  return lines.join('\n');
}

/** Transcript text for the clipboard: one markdown line per segment, labelled when labels are useful. */
export function formatTranscriptForCopy(
  rows: readonly FormattableTranscript[],
  speakers: readonly MeetingSpeaker[],
): string {
  const labelFor = resolveLabels(rows, speakers);
  return rows
    .map(row => {
      const time = formatTranscriptTime(row.audio_start_time, row.timestamp);
      const label = labelFor && row.speaker_key ? `${labelFor(row.speaker_key)}: ` : '';
      return `${time} ${label}${row.text}  `;
    })
    .join('\n');
}
