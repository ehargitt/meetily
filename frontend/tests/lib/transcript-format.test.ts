import { describe, expect, test } from 'bun:test';
import {
  formatTranscriptForCopy,
  formatTranscriptForSummary,
  type FormattableTranscript,
} from '../../src/lib/transcript-format';
import type { MeetingSpeaker } from '../../src/types';

// The formatting the summary and copy paths used before speaker labels existed.
const legacyTime = (seconds: number | undefined, fallback: string) => {
  if (seconds === undefined) return fallback;
  const total = Math.floor(seconds);
  return `[${Math.floor(total / 60).toString().padStart(2, '0')}:${(total % 60).toString().padStart(2, '0')}]`;
};
const legacySummary = (rows: FormattableTranscript[]) =>
  rows.map(row => `${legacyTime(row.audio_start_time, row.timestamp)} ${row.text}`).join('\n');
const legacyCopy = (rows: FormattableTranscript[]) =>
  rows.map(row => `${legacyTime(row.audio_start_time, row.timestamp)} ${row.text}  `).join('\n');

const row = (start: number | undefined, text: string, speaker_key?: string | null): FormattableTranscript => ({
  text, timestamp: '14:30:05', audio_start_time: start, speaker_key,
});
const speaker = (key: string, overrides: Partial<MeetingSpeaker> = {}): MeetingSpeaker => ({
  speaker_key: key, display_name: null, is_self: false, color_index: Number(key.slice(1)) - 1,
  segment_count: 1, talk_time_seconds: 1, voiceprint: 'too_short', ...overrides,
});

const unlabelled = [row(0.5, ' Hello there.'), row(65.9, 'Second line'), row(undefined, 'Legacy row')];

describe('formatTranscriptForSummary', () => {
  test('is byte-identical to the legacy format without speaker labels', () => {
    expect(formatTranscriptForSummary(unlabelled, [])).toBe(legacySummary(unlabelled));
    expect(formatTranscriptForSummary(unlabelled, [speaker('S1')])).toBe(legacySummary(unlabelled));
  });

  test('adds no prefix for a single unnamed speaker who is not the recorder', () => {
    const rows = [row(1, 'One', 'S1'), row(4, 'Two', 'S1')];
    expect(formatTranscriptForSummary(rows, [speaker('S1')])).toBe(legacySummary(rows));
    expect(formatTranscriptForSummary(rows, [])).toBe(legacySummary(rows));
  });

  test('labels a single speaker who is named or is the recorder', () => {
    const rows = [row(1, 'One', 'S1'), row(4, 'Two', 'S1')];
    expect(formatTranscriptForSummary(rows, [speaker('S1', { is_self: true })])).toBe('[00:01] Me: One Two');
    expect(formatTranscriptForSummary(rows, [speaker('S1', { display_name: 'Alice' })])).toBe('[00:01] Alice: One Two');
  });

  test('merges runs of one speaker and resolves renamed, self and default labels', () => {
    const rows = [
      row(0, ' Hi, everyone.', 'S1'),
      row(3.2, 'Let us start.', 'S1'),
      row(9.7, 'Sounds good.', 'S2'),
      row(12, 'Next item.', 'S3'),
      row(75, 'Back to me.', 'S1'),
    ];
    const speakers = [speaker('S1', { is_self: true }), speaker('S2', { display_name: 'Alice' }), speaker('S3')];
    expect(formatTranscriptForSummary(rows, speakers)).toBe([
      '[00:00] Me: Hi, everyone. Let us start.',
      '[00:09] Alice: Sounds good.',
      '[00:12] Speaker 3: Next item.',
      '[01:15] Me: Back to me.',
    ].join('\n'));
  });

  test('a custom name wins over "Me", and a blank name falls back to the default', () => {
    const rows = [row(0, 'A', 'S1'), row(1, 'B', 'S2')];
    const speakers = [speaker('S1', { is_self: true, display_name: 'Eric' }), speaker('S2', { display_name: '  ' })];
    expect(formatTranscriptForSummary(rows, speakers)).toBe('[00:00] Eric: A\n[00:01] Speaker 2: B');
  });

  test('keeps unlabelled rows as plain lines that break a run', () => {
    const rows = [row(0, 'A', 'S1'), row(1, 'no speaker', null), row(2, 'B', 'S1'), row(3, 'C', 'S2')];
    expect(formatTranscriptForSummary(rows, [])).toBe(
      '[00:00] Speaker 1: A\n[00:01] no speaker\n[00:02] Speaker 1: B\n[00:03] Speaker 2: C',
    );
  });

  test('splits a long run so every line keeps its label', () => {
    const sentence = 'x'.repeat(400);
    const rows = [0, 10, 20, 30].map(start => row(start, sentence, 'S1')).concat(row(40, 'reply', 'S2'));
    const lines = formatTranscriptForSummary(rows, []).split('\n');
    expect(lines).toHaveLength(3);
    expect(lines[0]).toBe(`[00:00] Speaker 1: ${sentence} ${sentence}`);
    expect(lines[1]).toBe(`[00:20] Speaker 1: ${sentence} ${sentence}`);
    expect(lines[2]).toBe('[00:40] Speaker 2: reply');
  });
});

describe('formatTranscriptForCopy', () => {
  test('is byte-identical to the legacy format without speaker labels', () => {
    expect(formatTranscriptForCopy(unlabelled, [])).toBe(legacyCopy(unlabelled));
    const single = [row(1, 'One', 'S1'), row(4, 'Two', 'S1')];
    expect(formatTranscriptForCopy(single, [speaker('S1')])).toBe(legacyCopy(single));
  });

  test('labels every segment without merging runs', () => {
    const rows = [row(0, 'Hi', 'S1'), row(2, 'There', 'S1'), row(5, 'Hello', 'S2')];
    const speakers = [speaker('S1', { is_self: true }), speaker('S2', { display_name: 'Alice' })];
    expect(formatTranscriptForCopy(rows, speakers)).toBe('[00:00] Me: Hi  \n[00:02] Me: There  \n[00:05] Alice: Hello  ');
  });
});
