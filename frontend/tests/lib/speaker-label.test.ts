import { describe, expect, test } from 'bun:test';
import { speakerColorIndex, speakerLabel, speakerNumber, speakersByKey } from '../../src/lib/speaker-label';
import type { MeetingSpeaker } from '../../src/types';

const speaker = (overrides: Partial<MeetingSpeaker> = {}): MeetingSpeaker => ({
  speaker_key: 'S2', display_name: null, is_self: false, color_index: 1,
  segment_count: 3, talk_time_seconds: 12, ...overrides,
});

describe('speaker labels', () => {
  test('parses the speaker number from S<n> keys only', () => {
    expect(speakerNumber('S1')).toBe(1);
    expect(speakerNumber('S12')).toBe(12);
    expect(speakerNumber('s1')).toBeNull();
    expect(speakerNumber('S')).toBeNull();
    expect(speakerNumber('SPEAKER_00')).toBeNull();
  });

  test('prefers the display name, then "Me", then "Speaker N"', () => {
    expect(speakerLabel('S2', speaker({ display_name: 'Alice', is_self: true }))).toBe('Alice');
    expect(speakerLabel('S2', speaker({ display_name: '  Alice ' }))).toBe('Alice');
    expect(speakerLabel('S2', speaker({ is_self: true }))).toBe('Me');
    expect(speakerLabel('S2', speaker({ display_name: '' }))).toBe('Speaker 2');
    expect(speakerLabel('S2', speaker())).toBe('Speaker 2');
  });

  test('labels keys without a speaker row and keeps unknown key shapes verbatim', () => {
    expect(speakerLabel('S7')).toBe('Speaker 7');
    expect(speakerLabel('S7', null)).toBe('Speaker 7');
    expect(speakerLabel('guest')).toBe('guest');
  });

  test('uses the stored colour, else (n - 1) mod 8 from the key', () => {
    expect(speakerColorIndex('S2', speaker({ color_index: 5 }))).toBe(5);
    expect(speakerColorIndex('S1')).toBe(0);
    expect(speakerColorIndex('S8')).toBe(7);
    expect(speakerColorIndex('S9')).toBe(0);
    expect(speakerColorIndex('guest')).toBe(0);
  });

  test('indexes speakers by key', () => {
    const map = speakersByKey([speaker({ speaker_key: 'S1' }), speaker({ speaker_key: 'S2', display_name: 'Bo' })]);
    expect([...map.keys()]).toEqual(['S1', 'S2']);
    expect(map.get('S2')?.display_name).toBe('Bo');
  });
});
