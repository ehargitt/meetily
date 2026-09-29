import { afterAll, afterEach, describe, expect, mock, test } from 'bun:test';
import type { ReactNode } from 'react';
import { act, create, type ReactTestInstance, type ReactTestRenderer } from 'react-test-renderer';
import type { MeetingSpeaker } from '../../src/types';

// Bun shares module mocks between test files; restore the real modules after this suite.
const originalPopover = { ...await import('../../src/components/ui/popover') };
const originalDropdown = { ...await import('../../src/components/ui/dropdown-menu') };
const originalSwitch = { ...await import('../../src/components/ui/switch') };
afterAll(() => {
  mock.module('../../src/components/ui/popover', () => originalPopover);
  mock.module('../../src/components/ui/dropdown-menu', () => originalDropdown);
  mock.module('../../src/components/ui/switch', () => originalSwitch);
});

// Radix renders popovers and menus through DOM portals, which the test renderer lacks.
const Pass = ({ children }: { children?: ReactNode }) => <div>{children}</div>;
mock.module('../../src/components/ui/popover', () => ({ Popover: Pass, PopoverContent: Pass, PopoverTrigger: Pass }));
mock.module('../../src/components/ui/dropdown-menu', () => ({
  DropdownMenu: Pass, DropdownMenuContent: Pass, DropdownMenuItem: Pass, DropdownMenuLabel: Pass, DropdownMenuTrigger: Pass,
}));
mock.module('../../src/components/ui/switch', () => ({
  Switch: ({ checked, onCheckedChange }: { checked: boolean; onCheckedChange: (checked: boolean) => void }) => (
    <button role="switch" aria-checked={checked} onClick={() => onCheckedChange(!checked)} />
  ),
}));

const { SpeakersPopover } = await import('../../src/components/Speakers/SpeakersPopover');
const { MISSING_VOICE_MESSAGE, SHORT_VOICEPRINT_MESSAGE } = await import('../../src/hooks/meeting-details/useMeetingSpeakers');

const speaker = (overrides: Partial<MeetingSpeaker>): MeetingSpeaker => ({
  speaker_key: 'S1', display_name: null, is_self: false, color_index: 0, segment_count: 3,
  talk_time_seconds: 60, voiceprint: 'ready', ...overrides,
});

let renderer: ReactTestRenderer | undefined;
afterEach(() => { renderer?.unmount(); renderer = undefined; });

const onSetSelf = mock(async (_key: string, _isSelf: boolean) => {});

async function render(speakers: MeetingSpeaker[]) {
  onSetSelf.mockClear();
  await act(async () => {
    renderer = create(
      <SpeakersPopover speakers={speakers} onRename={async () => {}} onSetSelf={onSetSelf} onMerge={async () => {}}>
        <button>Speakers</button>
      </SpeakersPopover>,
    );
  });
  return renderer!;
}

const text = (node: ReactTestRenderer) =>
  node.root.findAll(instance => typeof instance.type === 'string')
    .flatMap(instance => instance.children.filter((child): child is string => typeof child === 'string'))
    .join(' ');
const buttonLabelled = (node: ReactTestRenderer, label: string): ReactTestInstance | undefined =>
  node.root.findAll(instance => instance.type === 'button' && text({ root: instance } as ReactTestRenderer).includes(label))[0];

describe('SpeakersPopover voiceprint hints', () => {
  test('a "Me" speaker whose voiceprint was forgotten and whose voice is back offers "Save voiceprint"', async () => {
    const node = await render([speaker({ is_self: true, voiceprint: 'not_saved' })]);
    expect(text(node)).toContain('No voiceprint saved.');

    const save = buttonLabelled(node, 'Save voiceprint');
    expect(save).toBeDefined();
    await act(async () => { save!.props.onClick(); });
    expect(onSetSelf).toHaveBeenCalledWith('S1', true);
  });

  test('a missing voice is explained only on the "Me" speaker', async () => {
    const self = await render([speaker({ is_self: true, voiceprint: 'voice_missing' })]);
    expect(text(self)).toContain(MISSING_VOICE_MESSAGE);
    expect(buttonLabelled(self, 'Save voiceprint')).toBeUndefined();

    const other = await render([speaker({ voiceprint: 'voice_missing' })]);
    expect(text(other)).not.toContain(MISSING_VOICE_MESSAGE);
  });

  test('short speech is explained on every speaker; an enrollable one shows no hint', async () => {
    const node = await render([
      speaker({ speaker_key: 'S1', talk_time_seconds: 4, voiceprint: 'too_short' }),
      speaker({ speaker_key: 'S2', is_self: true, voiceprint: 'ready' }),
    ]);
    const shown = text(node);
    expect(shown.split(SHORT_VOICEPRINT_MESSAGE)).toHaveLength(2);
    expect(shown).not.toContain('No voiceprint saved');
    expect(shown).not.toContain(MISSING_VOICE_MESSAGE);
  });
});
