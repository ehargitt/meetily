import { describe, expect, mock, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';
import { act, create } from 'react-test-renderer';
import { WaitingForSpeakersSummary } from '../../src/components/MeetingDetails/WaitingForSpeakersSummary';

describe('summary pane while waiting for speaker identification', () => {
  test('says the summary is waiting and offers to generate it now', async () => {
    const html = renderToStaticMarkup(<WaitingForSpeakersSummary onGenerateNow={() => {}} />);
    expect(html).toContain('role="status"');
    expect(html).toContain('Waiting for speaker identification…');

    const onGenerateNow = mock(() => {});
    const view = create(<WaitingForSpeakersSummary onGenerateNow={onGenerateNow} />);
    const button = view.root.find(node => node.type === 'button');
    expect(button.children.join('')).toContain('Generate now');
    await act(async () => { button.props.onClick(); });
    expect(onGenerateNow).toHaveBeenCalledTimes(1);
  });
});
