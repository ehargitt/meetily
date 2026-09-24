import { afterAll, describe, expect, mock, test } from 'bun:test';
import type { ComponentProps } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import type { MeetingSummary } from '../../src/types';

// Bun shares module mocks between test files; restore the real modules after this suite.
const originalBlockNote = { ...await import('../../src/components/AISummary/BlockNoteSummaryView') };
const originalGeneratorButtons = { ...await import('../../src/components/MeetingDetails/SummaryGeneratorButtonGroup') };
const originalUpdaterButtons = { ...await import('../../src/components/MeetingDetails/SummaryUpdaterButtonGroup') };
const originalPreferences = { ...await import('../../src/lib/summary-language-preferences') };
afterAll(() => {
  mock.module('../../src/components/AISummary/BlockNoteSummaryView', () => originalBlockNote);
  mock.module('../../src/components/MeetingDetails/SummaryGeneratorButtonGroup', () => originalGeneratorButtons);
  mock.module('../../src/components/MeetingDetails/SummaryUpdaterButtonGroup', () => originalUpdaterButtons);
  mock.module('../../src/lib/summary-language-preferences', () => originalPreferences);
});

// The editor and the header buttons need a browser; the panel's own branches do not.
mock.module('../../src/components/AISummary/BlockNoteSummaryView', () => ({ BlockNoteSummaryView: () => <article>saved summary</article> }));
mock.module('../../src/components/MeetingDetails/SummaryGeneratorButtonGroup', () => ({ SummaryGeneratorButtonGroup: () => null }));
mock.module('../../src/components/MeetingDetails/SummaryUpdaterButtonGroup', () => ({ SummaryUpdaterButtonGroup: () => null }));
mock.module('../../src/lib/summary-language-preferences', () => ({
  ...originalPreferences,
  readMeetingSummaryLanguage: async () => ({ language: null, storage: 'metadata' }),
}));

const { SummaryPanel } = await import('../../src/components/MeetingDetails/SummaryPanel');

type SummaryPanelProps = ComponentProps<typeof SummaryPanel>;
const summary = { markdown: 'Alice and Bob agreed to ship.' } as MeetingSummary;
const baseProps: SummaryPanelProps = {
  meeting: { id: 'meeting-a', title: 'Meeting A', created_at: '2026-09-24' },
  meetingTitle: 'Meeting A',
  isSummaryDirty: false,
  summaryRef: { current: null },
  isSaving: false,
  onSaveAll: async () => {},
  onCopySummary: async () => {},
  aiSummary: null,
  summaryStatus: 'idle',
  transcripts: [{ id: 't1', text: 'hello', timestamp: '00:00' }],
  modelConfig: { provider: 'ollama', model: 'm', whisperModel: 'base' },
  setModelConfig: () => {},
  onSaveModelConfig: async () => {},
  onGenerateSummary: async () => {},
  onStopGeneration: () => {},
  customPrompt: '',
  onSaveSummary: async () => {},
  onSummaryChange: () => {},
  onDirtyChange: () => {},
  summaryError: null,
  onRegenerateSummary: async () => {},
  getSummaryStatusMessage: () => '',
  availableTemplates: [],
  selectedTemplate: 'standard_meeting',
  onTemplateSelect: () => {},
};
const render = (props: Partial<SummaryPanelProps>) => renderToStaticMarkup(<SummaryPanel {...baseProps} {...props} />);
const waiting = { isWaitingForSpeakers: true, onGenerateNow: () => {} };
const WAITING_TEXT = 'Waiting for speaker identification';
const HINT_TEXT = 'Regenerate to include speaker names';

describe('SummaryPanel waiting pane', () => {
  test('replaces the empty state while the summary waits for speakers', () => {
    const html = render(waiting);
    expect(html).toContain(WAITING_TEXT);
    expect(html).toContain('Generate now');
  });

  test('is not shown when nothing is waiting', () => {
    expect(render({})).not.toContain(WAITING_TEXT);
  });

  test('gives way to the progress view once the summary is generating', () => {
    const html = render({ ...waiting, summaryStatus: 'processing' });
    expect(html).not.toContain(WAITING_TEXT);
    expect(html).toContain('Generating AI Summary');
  });

  test('gives way to a saved summary', () => {
    const html = render({ ...waiting, aiSummary: summary, summaryStatus: 'completed' });
    expect(html).not.toContain(WAITING_TEXT);
    expect(html).toContain('saved summary');
  });
});

describe('SummaryPanel speaker names hint', () => {
  test('offers to regenerate a saved summary', () => {
    expect(render({ aiSummary: summary, summaryStatus: 'completed', showSpeakerNamesHint: true })).toContain(HINT_TEXT);
  });

  test('stays hidden when not requested or when there is no summary', () => {
    expect(render({ aiSummary: summary, summaryStatus: 'completed' })).not.toContain(HINT_TEXT);
    expect(render({ showSpeakerNamesHint: true })).not.toContain(HINT_TEXT);
  });
});
