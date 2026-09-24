import { afterAll, describe, expect, mock, test } from 'bun:test';
import { act, create } from 'react-test-renderer';
import type { UseMeetingSpeakersReturn } from '../../src/hooks/meeting-details/useMeetingSpeakers';

// Bun shares module mocks between test files; restore the real modules other suites use.
// RetranscribeDialog is never loaded for real (it loads the event module); its mock stays registered.
const originalConfig = { ...await import('../../src/contexts/ConfigContext') };
const originalAnalytics = { ...await import('../../src/lib/analytics') };
afterAll(() => {
  mock.module('../../src/contexts/ConfigContext', () => originalConfig);
  mock.module('../../src/lib/analytics', () => originalAnalytics);
});

mock.module('../../src/contexts/ConfigContext', () => ({
  useConfig: () => ({ betaFeatures: { importAndRetranscribe: true } }),
}));
mock.module('../../src/lib/analytics', () => ({ default: { trackButtonClick() {} } }));
let retranscriptionComplete: (() => Promise<void>) | undefined;
mock.module('../../src/components/MeetingDetails/RetranscribeDialog', () => ({
  RetranscribeDialog: ({ onComplete }: { onComplete: () => Promise<void> }) => {
    retranscriptionComplete = onComplete;
    return null;
  },
}));

const { TranscriptButtonGroup } = await import('../../src/components/MeetingDetails/TranscriptButtonGroup');

describe('transcript actions', () => {
  test('an enhanced transcript reloads both the rows and the speaker statistics', async () => {
    const refetchTranscripts = mock(async () => {});
    const refreshSpeakers = mock(async () => {});
    const meetingSpeakers: UseMeetingSpeakersReturn = {
      speakers: [], speakerMap: new Map(), refresh: refreshSpeakers,
      renameSpeaker: async () => {}, setSelf: async () => {}, mergeSpeakers: async () => {},
    };
    const view = create(
      <TranscriptButtonGroup
        transcriptCount={2}
        onCopyTranscript={() => {}}
        onOpenMeetingFolder={async () => {}}
        meetingId="meeting-a"
        meetingFolderPath="/recordings/meeting-a"
        onRefetchTranscripts={refetchTranscripts}
        meetingSpeakers={meetingSpeakers}
      />,
    );
    await act(async () => { await retranscriptionComplete!(); });
    expect(refetchTranscripts).toHaveBeenCalledTimes(1);
    expect(refreshSpeakers).toHaveBeenCalledTimes(1);
    view.unmount();
  });
});
