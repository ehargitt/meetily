'use client';

import { Loader2, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface WaitingForSpeakersSummaryProps {
  onGenerateNow: () => void;
}

/** Summary pane while the post-recording summary waits for speaker identification. */
export function WaitingForSpeakersSummary({ onGenerateNow }: WaitingForSpeakersSummaryProps) {
  return (
    <div role="status" className="flex flex-col items-center justify-center h-full p-8 text-center">
      <Loader2 className="w-10 h-10 text-blue-500 animate-spin mb-4" />
      <h3 className="text-lg font-semibold text-gray-900 mb-2">
        Waiting for speaker identification…
      </h3>
      <p className="text-sm text-gray-500 mb-6 max-w-md">
        The summary starts on its own once speakers are labeled, so it can say who said what.
      </p>
      <Button variant="outline" onClick={onGenerateNow} className="gap-2">
        <Sparkles className="w-4 h-4" />
        Generate now
      </Button>
    </div>
  );
}
