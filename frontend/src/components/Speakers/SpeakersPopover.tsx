import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ChevronDown, GitMerge } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { MeetingSpeaker } from '@/types';
import { speakerColorIndex, speakerLabel } from '@/lib/speaker-label';
import { speakerColorClasses } from './speakerColors';
import {
  MISSING_VOICE_MESSAGE,
  SHORT_VOICEPRINT_MESSAGE,
} from '@/hooks/meeting-details/useMeetingSpeakers';
import { cn } from '@/lib/utils';

interface SpeakersPopoverProps {
  speakers: MeetingSpeaker[];
  onRename: (speakerKey: string, displayName: string) => Promise<void>;
  onSetSelf: (speakerKey: string, isSelf: boolean) => Promise<void>;
  onMerge: (fromKey: string, intoKey: string) => Promise<void>;
  /** The button that opens the popover. */
  children: ReactNode;
}

export function SpeakersPopover({ speakers, onRename, onSetSelf, onMerge, children }: SpeakersPopoverProps) {
  const selfSpeaker = speakers.find(speaker => speaker.is_self);

  return (
    <Popover>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent align="end" className="w-80 p-0">
        <div className="border-b px-4 py-3">
          <p className="text-sm font-semibold">Speakers</p>
          <p className="text-xs text-muted-foreground">Names apply to this meeting only. Leave a name empty to reset it.</p>
          <p className="mt-1 text-xs text-muted-foreground">
            &quot;This is me&quot; saves a voiceprint on this computer so later meetings can label you automatically.
          </p>
        </div>
        {speakers.length === 0 ? (
          <p className="px-4 py-6 text-center text-sm text-muted-foreground">No speakers identified yet.</p>
        ) : (
          <ul className="max-h-96 divide-y overflow-y-auto">
            {speakers.map(speaker => (
              <SpeakerRow
                key={speaker.speaker_key}
                speaker={speaker}
                others={speakers.filter(other => other.speaker_key !== speaker.speaker_key)}
                showSelfSuggestion={!selfSpeaker && Boolean(speaker.suggested_self)}
                onRename={onRename}
                onSetSelf={onSetSelf}
                onMerge={onMerge}
              />
            ))}
          </ul>
        )}
      </PopoverContent>
    </Popover>
  );
}

interface SpeakerRowProps {
  speaker: MeetingSpeaker;
  others: MeetingSpeaker[];
  showSelfSuggestion: boolean;
  onRename: SpeakersPopoverProps['onRename'];
  onSetSelf: SpeakersPopoverProps['onSetSelf'];
  onMerge: SpeakersPopoverProps['onMerge'];
}

function SpeakerRow({ speaker, others, showSelfSuggestion, onRename, onSetSelf, onMerge }: SpeakerRowProps) {
  const savedName = speaker.display_name ?? '';
  const [draft, setDraft] = useState(savedName);
  const [isSaving, setIsSaving] = useState(false);
  // Escape also closes the popover, and the blur that follows must not save the discarded draft.
  const discardDraftRef = useRef(false);
  useEffect(() => setDraft(savedName), [savedName]);

  const key = speaker.speaker_key;
  const colors = speakerColorClasses(speakerColorIndex(key, speaker));
  const defaultLabel = speakerLabel(key, { display_name: null, is_self: speaker.is_self });

  const run = async (action: () => Promise<void>) => {
    setIsSaving(true);
    try {
      await action();
    } finally {
      setIsSaving(false);
    }
  };

  const commitName = () => {
    if (discardDraftRef.current) {
      discardDraftRef.current = false;
      return;
    }
    if (draft.trim() === savedName.trim()) return;
    void run(() => onRename(key, draft));
  };

  return (
    <li className="space-y-2 px-4 py-3">
      <div className="flex items-center gap-2">
        <span className={cn('h-3 w-3 flex-shrink-0 rounded-full', colors.dot)} aria-hidden="true" />
        <Input
          value={draft}
          placeholder={defaultLabel}
          aria-label={`Name for ${speakerLabel(key, speaker)}`}
          disabled={isSaving}
          className="h-8"
          onChange={event => setDraft(event.target.value)}
          onBlur={commitName}
          onKeyDown={event => {
            if (event.key === 'Enter') event.currentTarget.blur();
            if (event.key === 'Escape') {
              discardDraftRef.current = true;
              setDraft(savedName);
              event.currentTarget.blur();
            }
          }}
        />
      </div>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>
          {formatTalkTime(speaker.talk_time_seconds)} talk time · {speaker.segment_count} segment{speaker.segment_count === 1 ? '' : 's'}
        </span>
        {others.length > 0 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" disabled={isSaving}>
                <GitMerge className="h-3 w-3" />
                Merge into
                <ChevronDown className="h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel className="text-xs">Same person as...</DropdownMenuLabel>
              {others.map(other => (
                <DropdownMenuItem
                  key={other.speaker_key}
                  onSelect={() => void run(() => onMerge(key, other.speaker_key))}
                >
                  {speakerLabel(other.speaker_key, other)}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>
      <label className="flex items-center justify-between text-sm">
        <span>This is me</span>
        <Switch
          checked={speaker.is_self}
          disabled={isSaving}
          onCheckedChange={checked => void run(() => onSetSelf(key, checked))}
        />
      </label>
      {speaker.voiceprint === 'too_short' && (
        <p className="text-xs text-muted-foreground">{SHORT_VOICEPRINT_MESSAGE}</p>
      )}
      {speaker.is_self && speaker.voiceprint === 'voice_missing' && (
        <p className="text-xs text-muted-foreground">This voice is not in your voiceprint. {MISSING_VOICE_MESSAGE}.</p>
      )}
      {speaker.is_self && speaker.voiceprint === 'not_saved' && (
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>No voiceprint saved.</span>
          <Button
            variant="outline"
            size="sm"
            className="h-6 px-2 text-xs"
            disabled={isSaving}
            onClick={() => void run(() => onSetSelf(key, true))}
          >
            Save voiceprint
          </Button>
        </div>
      )}
      {showSelfSuggestion && (
        <div className="flex items-center justify-between rounded-md border border-blue-200 bg-blue-50 px-2 py-1.5 text-xs text-blue-800">
          <span>This sounds like you. Is this you?</span>
          <Button
            variant="outline"
            size="sm"
            className="h-6 px-2 text-xs"
            disabled={isSaving}
            onClick={() => void run(() => onSetSelf(key, true))}
          >
            Yes, it&apos;s me
          </Button>
        </div>
      )}
    </li>
  );
}

function formatTalkTime(seconds: number): string {
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = (total % 60).toString().padStart(2, '0');
  return hours > 0 ? `${hours}:${minutes.toString().padStart(2, '0')}:${secs}` : `${minutes}:${secs}`;
}
