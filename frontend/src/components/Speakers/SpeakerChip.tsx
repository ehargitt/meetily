import { cn } from '@/lib/utils';
import { speakerColorClasses } from './speakerColors';

interface SpeakerChipProps {
  label: string;
  colorIndex: number;
  className?: string;
}

export function SpeakerChip({ label, colorIndex, className }: SpeakerChipProps) {
  const colors = speakerColorClasses(colorIndex);
  return (
    <span
      className={cn(
        'inline-flex max-w-full items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium',
        colors.chip,
        className,
      )}
    >
      <span className={cn('h-2 w-2 flex-shrink-0 rounded-full', colors.dot)} aria-hidden="true" />
      <span className="truncate">{label}</span>
    </span>
  );
}
