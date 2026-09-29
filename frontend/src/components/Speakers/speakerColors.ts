export interface SpeakerColorClasses {
  /** Chip background, text and border. */
  chip: string;
  /** Small solid swatch. */
  dot: string;
}

// Literal class strings: Tailwind only keeps classes it finds verbatim under src/components.
// One entry per colour slot (SPEAKER_COLOR_COUNT in lib/speaker-label).
export const SPEAKER_COLORS: readonly SpeakerColorClasses[] = [
  { chip: 'bg-blue-50 text-blue-700 border-blue-200', dot: 'bg-blue-500' },
  { chip: 'bg-emerald-50 text-emerald-700 border-emerald-200', dot: 'bg-emerald-500' },
  { chip: 'bg-amber-50 text-amber-800 border-amber-200', dot: 'bg-amber-500' },
  { chip: 'bg-purple-50 text-purple-700 border-purple-200', dot: 'bg-purple-500' },
  { chip: 'bg-rose-50 text-rose-700 border-rose-200', dot: 'bg-rose-500' },
  { chip: 'bg-cyan-50 text-cyan-700 border-cyan-200', dot: 'bg-cyan-500' },
  { chip: 'bg-orange-50 text-orange-700 border-orange-200', dot: 'bg-orange-500' },
  { chip: 'bg-slate-100 text-slate-700 border-slate-300', dot: 'bg-slate-500' },
];

export function speakerColorClasses(colorIndex: number): SpeakerColorClasses {
  return SPEAKER_COLORS[colorIndex % SPEAKER_COLORS.length];
}
