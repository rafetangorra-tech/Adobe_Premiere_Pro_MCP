/**
 * Silence detection over a transcript.
 *
 * "Silence" here = a gap longer than `minSilenceSec` between the end of one
 * word and the start of the next, plus optional lead-in and trail-out silence
 * relative to the transcript's known duration.
 *
 * Limitation: this uses transcript word boundaries, not actual audio energy.
 * That means a non-speech sound (background noise, breath, music sting) inside
 * a "silence" will still be reported as silence. For talking-head footage with
 * a quiet room that's the right approximation 95% of the time. If you need
 * true acoustic silence later, swap in ffmpeg's `silencedetect` filter.
 */

import type { Transcript } from './types.js';

export interface SilenceRange {
  startSec: number;
  endSec: number;
  durationSec: number;
  /** Where in the timeline this silence sits. */
  position: 'lead' | 'middle' | 'trail';
}

export interface DetectSilencesOptions {
  /** Minimum gap (seconds) to count as silence. Default 0.5s. */
  minSilenceSec?: number;
  /** Include the leading silence before the first word. Default true. */
  includeLead?: boolean;
  /** Include the trailing silence after the last word. Default true. */
  includeTrail?: boolean;
}

export interface DetectSilencesResult {
  silences: SilenceRange[];
  stats: {
    count: number;
    totalSilenceSec: number;
    longestSec: number;
    /** Helpful for "what % of the clip is dead air". */
    silenceRatio: number;
  };
}

export function detectSilences(
  transcript: Transcript,
  opts: DetectSilencesOptions = {}
): DetectSilencesResult {
  const minSilence = opts.minSilenceSec ?? 0.5;
  const includeLead = opts.includeLead ?? true;
  const includeTrail = opts.includeTrail ?? true;

  const words = transcript.words;
  const ranges: SilenceRange[] = [];

  // Lead silence: from 0 to first word's start.
  if (includeLead && words.length > 0) {
    const first = words[0]!;
    if (first.start >= minSilence) {
      ranges.push({
        startSec: 0,
        endSec: first.start,
        durationSec: first.start,
        position: 'lead',
      });
    }
  }

  // Mid silences: gaps between consecutive words.
  for (let i = 0; i < words.length - 1; i++) {
    const cur = words[i]!;
    const next = words[i + 1]!;
    const gap = next.start - cur.end;
    if (gap >= minSilence) {
      ranges.push({
        startSec: cur.end,
        endSec: next.start,
        durationSec: gap,
        position: 'middle',
      });
    }
  }

  // Trail silence: from last word's end to transcript.duration.
  if (includeTrail && words.length > 0) {
    const last = words[words.length - 1]!;
    const trail = transcript.duration - last.end;
    if (trail >= minSilence) {
      ranges.push({
        startSec: last.end,
        endSec: transcript.duration,
        durationSec: trail,
        position: 'trail',
      });
    }
  }

  const total = ranges.reduce((a, r) => a + r.durationSec, 0);
  const longest = ranges.reduce((a, r) => Math.max(a, r.durationSec), 0);

  return {
    silences: ranges,
    stats: {
      count: ranges.length,
      totalSilenceSec: total,
      longestSec: longest,
      silenceRatio: transcript.duration > 0 ? total / transcript.duration : 0,
    },
  };
}
