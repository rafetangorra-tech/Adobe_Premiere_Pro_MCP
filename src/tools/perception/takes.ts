/**
 * Score multiple candidate takes within a single transcript and rank them.
 *
 * Use case: speaker recorded the same line several times in one continuous
 * clip. `find_false_starts` (or Claude itself) identifies the take ranges,
 * then this scores each so we can keep the best.
 *
 * Score breakdown (each 0..1, higher = better):
 *   - completeness:  range duration relative to the longest in the group
 *   - smoothness:    1 - (filler-word density inside the range)
 *   - confidence:    average whisper confidence on words inside the range
 *   - cleanliness:   1 - (stutter count / words) inside the range
 *   - pace:          1 if avg-words-per-second is in [1.5, 3.5], otherwise penalty
 *
 * Final score is a weighted sum (weights tunable). Ties go to higher confidence.
 */

import type { Transcript, WhisperWord } from './types.js';
import { findFillerWords } from './filler.js';

export interface TakeRange {
  startSec: number;
  endSec: number;
  /** Optional human-readable label, surfaced in the response. */
  label?: string;
}

export interface ScoreTakesOptions {
  /** Per-component weights (must sum > 0). Defaults are sane for talking-head. */
  weights?: Partial<{
    completeness: number;
    smoothness: number;
    confidence: number;
    cleanliness: number;
    pace: number;
  }>;
}

export interface ScoredTake {
  index: number;
  range: TakeRange;
  score: number;
  rank: number;
  breakdown: {
    completeness: number;
    smoothness: number;
    confidence: number;
    cleanliness: number;
    pace: number;
  };
  details: {
    durationSec: number;
    wordCount: number;
    fillerCount: number;
    stutterCount: number;
    avgConfidence: number;
    wordsPerSecond: number;
  };
}

export interface ScoreTakesResult {
  takes: ScoredTake[];
  /** Convenience: the index of the highest-scoring take. */
  bestIndex: number;
}

const DEFAULT_WEIGHTS = {
  completeness: 0.2,
  smoothness: 0.25,
  confidence: 0.2,
  cleanliness: 0.2,
  pace: 0.15,
};

const PUNCT_RE = /[^\p{L}\p{N}'’]/gu;
const norm = (w: string): string => w.toLowerCase().replace(PUNCT_RE, '');

export function scoreTakes(
  transcript: Transcript,
  ranges: TakeRange[],
  opts: ScoreTakesOptions = {}
): ScoreTakesResult {
  const weights = { ...DEFAULT_WEIGHTS, ...(opts.weights || {}) };
  const totalWeight =
    weights.completeness +
    weights.smoothness +
    weights.confidence +
    weights.cleanliness +
    weights.pace;

  // Pre-compute filler ranges over the whole transcript once.
  const fillerHits = findFillerWords(transcript).hits;

  const wordsByRange: WhisperWord[][] = ranges.map((r) =>
    transcript.words.filter((w) => w.start >= r.startSec && w.end <= r.endSec)
  );

  const longestDuration = ranges.reduce(
    (a, r) => Math.max(a, r.endSec - r.startSec),
    0
  );

  const partial = ranges.map((range, i) => {
    const ws = wordsByRange[i]!;
    const duration = Math.max(0.001, range.endSec - range.startSec);
    const wordCount = ws.length;

    // Count fillers inside this range.
    const fillerCount = fillerHits.filter(
      (h) => h.startSec >= range.startSec && h.endSec <= range.endSec
    ).length;

    // Count immediate-repeat stutters inside this range.
    let stutterCount = 0;
    for (let k = 0; k + 1 < ws.length; k++) {
      if (norm(ws[k]!.word) && norm(ws[k]!.word) === norm(ws[k + 1]!.word)) {
        stutterCount++;
      }
    }

    const avgConfidence =
      ws.length > 0 ? ws.reduce((a, w) => a + w.confidence, 0) / ws.length : 0;
    const wordsPerSecond = wordCount / duration;

    const completeness = longestDuration > 0 ? duration / longestDuration : 1;
    const fillerDensity = wordCount > 0 ? fillerCount / wordCount : 0;
    const smoothness = clamp(1 - fillerDensity * 4, 0, 1); // x4 so 25% fillers → 0
    const confidence = clamp(avgConfidence, 0, 1);
    const stutterDensity = wordCount > 0 ? stutterCount / wordCount : 0;
    const cleanliness = clamp(1 - stutterDensity * 6, 0, 1);
    const pace = paceScore(wordsPerSecond);

    const score =
      (completeness * weights.completeness +
        smoothness * weights.smoothness +
        confidence * weights.confidence +
        cleanliness * weights.cleanliness +
        pace * weights.pace) /
      Math.max(totalWeight, 0.0001);

    return {
      index: i,
      range,
      score,
      rank: 0,
      breakdown: { completeness, smoothness, confidence, cleanliness, pace },
      details: {
        durationSec: duration,
        wordCount,
        fillerCount,
        stutterCount,
        avgConfidence,
        wordsPerSecond,
      },
    } satisfies Omit<ScoredTake, 'rank'> & { rank: number };
  });

  // Rank: highest score = rank 1, with confidence as tiebreaker.
  const ranked = [...partial].sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return b.details.avgConfidence - a.details.avgConfidence;
  });
  ranked.forEach((t, i) => (t.rank = i + 1));

  // Restore original ordering for the return value but keep ranks.
  partial.sort((a, b) => a.index - b.index);
  const bestIndex = ranked[0]?.index ?? -1;

  return { takes: partial, bestIndex };
}

/**
 * Talking-head sweet spot is roughly 2.0–2.8 words/sec. We grade smoothly:
 * full credit inside [1.5, 3.5], decaying outside.
 */
function paceScore(wps: number): number {
  if (wps >= 1.5 && wps <= 3.5) return 1;
  if (wps < 1.5) return clamp(wps / 1.5, 0, 1);
  // wps > 3.5 — too fast.
  return clamp(1 - (wps - 3.5) / 3, 0, 1);
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
