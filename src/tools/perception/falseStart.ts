/**
 * False-start / retake detection over a transcript.
 *
 * Looks for three signals talking-head editors care about:
 *
 *   1. EXPLICIT MARKERS — phrases like "wait, let me start over", "scratch that",
 *      "from the top", "let me try again". These are near-certain retake signals.
 *
 *   2. REPEAT OPENINGS — the same N opening words appear twice within a time
 *      window. Strong signal that the speaker restarted the same line. The
 *      earlier occurrence is the candidate to cut.
 *
 *   3. STUTTERS — the same word repeated immediately ("the the", "I-I-I",
 *      "can-can"). Usually a small disfluency to clean up.
 *
 * These are heuristics. Output is always candidates with a confidence score —
 * Claude does the final reasoning before anything gets cut.
 */

import type { Transcript, WhisperWord } from './types.js';

export type FalseStartKind = 'explicit_marker' | 'repeat_opening' | 'stutter';

export interface FalseStartCandidate {
  kind: FalseStartKind;
  /** Range that should typically be removed. */
  startSec: number;
  endSec: number;
  /** Confidence 0..1 — how strongly this looks like a retake/disfluency. */
  confidence: number;
  /** Human-readable explanation Claude can echo to the user. */
  reason: string;
  /** Word index range matched. */
  wordIndices: { start: number; end: number };
  /** The matched text snippet. */
  matchedText: string;
  /** For repeat_opening: the second (kept) occurrence range. */
  alternateOccurrence?: { startSec: number; endSec: number; matchedText: string };
}

export interface FindFalseStartsOptions {
  /** Custom explicit marker phrases (lowercase, space-separated). Merged with defaults. */
  customMarkers?: string[];
  /** Window (sec) within which a repeated opening counts as a retake. Default 30s. */
  repeatWindowSec?: number;
  /** Number of opening words required to consider a repeat. Default 5. */
  minRepeatWords?: number;
  /** Minimum stutter run length (default 2 = same word twice in a row). */
  stutterMinRun?: number;
  /** Skip marker detection. */
  skipMarkers?: boolean;
  /** Skip repeat-opening detection. */
  skipRepeats?: boolean;
  /** Skip stutter detection. */
  skipStutters?: boolean;
}

export interface FindFalseStartsResult {
  candidates: FalseStartCandidate[];
  stats: {
    explicitMarkers: number;
    repeatOpenings: number;
    stutters: number;
    totalSecondsSuggestedCut: number;
  };
}

// Default explicit-marker phrases. Each is space-separated lowercase. Order
// doesn't matter — we pick the longest match at each position.
const DEFAULT_MARKERS: string[] = [
  'let me start over',
  'let me try that again',
  'let me try again',
  'let me redo that',
  'let me redo this',
  'let me do that again',
  'let me do this again',
  'wait let me start over',
  'wait scratch that',
  'scratch that',
  'sorry let me start over',
  'sorry start over',
  'from the top',
  'one more time',
  'take two',
  'take three',
  'ignore that',
  'cut that',
  'do over',
  'redo',
];

const PUNCT_RE = /[^\p{L}\p{N}'’]/gu;
const norm = (w: string): string => w.toLowerCase().replace(PUNCT_RE, '');

export function findFalseStarts(
  transcript: Transcript,
  opts: FindFalseStartsOptions = {}
): FindFalseStartsResult {
  const repeatWindow = opts.repeatWindowSec ?? 30;
  const minRepeatWords = Math.max(2, opts.minRepeatWords ?? 5);
  const stutterMinRun = Math.max(2, opts.stutterMinRun ?? 2);

  const markers = [...DEFAULT_MARKERS, ...(opts.customMarkers || [])]
    .map((m) => m.toLowerCase().trim().split(/\s+/))
    // Longest first so we prefer the most specific match.
    .sort((a, b) => b.length - a.length);

  const words = transcript.words;
  const normWords: string[] = words.map((w) => norm(w.word));
  const consumed = new Array(words.length).fill(false);

  const candidates: FalseStartCandidate[] = [];

  // ---------- 1. Explicit markers ----------
  if (!opts.skipMarkers) {
    for (let i = 0; i < words.length; i++) {
      if (consumed[i]) continue;
      for (const phrase of markers) {
        if (i + phrase.length > words.length) continue;
        let match = true;
        for (let j = 0; j < phrase.length; j++) {
          if (normWords[i + j] !== phrase[j]) {
            match = false;
            break;
          }
        }
        if (match) {
          // The retake range typically extends from the previous sentence-ish
          // boundary up through the marker phrase itself. Heuristic: cut from
          // the start of speech or the most recent "long-ish" gap (>0.6s)
          // before the marker.
          const cutStartIdx = findRetakeStartIdx(words, i, 0.6);
          const startWord = words[cutStartIdx]!;
          const endWord = words[i + phrase.length - 1]!;
          candidates.push({
            kind: 'explicit_marker',
            startSec: startWord.start,
            endSec: endWord.end,
            confidence: 0.95,
            reason: `Explicit retake marker: "${words
              .slice(i, i + phrase.length)
              .map((w) => w.word)
              .join(' ')}"`,
            wordIndices: { start: cutStartIdx, end: i + phrase.length - 1 },
            matchedText: words
              .slice(cutStartIdx, i + phrase.length)
              .map((w) => w.word)
              .join(' '),
          });
          for (let j = cutStartIdx; j < i + phrase.length; j++) consumed[j] = true;
          break;
        }
      }
    }
  }

  // ---------- 2. Repeat openings ----------
  // For every position p, look at the next minRepeatWords words. If those same
  // words appear again starting at some later position q within the repeat
  // window, the earlier (cutStartIdx..p+minRepeatWords-1) range is a candidate.
  if (!opts.skipRepeats) {
    for (let i = 0; i + minRepeatWords < words.length; i++) {
      if (consumed[i]) continue;
      const pattern = normWords.slice(i, i + minRepeatWords);
      // Skip if any pattern slot is empty (e.g., punctuation-only word).
      if (pattern.some((p) => !p)) continue;
      const startWord = words[i]!;
      // Search forward for a non-overlapping repeat.
      for (let j = i + minRepeatWords; j + minRepeatWords <= words.length; j++) {
        const candStart = words[j]!;
        if (candStart.start - startWord.start > repeatWindow) break;
        let match = true;
        for (let k = 0; k < minRepeatWords; k++) {
          if (normWords[j + k] !== pattern[k]) {
            match = false;
            break;
          }
        }
        if (match) {
          // Confidence scales with pattern length and inversely with the gap.
          const gap = words[j]!.start - words[i + minRepeatWords - 1]!.end;
          const conf = clamp(
            0.5 + 0.05 * (minRepeatWords - 5) - 0.01 * gap,
            0.4,
            0.9
          );
          // The earlier occurrence is the cut candidate. End at the word just
          // before the second occurrence so we keep the "good" take.
          const cutStartIdx = findRetakeStartIdx(words, i, 0.6);
          const cutEndIdx = j - 1;
          if (cutEndIdx < cutStartIdx) continue;
          candidates.push({
            kind: 'repeat_opening',
            startSec: words[cutStartIdx]!.start,
            endSec: words[cutEndIdx]!.end,
            confidence: conf,
            reason: `Same opening repeated ~${gap.toFixed(1)}s later — likely a retake.`,
            wordIndices: { start: cutStartIdx, end: cutEndIdx },
            matchedText: words
              .slice(cutStartIdx, cutEndIdx + 1)
              .map((w) => w.word)
              .join(' '),
            alternateOccurrence: {
              startSec: candStart.start,
              endSec: words[j + minRepeatWords - 1]!.end,
              matchedText: words
                .slice(j, j + minRepeatWords)
                .map((w) => w.word)
                .join(' '),
            },
          });
          for (let m = cutStartIdx; m <= cutEndIdx; m++) consumed[m] = true;
          break; // only flag each opening once
        }
      }
    }
  }

  // ---------- 3. Stutters ----------
  if (!opts.skipStutters) {
    for (let i = 0; i < words.length; i++) {
      if (consumed[i]) continue;
      const w = normWords[i];
      if (!w) continue;
      let runEnd = i;
      while (runEnd + 1 < words.length && normWords[runEnd + 1] === w) {
        runEnd++;
      }
      const runLen = runEnd - i + 1;
      if (runLen >= stutterMinRun) {
        // Cut all repetitions except the last.
        const startWord = words[i]!;
        const endWord = words[runEnd - 1]!;
        candidates.push({
          kind: 'stutter',
          startSec: startWord.start,
          endSec: endWord.end,
          confidence: 0.8,
          reason: `Word "${w}" stuttered ${runLen} times.`,
          wordIndices: { start: i, end: runEnd - 1 },
          matchedText: words
            .slice(i, runEnd)
            .map((x) => x.word)
            .join(' '),
        });
        for (let k = i; k < runEnd; k++) consumed[k] = true;
        i = runEnd;
      }
    }
  }

  candidates.sort((a, b) => a.startSec - b.startSec);

  const stats = {
    explicitMarkers: candidates.filter((c) => c.kind === 'explicit_marker').length,
    repeatOpenings: candidates.filter((c) => c.kind === 'repeat_opening').length,
    stutters: candidates.filter((c) => c.kind === 'stutter').length,
    totalSecondsSuggestedCut: candidates.reduce(
      (a, c) => a + (c.endSec - c.startSec),
      0
    ),
  };

  return { candidates, stats };
}

/**
 * Walk backward from `markerIdx` to find the most plausible start of the
 * retake. Heuristic: stop at the first word whose preceding gap is >= `gapSec`
 * (i.e. a clear pause, end of a "thought"). Falls back to the start of the
 * transcript.
 */
function findRetakeStartIdx(
  words: WhisperWord[],
  markerIdx: number,
  gapSec: number
): number {
  for (let k = markerIdx; k > 0; k--) {
    const prev = words[k - 1]!;
    const cur = words[k]!;
    if (cur.start - prev.end >= gapSec) return k;
  }
  return 0;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
