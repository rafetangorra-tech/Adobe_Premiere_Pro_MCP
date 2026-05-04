/**
 * Filler-word detection over a transcript.
 *
 * Returns word-level ranges Claude can either auto-cut or surface for review.
 * Conservative default list — opinionated against false positives. Users can
 * supply a custom list via the tool's `customFillers` parameter to expand it.
 */

import type { Transcript, WhisperWord } from './types.js';

// Word-level fillers: matched as a complete word (after stripping punctuation),
// case-insensitive. These are nearly always fillers in spoken English.
const DEFAULT_SINGLE_WORD_FILLERS: ReadonlySet<string> = new Set([
  'um',
  'umm',
  'uhh',
  'uh',
  'er',
  'erm',
  'ahh',
  'ah',
  'mm',
  'mmm',
  'hmm',
  'hm',
]);

// Multi-word fillers: matched as consecutive words, case-insensitive.
// Conservative — only phrases that are nearly always filler.
const DEFAULT_PHRASE_FILLERS: ReadonlyArray<string[]> = [
  ['you', 'know'],
  ['i', 'mean'],
  ['kind', 'of'],
  ['sort', 'of'],
];

export interface FillerHit {
  startSec: number;
  endSec: number;
  /** The matched filler text as it appeared (e.g. "Um," or "you know"). */
  match: string;
  /** Normalized form (lowercase, no punctuation, e.g. "um" or "you know"). */
  normalized: string;
  /** Index range of words in the transcript that matched. */
  wordIndices: { start: number; end: number };
  /** Surrounding words for Claude to evaluate context. */
  contextBefore: string;
  contextAfter: string;
  /** Lower confidence on the matched words = more likely a true filler. */
  averageConfidence: number;
}

export interface FindFillerWordsOptions {
  /** Extra single words to flag (lowercase, no punctuation). */
  customFillers?: string[];
  /** Extra multi-word phrases (each entry is space-separated, lowercase). */
  customPhrases?: string[];
  /** Number of context words to include on each side. Defaults to 3. */
  contextWords?: number;
  /** If true, skip phrase matching and only flag single-word fillers. */
  singleWordOnly?: boolean;
}

export interface FindFillerWordsResult {
  hits: FillerHit[];
  stats: {
    totalHits: number;
    singleWordHits: number;
    phraseHits: number;
    /** Total seconds of audio across all hits — useful for "how much will I save". */
    totalSecondsCut: number;
    /** Tally by normalized form. */
    byNormalizedForm: Record<string, number>;
  };
}

const PUNCT_RE = /[^\p{L}\p{N}'’]/gu;

function normalize(word: string): string {
  return word.toLowerCase().replace(PUNCT_RE, '');
}

export function findFillerWords(
  transcript: Transcript,
  opts: FindFillerWordsOptions = {}
): FindFillerWordsResult {
  const contextN = opts.contextWords ?? 3;
  const singles: ReadonlySet<string> = new Set([
    ...DEFAULT_SINGLE_WORD_FILLERS,
    ...(opts.customFillers || []).map((s) => s.toLowerCase().trim()),
  ]);
  const phrases: string[][] = opts.singleWordOnly
    ? []
    : [
        ...DEFAULT_PHRASE_FILLERS.map((p) => p.slice()),
        ...(opts.customPhrases || []).map((s) => s.toLowerCase().trim().split(/\s+/)),
      ];

  const words = transcript.words;
  const normalized: string[] = words.map((w) => normalize(w.word));
  const consumed = new Array(words.length).fill(false);

  const hits: FillerHit[] = [];

  // Phrase matches first (longer match wins) so single-word doesn't shadow them.
  for (let i = 0; i < words.length; i++) {
    if (consumed[i]) continue;
    for (const phrase of phrases) {
      if (i + phrase.length > words.length) continue;
      let match = true;
      for (let j = 0; j < phrase.length; j++) {
        if (normalized[i + j] !== phrase[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        const startWord = words[i];
        const endWord = words[i + phrase.length - 1];
        if (!startWord || !endWord) continue;
        const matchedWords = words.slice(i, i + phrase.length);
        hits.push(buildHit(words, i, i + phrase.length - 1, contextN, matchedWords, phrase.join(' ')));
        for (let j = 0; j < phrase.length; j++) consumed[i + j] = true;
        break;
      }
    }
  }

  // Single-word matches.
  for (let i = 0; i < words.length; i++) {
    if (consumed[i]) continue;
    const norm = normalized[i];
    if (norm && singles.has(norm)) {
      hits.push(buildHit(words, i, i, contextN, [words[i]!], norm));
      consumed[i] = true;
    }
  }

  hits.sort((a, b) => a.startSec - b.startSec);

  const byForm: Record<string, number> = {};
  let phraseHits = 0;
  let singleHits = 0;
  let totalCut = 0;
  for (const h of hits) {
    byForm[h.normalized] = (byForm[h.normalized] || 0) + 1;
    totalCut += h.endSec - h.startSec;
    if (h.normalized.includes(' ')) phraseHits++;
    else singleHits++;
  }

  return {
    hits,
    stats: {
      totalHits: hits.length,
      singleWordHits: singleHits,
      phraseHits,
      totalSecondsCut: totalCut,
      byNormalizedForm: byForm,
    },
  };
}

function buildHit(
  words: WhisperWord[],
  startIdx: number,
  endIdx: number,
  contextN: number,
  matched: WhisperWord[],
  normalized: string
): FillerHit {
  const startWord = matched[0]!;
  const endWord = matched[matched.length - 1]!;
  const before = words.slice(Math.max(0, startIdx - contextN), startIdx);
  const after = words.slice(endIdx + 1, Math.min(words.length, endIdx + 1 + contextN));
  const confSum = matched.reduce((a, w) => a + w.confidence, 0);
  return {
    startSec: startWord.start,
    endSec: endWord.end,
    match: matched.map((w) => w.word).join(' '),
    normalized,
    wordIndices: { start: startIdx, end: endIdx },
    contextBefore: before.map((w) => w.word).join(' '),
    contextAfter: after.map((w) => w.word).join(' '),
    averageConfidence: matched.length > 0 ? confSum / matched.length : 0,
  };
}
