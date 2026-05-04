/**
 * Shared types and Zod schemas for the perception layer.
 *
 * The Whisper wrapper produces a Transcript. The helper tools (filler, silence,
 * false-start, takes) all consume a Transcript and return structured findings
 * that Claude can reason over and turn into cut lists.
 */

import { z } from 'zod';

// ---------- Transcript shape (mirrors whisper.ts output) ----------

export const WhisperWordSchema = z.object({
  word: z.string(),
  start: z.number(),
  end: z.number(),
  confidence: z.number(),
});
export type WhisperWord = z.infer<typeof WhisperWordSchema>;

export const TranscriptSchema = z.object({
  language: z.string(),
  duration: z.number(),
  words: z.array(WhisperWordSchema),
  text: z.string(),
  sourceMedia: z.string(),
  modelPath: z.string(),
});
export type Transcript = z.infer<typeof TranscriptSchema>;

// ---------- Time range with provenance ----------

export const TimeRangeSchema = z.object({
  startSec: z.number(),
  endSec: z.number(),
});
export type TimeRange = z.infer<typeof TimeRangeSchema>;

// ---------- Cut list (consumed by the future apply_cut_list tool) ----------

export const CutActionSchema = z.enum(['remove', 'keep']);
export type CutAction = z.infer<typeof CutActionSchema>;

export const CutEntrySchema = z.object({
  startSec: z.number(),
  endSec: z.number(),
  action: CutActionSchema,
  reason: z.string().optional(),
  source: z
    .enum(['filler', 'silence', 'false_start', 'manual', 'low_confidence'])
    .optional(),
});
export type CutEntry = z.infer<typeof CutEntrySchema>;
