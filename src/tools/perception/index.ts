/**
 * Perception tools: AI-driven take selection layer.
 *
 * The base MCP exposes timeline manipulation (cut, paste, razor, remove). These tools
 * add the missing "see and hear what's actually in the footage" layer so Claude can
 * make content-aware editing decisions:
 *
 *   - transcribe_clip       word-level timestamps via local whisper.cpp
 *   - find_filler_words     "um"/"uh"/"you know"/etc. with timestamps
 *   - detect_silences       gaps in speech beyond a threshold
 *   - find_false_starts     explicit retake markers + repeat openings + stutters
 *   - score_takes           rank multiple candidate take ranges
 *   - (next)                apply_cut_list to execute cuts on the Premiere timeline
 *
 * Everything in this module runs locally. No network, no API keys, no per-minute costs.
 */

import { z } from 'zod';
import type { MCPTool } from '../index.js';
import type { PremiereProTransport } from '../../bridge/types.js';
import { Logger } from '../../utils/logger.js';
import { transcribeWithWhisper, getDefaultModelPath } from './whisper.js';
import { TranscriptSchema, CutEntrySchema, type CutEntry } from './types.js';
import { findFillerWords } from './filler.js';
import { detectSilences } from './silence.js';
import { findFalseStarts } from './falseStart.js';
import { scoreTakes, type ScoreTakesOptions } from './takes.js';
import { buildApplyCutListScript } from './cutlist.js';

// ---------- Schemas ----------

const TRANSCRIBE_CLIP_SCHEMA = z.object({
  mediaPath: z
    .string()
    .describe(
      'Absolute path to the source media file (video or audio). Get this from list_project_items or get_metadata for a clip.'
    ),
  language: z
    .string()
    .optional()
    .describe(
      'ISO language code (e.g. "en", "es") or "auto". Defaults to "en". Whisper auto-detection is fine but slightly slower.'
    ),
  modelPath: z
    .string()
    .optional()
    .describe(
      'Override the Whisper model file. Defaults to ggml-large-v3-turbo in the user models dir.'
    ),
  initialPrompt: z
    .string()
    .optional()
    .describe(
      'Optional prompt to bias transcription (e.g. proper nouns, technical terms used in the clip). Improves recognition of names.'
    ),
});

const FIND_FILLER_WORDS_SCHEMA = z.object({
  transcript: TranscriptSchema.describe(
    'A Transcript previously returned by transcribe_clip.'
  ),
  customFillers: z
    .array(z.string())
    .optional()
    .describe(
      'Extra single-word fillers to flag (lowercase, no punctuation). Merged with the default list.'
    ),
  customPhrases: z
    .array(z.string())
    .optional()
    .describe(
      'Extra multi-word filler phrases (lowercase, space-separated). Merged with the default list.'
    ),
  contextWords: z
    .number()
    .int()
    .min(0)
    .max(20)
    .optional()
    .describe('How many surrounding words to include for context. Default 3.'),
  singleWordOnly: z
    .boolean()
    .optional()
    .describe(
      'If true, skip phrase matching and only flag single-word fillers. Default false.'
    ),
});

const DETECT_SILENCES_SCHEMA = z.object({
  transcript: TranscriptSchema,
  minSilenceSec: z
    .number()
    .min(0.1)
    .max(60)
    .optional()
    .describe('Minimum gap (seconds) to count as silence. Default 0.5s.'),
  includeLead: z
    .boolean()
    .optional()
    .describe('Include the silence before the first word. Default true.'),
  includeTrail: z
    .boolean()
    .optional()
    .describe('Include the silence after the last word. Default true.'),
});

const FIND_FALSE_STARTS_SCHEMA = z.object({
  transcript: TranscriptSchema,
  customMarkers: z
    .array(z.string())
    .optional()
    .describe(
      'Extra retake-marker phrases (lowercase, space-separated). Merged with defaults like "let me start over", "scratch that", "from the top".'
    ),
  repeatWindowSec: z
    .number()
    .min(1)
    .max(600)
    .optional()
    .describe(
      'Window (seconds) within which a repeated opening counts as a retake. Default 30s.'
    ),
  minRepeatWords: z
    .number()
    .int()
    .min(2)
    .max(20)
    .optional()
    .describe('Number of opening words required to call something a repeat. Default 5.'),
  stutterMinRun: z
    .number()
    .int()
    .min(2)
    .max(10)
    .optional()
    .describe('Minimum stutter run length (e.g. 2 = "the the"). Default 2.'),
  skipMarkers: z.boolean().optional(),
  skipRepeats: z.boolean().optional(),
  skipStutters: z.boolean().optional(),
});

const TAKE_RANGE_SCHEMA = z.object({
  startSec: z.number().min(0),
  endSec: z.number().min(0),
  label: z.string().optional(),
});

const SCORE_TAKES_SCHEMA = z.object({
  transcript: TranscriptSchema,
  ranges: z
    .array(TAKE_RANGE_SCHEMA)
    .min(1)
    .describe(
      'List of candidate take ranges within the transcript. Typically obtained from find_false_starts or supplied by the user.'
    ),
  weights: z
    .object({
      completeness: z.number().min(0).max(1).optional(),
      smoothness: z.number().min(0).max(1).optional(),
      confidence: z.number().min(0).max(1).optional(),
      cleanliness: z.number().min(0).max(1).optional(),
      pace: z.number().min(0).max(1).optional(),
    })
    .optional()
    .describe('Override scoring weights. Defaults are tuned for talking-head video.'),
});

const APPLY_CUT_LIST_SCHEMA = z.object({
  sequenceId: z
    .string()
    .optional()
    .describe('Sequence ID to apply cuts to. Omit to use the active sequence.'),
  cuts: z
    .array(CutEntrySchema)
    .min(1)
    .describe(
      'Cut entries. Only entries with action="remove" are executed; "keep" entries are passed through for symmetry. Time ranges are in seconds from the start of the sequence.'
    ),
  videoTrackIndices: z
    .array(z.number().int().min(0))
    .optional()
    .describe('Specific video track indices to affect. Omit for all video tracks.'),
  audioTrackIndices: z
    .array(z.number().int().min(0))
    .optional()
    .describe('Specific audio track indices to affect. Omit for all audio tracks.'),
  rippleDelete: z
    .boolean()
    .optional()
    .describe(
      'Close the gap after each cut so subsequent clips slide left. Default true. Set false to "lift" instead (leave the gap).'
    ),
  dryRun: z
    .boolean()
    .optional()
    .describe(
      'When true, runs the razor + clip-discovery logic but skips the actual remove call. Returns the same shape so you can preview which clips would be affected. Default false.'
    ),
});

// ---------- PerceptionTools class ----------

export class PerceptionTools {
  private logger = new Logger('PerceptionTools');
  private bridge: PremiereProTransport | undefined;

  /** Tool names handled by this module — main dispatcher checks against this set. */
  static readonly TOOL_NAMES: ReadonlySet<string> = new Set([
    'transcribe_clip',
    'find_filler_words',
    'detect_silences',
    'find_false_starts',
    'score_takes',
    'apply_cut_list',
  ]);

  constructor(bridge?: PremiereProTransport) {
    this.bridge = bridge;
  }

  getAvailableTools(): MCPTool[] {
    return [
      {
        name: 'transcribe_clip',
        description:
          'Transcribes a media file to word-level timestamps using local Whisper. ' +
          'Returns: { language, duration, words: [{word, start, end, confidence}], text }. ' +
          'Use this as the foundation for take selection — find_filler_words, ' +
          'find_false_starts, detect_silences, and score_takes all consume the resulting ' +
          'transcript. Runs entirely on the local machine — no upload, no API key. ' +
          'Roughly 5-10x realtime on Apple Silicon with the turbo model.',
        inputSchema: TRANSCRIBE_CLIP_SCHEMA,
      },
      {
        name: 'find_filler_words',
        description:
          'Identifies filler words and phrases ("um", "uh", "you know", "i mean", etc.) ' +
          'in a transcript. Returns each hit with its time range, surrounding context, ' +
          'and average confidence so you can decide whether to cut it. Pure logic — no ' +
          'audio re-analysis required. Pass the result of transcribe_clip as the input.',
        inputSchema: FIND_FILLER_WORDS_SCHEMA,
      },
      {
        name: 'detect_silences',
        description:
          'Finds gaps between spoken words longer than a threshold (default 0.5s). ' +
          'Useful for tightening pacing, removing dead air, or finding natural cut ' +
          'points. Operates on transcript word boundaries (fast, no audio re-read) — ' +
          'so non-speech audio inside a "silence" still counts as silence.',
        inputSchema: DETECT_SILENCES_SCHEMA,
      },
      {
        name: 'find_false_starts',
        description:
          'Detects retakes and disfluencies in a transcript using three signals: ' +
          '(1) explicit markers like "let me start over", "scratch that", "from the top"; ' +
          '(2) repeat openings — same N opening words said twice within a window, ' +
          'suggesting a restart; (3) stutters — same word repeated immediately. ' +
          'Returns candidate ranges with kind, confidence, and reason — Claude does ' +
          'the final reasoning before any cut is applied.',
        inputSchema: FIND_FALSE_STARTS_SCHEMA,
      },
      {
        name: 'score_takes',
        description:
          'Given a transcript and a list of candidate take ranges (e.g. multiple ' +
          'attempts at the same line in one continuous clip), scores each on ' +
          'completeness, smoothness, confidence, cleanliness, and pace. Returns ' +
          'a ranked list with breakdowns and a `bestIndex` shortcut to the highest ' +
          'scoring take.',
        inputSchema: SCORE_TAKES_SCHEMA,
      },
      {
        name: 'apply_cut_list',
        description:
          'Executes a cut list on the Premiere timeline. For each entry with ' +
          'action="remove", razors at the entry\'s start and end on the selected ' +
          'tracks (or all tracks by default) and ripple-deletes the resulting clip. ' +
          'Cuts are applied in reverse chronological order so earlier timestamps ' +
          'remain valid throughout. Set dryRun=true for a preview that reports ' +
          'which clips would be affected without modifying the timeline. This is ' +
          'the executor used after Claude proposes cuts derived from find_filler_words, ' +
          'find_false_starts, detect_silences, and your review.',
        inputSchema: APPLY_CUT_LIST_SCHEMA,
      },
    ];
  }

  async executeTool(name: string, args: Record<string, any>): Promise<any> {
    switch (name) {
      case 'transcribe_clip':
        return this.transcribeClip(args as z.infer<typeof TRANSCRIBE_CLIP_SCHEMA>);
      case 'find_filler_words':
        return this.findFillerWords(args as z.infer<typeof FIND_FILLER_WORDS_SCHEMA>);
      case 'detect_silences':
        return this.detectSilences(args as z.infer<typeof DETECT_SILENCES_SCHEMA>);
      case 'find_false_starts':
        return this.findFalseStarts(args as z.infer<typeof FIND_FALSE_STARTS_SCHEMA>);
      case 'score_takes':
        return this.scoreTakes(args as z.infer<typeof SCORE_TAKES_SCHEMA>);
      case 'apply_cut_list':
        return this.applyCutList(args as z.infer<typeof APPLY_CUT_LIST_SCHEMA>);
      default:
        return {
          success: false,
          error: `Unknown perception tool: ${name}`,
          availableTools: Array.from(PerceptionTools.TOOL_NAMES),
        };
    }
  }

  // ---------- Tool impls ----------

  private async transcribeClip(
    args: z.infer<typeof TRANSCRIBE_CLIP_SCHEMA>
  ): Promise<any> {
    this.logger.info(`transcribe_clip: ${args.mediaPath}`);
    const startedAt = Date.now();
    try {
      const opts: Parameters<typeof transcribeWithWhisper>[1] = {
        language: args.language || 'en',
      };
      if (args.modelPath) opts.modelPath = args.modelPath;
      if (args.initialPrompt) opts.initialPrompt = args.initialPrompt;
      const transcript = await transcribeWithWhisper(args.mediaPath, opts);
      const elapsedMs = Date.now() - startedAt;
      this.logger.info(
        `transcribe_clip ok: ${transcript.words.length} words in ${(elapsedMs / 1000).toFixed(1)}s ` +
          `(${(transcript.duration / (elapsedMs / 1000)).toFixed(1)}x realtime)`
      );
      return {
        success: true,
        transcript,
        stats: {
          elapsedMs,
          wordsPerSecond: transcript.words.length / (elapsedMs / 1000),
          realtimeMultiplier: transcript.duration / (elapsedMs / 1000),
        },
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.logger.error(`transcribe_clip failed: ${message}`);
      return {
        success: false,
        error: message,
        hint: message.includes('not found at')
          ? `Default model path: ${getDefaultModelPath()}. Set PREMIERE_MCP_MODELS_DIR to override.`
          : undefined,
      };
    }
  }

  private async findFillerWords(
    args: z.infer<typeof FIND_FILLER_WORDS_SCHEMA>
  ): Promise<any> {
    try {
      const opts: Parameters<typeof findFillerWords>[1] = {};
      if (args.customFillers) opts.customFillers = args.customFillers;
      if (args.customPhrases) opts.customPhrases = args.customPhrases;
      if (typeof args.contextWords === 'number') opts.contextWords = args.contextWords;
      if (typeof args.singleWordOnly === 'boolean') opts.singleWordOnly = args.singleWordOnly;
      const result = findFillerWords(args.transcript, opts);
      return { success: true, ...result };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, error: message };
    }
  }

  private async detectSilences(
    args: z.infer<typeof DETECT_SILENCES_SCHEMA>
  ): Promise<any> {
    try {
      const opts: Parameters<typeof detectSilences>[1] = {};
      if (typeof args.minSilenceSec === 'number') opts.minSilenceSec = args.minSilenceSec;
      if (typeof args.includeLead === 'boolean') opts.includeLead = args.includeLead;
      if (typeof args.includeTrail === 'boolean') opts.includeTrail = args.includeTrail;
      const result = detectSilences(args.transcript, opts);
      return { success: true, ...result };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, error: message };
    }
  }

  private async findFalseStarts(
    args: z.infer<typeof FIND_FALSE_STARTS_SCHEMA>
  ): Promise<any> {
    try {
      const opts: Parameters<typeof findFalseStarts>[1] = {};
      if (args.customMarkers) opts.customMarkers = args.customMarkers;
      if (typeof args.repeatWindowSec === 'number') opts.repeatWindowSec = args.repeatWindowSec;
      if (typeof args.minRepeatWords === 'number') opts.minRepeatWords = args.minRepeatWords;
      if (typeof args.stutterMinRun === 'number') opts.stutterMinRun = args.stutterMinRun;
      if (typeof args.skipMarkers === 'boolean') opts.skipMarkers = args.skipMarkers;
      if (typeof args.skipRepeats === 'boolean') opts.skipRepeats = args.skipRepeats;
      if (typeof args.skipStutters === 'boolean') opts.skipStutters = args.skipStutters;
      const result = findFalseStarts(args.transcript, opts);
      return { success: true, ...result };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, error: message };
    }
  }

  private async scoreTakes(
    args: z.infer<typeof SCORE_TAKES_SCHEMA>
  ): Promise<any> {
    try {
      const opts: ScoreTakesOptions = {};
      if (args.weights) {
        const w: NonNullable<ScoreTakesOptions['weights']> = {};
        if (typeof args.weights.completeness === 'number') w.completeness = args.weights.completeness;
        if (typeof args.weights.smoothness === 'number') w.smoothness = args.weights.smoothness;
        if (typeof args.weights.confidence === 'number') w.confidence = args.weights.confidence;
        if (typeof args.weights.cleanliness === 'number') w.cleanliness = args.weights.cleanliness;
        if (typeof args.weights.pace === 'number') w.pace = args.weights.pace;
        opts.weights = w;
      }
      const cleanRanges = args.ranges.map((r) => {
        const clean: { startSec: number; endSec: number; label?: string } = {
          startSec: r.startSec,
          endSec: r.endSec,
        };
        if (r.label !== undefined) clean.label = r.label;
        return clean;
      });
      const result = scoreTakes(args.transcript, cleanRanges, opts);
      return { success: true, ...result };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { success: false, error: message };
    }
  }

  private async applyCutList(
    args: z.infer<typeof APPLY_CUT_LIST_SCHEMA>
  ): Promise<any> {
    if (!this.bridge) {
      return {
        success: false,
        error:
          'apply_cut_list requires the Premiere bridge but PerceptionTools was constructed without one.',
      };
    }

    // Build the input the script builder expects, only setting fields we
    // actually have so we don't carry undefineds through exactOptionalPropertyTypes.
    const cuts: CutEntry[] = args.cuts.map((c) => {
      const entry: CutEntry = {
        startSec: c.startSec,
        endSec: c.endSec,
        action: c.action,
      };
      if (c.reason !== undefined) entry.reason = c.reason;
      if (c.source !== undefined) entry.source = c.source;
      return entry;
    });

    const removeCount = cuts.filter((c) => c.action === 'remove').length;
    if (removeCount === 0) {
      return {
        success: true,
        message: 'No remove-action cuts provided — nothing to apply.',
        applied: [],
        skipped: [],
        errors: [],
        summary: {
          requested: cuts.length,
          appliedCount: 0,
          skippedCount: 0,
          errorCount: 0,
          totalSecondsCut: 0,
        },
      };
    }

    const scriptInput: Parameters<typeof buildApplyCutListScript>[0] = { cuts };
    if (args.sequenceId !== undefined) scriptInput.sequenceId = args.sequenceId;
    if (args.videoTrackIndices !== undefined) scriptInput.videoTrackIndices = args.videoTrackIndices;
    if (args.audioTrackIndices !== undefined) scriptInput.audioTrackIndices = args.audioTrackIndices;
    if (args.rippleDelete !== undefined) scriptInput.rippleDelete = args.rippleDelete;
    if (args.dryRun !== undefined) scriptInput.dryRun = args.dryRun;

    const script = buildApplyCutListScript(scriptInput);
    this.logger.info(
      `apply_cut_list: ${removeCount} remove-action cuts, dryRun=${args.dryRun ?? false}, ripple=${args.rippleDelete ?? true}`
    );

    try {
      return await this.bridge.executeScript(script);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.logger.error(`apply_cut_list bridge error: ${message}`);
      return { success: false, error: message };
    }
  }
}
