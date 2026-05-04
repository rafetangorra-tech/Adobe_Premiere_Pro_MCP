/**
 * Whisper.cpp wrapper for word-level transcription.
 *
 * Pipeline:
 *   media file → ffmpeg (16kHz mono pcm_s16le wav) → whisper-cli (-ojf -ml 1 -sow)
 *               → parse JSON → normalized Transcript with word-level timestamps.
 *
 * Everything runs locally. No network, no API keys, no per-minute charges.
 */

import { spawn } from 'node:child_process';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Logger } from '../../utils/logger.js';
import type { Transcript, WhisperWord } from './types.js';

export type { Transcript, WhisperWord } from './types.js';

const log = new Logger('Whisper');

export interface TranscribeOptions {
  /** Override model path. Defaults to large-v3-turbo from the user's models dir. */
  modelPath?: string;
  /** ISO language code or 'auto'. Defaults to 'en'. */
  language?: string;
  /** Number of CPU threads. Defaults to (cpu count - 2, min 4). */
  threads?: number;
  /** Initial prompt to bias the model (e.g. proper nouns in your script). */
  initialPrompt?: string;
}

// ---------- Defaults ----------

export function getModelsDir(): string {
  return (
    process.env.PREMIERE_MCP_MODELS_DIR ||
    path.join(os.homedir(), 'Library', 'Application Support', 'PremiereProMCP', 'models')
  );
}

export function getDefaultModelPath(): string {
  return path.join(getModelsDir(), 'ggml-large-v3-turbo.bin');
}

const ENDOFTEXT_RE = /^<\|.*\|>$/;

// ---------- Public API ----------

export async function transcribeWithWhisper(
  mediaPath: string,
  opts: TranscribeOptions = {}
): Promise<Transcript> {
  const absMedia = path.resolve(mediaPath);
  const modelPath = opts.modelPath || getDefaultModelPath();

  // Preflight checks fail fast with actionable errors.
  await assertExists(modelPath, () =>
    `Whisper model not found at ${modelPath}. Download with:\n` +
    `  curl -fL -o "${modelPath}" \\\n` +
    `    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin`
  );
  await assertExists(absMedia, () => `Media file not found: ${absMedia}`);

  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'mcp-whisper-'));
  const wavPath = path.join(tmpDir, 'audio.wav');
  const jsonPath = wavPath + '.json'; // whisper-cli writes <input>.json next to the input

  try {
    log.info(`transcribe: extracting audio: ${absMedia} → ${wavPath}`);
    await runFFmpeg(absMedia, wavPath);

    log.info(`transcribe: running whisper-cli (model: ${path.basename(modelPath)})`);
    await runWhisperCli(wavPath, modelPath, opts);

    const raw = JSON.parse(await fs.readFile(jsonPath, 'utf8'));
    return parseWhisperJson(raw, absMedia, modelPath);
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
}

// ---------- Internals ----------

async function assertExists(p: string, errMsg: () => string): Promise<void> {
  try {
    await fs.access(p);
  } catch {
    throw new Error(errMsg());
  }
}

function runFFmpeg(input: string, output: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const ff = spawn(
      'ffmpeg',
      ['-y', '-loglevel', 'error', '-i', input, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', output],
      { stdio: ['ignore', 'pipe', 'pipe'] }
    );
    let stderr = '';
    ff.stderr.on('data', (d) => (stderr += d.toString()));
    ff.on('error', reject);
    ff.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg exited ${code}: ${stderr.slice(-800)}`));
    });
  });
}

function runWhisperCli(wavPath: string, modelPath: string, opts: TranscribeOptions): Promise<void> {
  const threads = opts.threads ?? Math.max(4, os.cpus().length - 2);
  const args = [
    '-m', modelPath,
    '-f', wavPath,
    '-ojf', // output JSON full
    '-ml', '1', // max-len 1 char per segment → near-word splitting
    '-sow', // split on word
    '-nt', // suppress stdout timestamp prints (we read JSON)
    '-t', String(threads),
    '-l', opts.language || 'en',
  ];
  if (opts.initialPrompt) {
    args.push('--prompt', opts.initialPrompt);
  }

  return new Promise((resolve, reject) => {
    const w = spawn('whisper-cli', args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stderr = '';
    w.stderr.on('data', (d) => (stderr += d.toString()));
    w.on('error', (e: NodeJS.ErrnoException) => {
      if (e.code === 'ENOENT') {
        reject(
          new Error(
            `whisper-cli not found on PATH. Install with: brew install whisper-cpp`
          )
        );
      } else {
        reject(e);
      }
    });
    w.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`whisper-cli exited ${code}: ${stderr.slice(-800)}`));
    });
  });
}

interface RawSegment {
  text: string;
  offsets: { from: number; to: number };
  tokens?: Array<{ text: string; offsets: { from: number; to: number }; p?: number }>;
}

function parseWhisperJson(raw: any, sourceMedia: string, modelPath: string): Transcript {
  const segments = (raw?.transcription || []) as RawSegment[];
  const language = raw?.result?.language || 'en';

  const words: WhisperWord[] = [];

  for (const seg of segments) {
    const text = (seg.text || '').trim();
    if (!text) continue;
    if (ENDOFTEXT_RE.test(text)) continue;

    // Compute confidence from the non-special tokens of this segment.
    const tokens = (seg.tokens || []).filter(
      (t) => !ENDOFTEXT_RE.test((t.text || '').trim())
    );
    const probs = tokens.map((t) => t.p).filter((p): p is number => typeof p === 'number');
    const confidence = probs.length > 0 ? probs.reduce((a, b) => a + b, 0) / probs.length : 0;

    // Token offsets are tighter than segment offsets; prefer them when available.
    const firstTok = tokens[0];
    const lastTok = tokens[tokens.length - 1];
    const startMs = firstTok ? firstTok.offsets.from : seg.offsets.from;
    const endMs = lastTok ? lastTok.offsets.to : seg.offsets.to;

    words.push({
      word: text,
      start: startMs / 1000,
      end: endMs / 1000,
      confidence,
    });
  }

  // Concatenate raw text. Whisper's segment text already includes leading spaces
  // where appropriate, so trimming each word and joining with spaces gives
  // readable output.
  const text = words.map((w) => w.word).join(' ');
  const lastWord = words[words.length - 1];
  const duration = lastWord ? lastWord.end : 0;

  return {
    language,
    duration,
    words,
    text,
    sourceMedia,
    modelPath,
  };
}
