#!/usr/bin/env bun
// examples/whisper/cli.js
// Whisper speech-to-text CLI powered by Smith (Metal GPU).
//
// Usage:
//   bun examples/whisper/cli.js --model whisper-tiny --file audio.wav
//   bun examples/whisper/cli.js --model whisper-base-en --file audio.wav --language en
//   bun examples/whisper/cli.js --model path/to/ggml-tiny.bin --file audio.wav
//
// Models can be a registry ID (auto-fetched from HF) or a file path.
// Run with --list-models to see available models.

import { parseArgs } from 'util'
import { existsSync } from 'fs'
import { loadAudio } from './audio.js'
import { melSpectrogram } from './mel.js'
import { loadWhisperGGML } from './loader.js'
import { whisperEncode, whisperDecode, whisperTranscribe } from './model.js'
import { createTokenizer, SPECIAL_TOKENS, languageToken } from './tokenizer.js'
import smith from '../../src/index.js'

const { values: args } = parseArgs({
  options: {
    model: { type: 'string', short: 'm' },
    file: { type: 'string', short: 'f' },
    language: { type: 'string', short: 'l', default: 'en' },
    'max-tokens': { type: 'string', default: '224' },
    temperature: { type: 'string', default: '0' },
    output: { type: 'string', short: 'o' },
    format: { type: 'string', default: 'text' },
    verbose: { type: 'boolean', short: 'v', default: false },
    help: { type: 'boolean', short: 'h', default: false },
    'list-models': { type: 'boolean', default: false },
  },
  strict: false,
})

// List available whisper models from registry
if (args['list-models']) {
  const models = smith.listModels().filter(m => m.loader === 'whisper')
  console.log('Available Whisper models:\n')
  for (const m of models) {
    const status = m.cached ? '  [cached]' : ''
    console.log(`  ${m.id.padEnd(20)} ${m.description}${status}`)
  }
  console.log(`\nUsage: bun examples/whisper/cli.js --model <id> --file <audio.wav>`)
  process.exit(0)
}

if (args.help || !args.model || !args.file) {
  console.log(`
Whisper — Speech-to-Text on Apple Silicon GPU

Usage:
  bun examples/whisper/cli.js --model <id|path> --file <audio.wav> [options]

Options:
  -m, --model <id|path>    Model registry ID or path to GGML file (.bin)
  -f, --file <path>        Path to audio file (.wav, 16-bit PCM)
  -l, --language <code>    Language code (default: en)
  --max-tokens <n>         Maximum tokens to generate (default: 224)
  --temperature <t>        Sampling temperature, 0 = greedy (default: 0)
  -o, --output <path>      Write output to file instead of stdout
  --format <fmt>           Output format: text, json, srt, vtt (default: text)
  -v, --verbose            Show timing and model info
  --list-models            List available models from registry
  -h, --help               Show this help

Models:
  Use a registry ID (auto-downloads from HF if not cached):
    whisper-tiny       39M params, multilingual, 75 MB
    whisper-tiny-en    39M params, English only, 75 MB
    whisper-base       74M params, multilingual, 142 MB
    whisper-base-en    74M params, English only, 142 MB
    whisper-small      244M params, multilingual, 466 MB
    whisper-medium     769M params, multilingual, 1.5 GB

  Or pass a direct path to a GGML model file (.bin).
`)
  process.exit(args.help ? 0 : 1)
}

/** Resolve --model to a file path. Registry ID → fetch if needed. File path → use directly. */
async function resolveModelPath(modelArg) {
  // If it looks like a file path (has extension or separator), use directly
  if (existsSync(modelArg) || modelArg.includes('/') || modelArg.includes('.')) {
    return modelArg
  }

  // Try registry
  const entry = smith.getModel(modelArg)
  if (!entry) {
    throw new Error(`Unknown model "${modelArg}". Use --list-models to see available models, or pass a file path.`)
  }

  // Check if cached
  const cached = smith.modelPath(modelArg)
  if (cached) return cached

  // Fetch from HF
  console.error(`Downloading ${entry.description}...`)
  const result = await smith.fetchModel(modelArg, {
    onProgress: (file, downloaded, total) => {
      if (total > 0) {
        const pct = (downloaded / total * 100).toFixed(1)
        const mb = (downloaded / 1e6).toFixed(1)
        process.stderr.write(`\r  ${file}: ${mb} MB (${pct}%)`)
      }
    },
  })
  console.error('')  // newline after progress

  return smith.modelPath(modelArg)
}

async function main() {
  const t0 = performance.now()

  // Resolve model path (registry ID or file path)
  if (args.verbose) console.error('Resolving model...')
  const modelFile = await resolveModelPath(args.model)

  // Load model
  if (args.verbose) console.error(`Loading model from ${modelFile}...`)
  const { model, config, vocab, melFilters, hparams } = await loadWhisperGGML(modelFile)

  if (args.verbose) {
    const info = smith.info()
    console.error(`Device: ${info.device}`)
    console.error(`Model: ${hparams.nAudioLayer} encoder layers, ${hparams.nTextLayer} decoder layers, dim=${hparams.nAudioState}`)
    console.error(`Vocab: ${vocab.length} tokens, ${hparams.nMels} mels`)
  }

  const t1 = performance.now()

  // Load audio
  if (args.verbose) console.error('Loading audio...')
  const { samples, sampleRate } = await loadAudio(args.file)
  if (args.verbose) console.error(`Audio: ${(samples.length / sampleRate).toFixed(1)}s at ${sampleRate}Hz`)

  const t2 = performance.now()

  // Extract mel spectrogram
  if (args.verbose) console.error('Computing mel spectrogram...')
  const { mel, nMels, numFrames } = melSpectrogram(samples, {
    nMels: config.nMels,
    sampleRate,
  })
  if (args.verbose) console.error(`Mel: ${nMels}x${numFrames}`)

  const t3 = performance.now()

  // Create tokenizer
  const tokenizer = createTokenizer(vocab)

  // Transcribe
  if (args.verbose) console.error('Transcribing...')
  const tokens = whisperTranscribe(model, mel, {
    maxTokens: parseInt(args['max-tokens']),
    temperature: parseFloat(args.temperature),
    eotToken: SPECIAL_TOKENS.EOT,
    sotToken: SPECIAL_TOKENS.SOT,
    langToken: languageToken(args.language),
    transcribeToken: SPECIAL_TOKENS.TRANSCRIBE,
    noTimestamps: SPECIAL_TOKENS.NOT,
    onToken: args.verbose ? (token, step) => {
      process.stderr.write('.')
    } : undefined,
  })

  if (args.verbose) console.error('')

  const t4 = performance.now()

  // Decode tokens to text
  const text = tokenizer.decode(tokens)

  // Format output
  let output
  if (args.format === 'json') {
    output = JSON.stringify({
      text: text.trim(),
      tokens,
      language: args.language,
      duration: samples.length / sampleRate,
      timing: args.verbose ? {
        modelLoadMs: t1 - t0,
        audioLoadMs: t2 - t1,
        melMs: t3 - t2,
        transcribeMs: t4 - t3,
        totalMs: t4 - t0,
      } : undefined,
    }, null, 2)
  } else if (args.format === 'srt' || args.format === 'vtt') {
    // Without timestamps, we just output the text as a single segment
    if (args.format === 'vtt') {
      output = `WEBVTT\n\n1\n00:00:00.000 --> ${formatTime(samples.length / sampleRate, args.format)}\n${text.trim()}\n`
    } else {
      output = `1\n00:00:00,000 --> ${formatTime(samples.length / sampleRate, args.format)}\n${text.trim()}\n`
    }
  } else {
    output = text.trim()
  }

  // Write output
  if (args.output) {
    await Bun.write(args.output, output)
    if (args.verbose) console.error(`Written to ${args.output}`)
  } else {
    console.log(output)
  }

  if (args.verbose) {
    console.error(`\nTiming:`)
    console.error(`  Model load: ${(t1 - t0).toFixed(0)}ms`)
    console.error(`  Audio load: ${(t2 - t1).toFixed(0)}ms`)
    console.error(`  Mel spectrogram: ${(t3 - t2).toFixed(0)}ms`)
    console.error(`  Transcription: ${(t4 - t3).toFixed(0)}ms`)
    console.error(`  Total: ${(t4 - t0).toFixed(0)}ms`)
  }
}

function formatTime(seconds, format) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  const ms = Math.floor((seconds % 1) * 1000)
  const sep = format === 'vtt' ? '.' : ','
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}${sep}${String(ms).padStart(3, '0')}`
}

main().catch(err => {
  console.error('Error:', err.message)
  if (args.verbose) console.error(err.stack)
  process.exit(1)
})
