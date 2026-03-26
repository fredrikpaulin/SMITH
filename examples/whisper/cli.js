#!/usr/bin/env bun
// examples/whisper/cli.js
// Whisper speech-to-text CLI powered by Smith (Metal GPU).
//
// Usage:
//   bun examples/whisper/cli.js --model ggml-tiny.bin --file audio.wav
//   bun examples/whisper/cli.js --model ggml-base.bin --file audio.wav --language en
//
// Requires a whisper.cpp GGML model file (.bin) from:
//   https://huggingface.co/ggerganov/whisper.cpp

import { parseArgs } from 'util'
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
  },
  strict: false,
})

if (args.help || !args.model || !args.file) {
  console.log(`
Whisper — Speech-to-Text on Apple Silicon GPU

Usage:
  bun examples/whisper/cli.js --model <model.bin> --file <audio.wav> [options]

Options:
  -m, --model <path>       Path to whisper.cpp GGML model (.bin)
  -f, --file <path>        Path to audio file (.wav, 16-bit PCM)
  -l, --language <code>    Language code (default: en)
  --max-tokens <n>         Maximum tokens to generate (default: 224)
  --temperature <t>        Sampling temperature, 0 = greedy (default: 0)
  -o, --output <path>      Write output to file instead of stdout
  --format <fmt>           Output format: text, json, srt, vtt (default: text)
  -v, --verbose            Show timing and model info
  -h, --help               Show this help

Models:
  Download GGML models from https://huggingface.co/ggerganov/whisper.cpp
  Recommended: ggml-tiny.bin (75MB) or ggml-base.en.bin (142MB)
`)
  process.exit(args.help ? 0 : 1)
}

async function main() {
  const t0 = performance.now()

  // Load model
  if (args.verbose) console.error('Loading model...')
  const { model, config, vocab, melFilters, hparams } = await loadWhisperGGML(args.model)

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
