#!/usr/bin/env bun
// examples/tts/cli.js
// TUI for Qwen3-TTS. Type text, hear speech.
// Usage: bun examples/tts/cli.js [model-dir]
// If no dir given, fetches via Smith model registry (models/qwen3-tts/).

import { loadModel } from './model.js'
import { loadTokenizer, SPECIAL_TOKENS } from './tokenizer.js'
import smith from '../../src/index.js'
import { prepareTalker, createTalkerCache, buildInputEmbeds, talkerPrefill, talkerDecode, sampleToken, embed } from './talker.js'
import { preparePredictor, predictCodes, embedCode } from './predictor.js'
import { prepareDecoder, decode } from './decoder.js'
import { playAudio, saveWAV } from './audio.js'
import { fetchModel, modelPath, getModel } from '../../src/models.js'
import { createInterface } from 'node:readline'

const MODEL_ID = 'qwen3-tts'

async function resolveModelDir() {
  // Explicit path given on CLI
  if (process.argv[2]) return process.argv[2]

  // Use Smith model registry
  const entry = getModel(MODEL_ID)
  if (!entry) throw new Error(`Model "${MODEL_ID}" not in registry. Add it to models/registry.json.`)

  if (!entry.cached) {
    console.log('Downloading Qwen3-TTS (~4.5 GB)...')
    const result = await fetchModel(MODEL_ID, {
      onProgress(file, downloaded, total) {
        if (total > 0) {
          const pct = ((downloaded / total) * 100).toFixed(1)
          const mb = (downloaded / 1e6).toFixed(0)
          process.stdout.write(`\r  ${file}: ${mb} MB (${pct}%)`)
        }
      },
    })
    console.log('\nDownload complete.')
    return result.path
  }

  // Already cached — resolve the directory
  const p = modelPath(MODEL_ID, 'config.json')
  if (!p) throw new Error('Model cached but config.json not found')
  return p.replace(/\/config\.json$/, '')
}

async function main() {
  console.log('Qwen3-TTS (Smith)')
  console.log('==================')

  const modelDir = await resolveModelDir()
  console.log(`Model: ${modelDir}\n`)

  // Load model weights
  const t0 = performance.now()
  const model = await loadModel(modelDir)
  const tokenizer = await loadTokenizer(modelDir)
  console.log(`Loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s\n`)

  // Prepare submodules (precompute RoPE etc.)
  prepareTalker(model.talker)
  preparePredictor(model.predictor)
  prepareDecoder(model.decoder)

  // Generation config
  const genConfig = {
    temperature: 0.9,
    topK: 50,
    maxNewTokens: 128,
    repetitionPenalty: 1.05,
  }

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '\n> ',
  })

  console.log('Type text to synthesize (Ctrl+C to quit):')
  rl.prompt()

  rl.on('line', async (line) => {
    const text = line.trim()
    if (!text) { rl.prompt(); return }

    // Handle commands
    if (text.startsWith('/')) {
      if (text.startsWith('/save ')) {
        // Last result will be saved by the generate function
        console.log('(save handled after generation)')
      } else if (text === '/config') {
        console.log(genConfig)
      } else if (text.startsWith('/temp ')) {
        genConfig.temperature = parseFloat(text.slice(6))
        console.log(`temperature = ${genConfig.temperature}`)
      } else if (text.startsWith('/topk ')) {
        genConfig.topK = parseInt(text.slice(6))
        console.log(`topK = ${genConfig.topK}`)
      } else if (text === '/help') {
        console.log('Commands: /config /temp <n> /topk <n> /save <path>')
      }
      rl.prompt()
      return
    }

    try {
      const pcm = await synthesize(text, model, tokenizer, genConfig)
      console.log(`\nPlaying ${(pcm.length / 24000).toFixed(2)}s audio...`)
      await playAudio(pcm)
    } catch (err) {
      console.error('Error:', err.message)
      if (process.env.DEBUG) console.error(err.stack)
    }

    rl.prompt()
  })

  rl.on('close', () => {
    console.log('\nBye.')
    process.exit(0)
  })
}

// Full TTS pipeline: text → codes → waveform
async function synthesize(text, model, tokenizer, config) {
  const { talker, predictor, decoder } = model

  // 1. Tokenize
  const t0 = performance.now()
  const textTokenIds = tokenizer.encodeChat(text)
  console.log(`\nTokenized: ${textTokenIds.length} tokens`)

  // 2. Build input embeddings
  const { inputEmbeds, ttsPadEmbed, totalLen } = buildInputEmbeds(textTokenIds, talker)
  const prefillLen = totalLen
  console.log(`Input embeddings: [${inputEmbeds.shape}]`)

  // 3. Prefill
  const caches = createTalkerCache(talker.config)
  const { logits: prefillLogits, hidden: prefillHidden } = talkerPrefill(talker, inputEmbeds, prefillLen, caches)
  console.log(`Prefill done (${((performance.now() - t0) / 1000).toFixed(2)}s)`)

  // 4. Interleaved decode: talker generates group-0, predictor fills groups 1-15,
  //    all 16 embeddings summed + tts_pad → next talker input.
  //    Reference: codec_hiddens.sum(1) + tts_pad_embed
  const t1 = performance.now()
  const allCodes = [] // [T, 16]
  const generated = []
  const dim = talker.config.dim

  // Sample first group-0 code from prefill
  let code = sampleToken(prefillLogits, config.temperature, config.topK, config.repetitionPenalty, generated)
  generated.push(code)

  // Debug: print first code + logit stats
  {
    const d = prefillLogits.data
    const vocabSize = prefillLogits.shape[1] || prefillLogits.shape[0]
    let max = -Infinity, maxI = 0
    for (let i = 0; i < vocabSize; i++) { if (d[i] > max) { max = d[i]; maxI = i } }
    console.log(`  First code: ${code} (argmax=${maxI}, EOS=${talker.config.codecEosId})`)
  }

  const eosId = talker.config.codecEosId
  const maxTokens = config.maxNewTokens

  // Copy hidden to a fresh Smith tensor (Metal-backed, has strides/buffer for GPU ops)
  function copyHidden(src) {
    const t = smith.zeros(src.shape)
    t.data.set(src.data instanceof Float32Array ? src.data : new Float32Array(src.data))
    return t
  }

  // First step: we have group-0 code + hidden from prefill
  let currentHidden = copyHidden(prefillHidden)

  for (let step = 0; step <= maxTokens; step++) {
    // Run predictor: get groups 1-15 for this step
    const restCodes = predictCodes(predictor, currentHidden, code, talker.codecEmbedding, {
      temperature: config.temperature,
      topK: config.topK,
    })
    allCodes.push([code, ...restCodes])

    if (step === maxTokens) break // don't decode another talker step

    // Build next talker input: sum ALL 16 group embeddings + tts_pad_embed
    // group-0: talker.codecEmbedding[code]
    // groups 1-15: predictor.codecEmbeddings[g-1][restCodes[g-1]]
    const nextInput = smith.zeros([1, dim])

    // Add group-0 from talker embedding
    const g0 = embed([code], talker.codecEmbedding)
    for (let d = 0; d < dim; d++) nextInput.data[d] += g0.data[d]

    // Add groups 1-15 from predictor embeddings
    for (let g = 0; g < 15; g++) {
      const ge = embedCode(restCodes[g], predictor.codecEmbeddings[g])
      for (let d = 0; d < dim; d++) nextInput.data[d] += ge.data[d]
    }

    // Add tts_pad_embed (text stream contribution during decode)
    for (let d = 0; d < dim; d++) nextInput.data[d] += ttsPadEmbed.data[d]

    // Talker decode one step
    const { logits: nextLogits, hidden } = talkerDecode(talker, nextInput, prefillLen + step, caches)
    code = sampleToken(nextLogits, config.temperature, config.topK, config.repetitionPenalty, generated)
    generated.push(code)

    if (code === eosId) {
      console.log(`  EOS at step ${step + 1}`)
      break
    }

    currentHidden = copyHidden(hidden)

    if (step < 10) process.stdout.write(` ${code}`)
    if (step === 10) process.stdout.write('\n')
    if ((step + 1) % 50 === 0) process.stdout.write(`  ${step + 1} codes...`)
  }

  const decodeTime = (performance.now() - t1) / 1000
  console.log(`\nTalker+Predictor: ${allCodes.length} × 16 codes (${decodeTime.toFixed(2)}s, ${(allCodes.length / decodeTime).toFixed(1)} tok/s)`)

  // 5. Speech Decoder: codes → PCM
  const t3 = performance.now()
  const pcm = decode(allCodes, decoder)
  console.log(`Decoder: ${pcm.length} samples (${((performance.now() - t3) / 1000).toFixed(2)}s)`)

  const totalTime = (performance.now() - t0) / 1000
  const audioLen = pcm.length / 24000
  console.log(`Total: ${totalTime.toFixed(2)}s for ${audioLen.toFixed(2)}s audio (${(audioLen / totalTime).toFixed(2)}× realtime)`)

  return pcm
}

main().catch(err => {
  console.error('Fatal:', err)
  process.exit(1)
})
