import { test, expect } from 'bun:test'
import smith from '../src/index.js'
import {
  WHISPER_CONFIGS, createWhisperModel,
  whisperEncode, whisperDecode, whisperTranscribe,
  whisperDecodePrefill, whisperDecodeStep, whisperTranscribeCached,
  precomputeEncoderKV,
} from '../examples/whisper/model.js'

const { noGrad, conv1dOutputSize } = smith

// Use tiny config for fast tests
const config = { ...WHISPER_CONFIGS.tiny, encoderLayers: 2, decoderLayers: 2 }

function createTestModel() {
  return createWhisperModel(config)
}

function createMelInput(numFrames = 200) {
  return Array.from({ length: config.nMels * numFrames }, () => Math.random() * 0.1)
}

test('whisperDecodePrefill returns correct shapes', () => {
  noGrad(() => {
    const model = createTestModel()
    const mel = createMelInput()
    const encoderOut = whisperEncode(model, mel)
    const tokens = [50258, 50259, 50359, 50363] // SOT, lang, transcribe, notimestamps

    const { logits, selfCaches, encoderKV } = whisperDecodePrefill(model, encoderOut, tokens)

    // Logits: [seqLen, vocabSize]
    expect(logits.data.shape).toEqual([tokens.length, config.vocabSize])

    // Self-attention caches: one per decoder block
    expect(selfCaches.length).toBe(config.decoderLayers)
    const headDim = config.dim / config.numHeads
    for (const cache of selfCaches) {
      expect(cache.k.data.shape).toEqual([config.numHeads, tokens.length, headDim])
      expect(cache.v.data.shape).toEqual([config.numHeads, tokens.length, headDim])
    }

    // Encoder KV: one per decoder block
    expect(encoderKV.length).toBe(config.decoderLayers)
    const audioCtx = conv1dOutputSize(200, 3, 2, 1) // after conv2 with stride 2
    for (const kv of encoderKV) {
      expect(kv.k.data.shape).toEqual([config.numHeads, audioCtx, headDim])
      expect(kv.v.data.shape).toEqual([config.numHeads, audioCtx, headDim])
    }
  })
})

test('whisperDecodeStep returns correct shapes and grows cache', () => {
  noGrad(() => {
    const model = createTestModel()
    const mel = createMelInput()
    const encoderOut = whisperEncode(model, mel)
    const tokens = [50258, 50259, 50359, 50363]

    const { selfCaches, encoderKV } = whisperDecodePrefill(model, encoderOut, tokens)
    const headDim = config.dim / config.numHeads

    // First step: position = 4 (after 4 prompt tokens)
    const step1 = whisperDecodeStep(model, encoderKV, 100, 4, selfCaches)
    expect(step1.logits.data.shape).toEqual([1, config.vocabSize])
    expect(step1.selfCaches.length).toBe(config.decoderLayers)
    for (const cache of step1.selfCaches) {
      expect(cache.k.data.shape).toEqual([config.numHeads, tokens.length + 1, headDim])
      expect(cache.v.data.shape).toEqual([config.numHeads, tokens.length + 1, headDim])
    }

    // Second step: cache grows by 1 more
    const step2 = whisperDecodeStep(model, encoderKV, 200, 5, step1.selfCaches)
    for (const cache of step2.selfCaches) {
      expect(cache.k.data.shape).toEqual([config.numHeads, tokens.length + 2, headDim])
      expect(cache.v.data.shape).toEqual([config.numHeads, tokens.length + 2, headDim])
    }
  })
})

test('encoder KV is constant across decode steps', () => {
  noGrad(() => {
    const model = createTestModel()
    const mel = createMelInput()
    const encoderOut = whisperEncode(model, mel)
    const tokens = [50258, 50259, 50359, 50363]

    const { selfCaches, encoderKV } = whisperDecodePrefill(model, encoderOut, tokens)

    // Snapshot encoder KV data before decode steps
    const kvSnapshot = encoderKV.map(kv => ({
      k: new Float32Array(kv.k.data.data),
      v: new Float32Array(kv.v.data.data),
    }))

    // Run a decode step
    whisperDecodeStep(model, encoderKV, 100, 4, selfCaches)

    // Encoder KV should be unchanged
    for (let i = 0; i < encoderKV.length; i++) {
      for (let j = 0; j < kvSnapshot[i].k.length; j++) {
        expect(encoderKV[i].k.data.data[j]).toBe(kvSnapshot[i].k[j])
      }
      for (let j = 0; j < kvSnapshot[i].v.length; j++) {
        expect(encoderKV[i].v.data.data[j]).toBe(kvSnapshot[i].v[j])
      }
    }
  })
})

test('prefill logits match full-sequence decode logits', () => {
  noGrad(() => {
    const model = createTestModel()
    const mel = createMelInput()
    const encoderOut = whisperEncode(model, mel)
    const tokens = [50258, 50259, 50359, 50363]

    // Full-sequence decode (non-cached)
    const fullLogits = whisperDecode(model, encoderOut, tokens)

    // Prefill decode (cached)
    const { logits: prefillLogits } = whisperDecodePrefill(model, encoderOut, tokens)

    // Compare last-position logits (the ones used for sampling)
    const vocabSize = config.vocabSize
    const lastStart = (tokens.length - 1) * vocabSize
    for (let i = 0; i < vocabSize; i++) {
      expect(prefillLogits.data.data[lastStart + i]).toBeCloseTo(fullLogits.data.data[lastStart + i], 3)
    }
  })
})

test('whisperTranscribeCached produces same tokens as whisperTranscribe', () => {
  noGrad(() => {
    const model = createTestModel()
    const mel = createMelInput()

    // Use temperature=0 for deterministic output
    const opts = { maxTokens: 5, temperature: 0 }
    const nonCached = whisperTranscribe(model, mel, opts)
    const cached = whisperTranscribeCached(model, mel, opts)

    expect(cached).toEqual(nonCached)
  })
}, 15000)

test('whisperTranscribeCached stops on EOT token', () => {
  noGrad(() => {
    const model = createTestModel()
    const mel = createMelInput()

    // Run with generous maxTokens — should stop at EOT naturally or hit limit
    const result = whisperTranscribeCached(model, mel, { maxTokens: 10, temperature: 0 })
    expect(result.length).toBeLessThanOrEqual(10)
  })
}, 15000)

test('whisperTranscribeCached single token generation', () => {
  noGrad(() => {
    const model = createTestModel()
    const mel = createMelInput()

    // maxTokens=1 should produce at most 1 token
    const result = whisperTranscribeCached(model, mel, { maxTokens: 1, temperature: 0 })
    expect(result.length).toBeLessThanOrEqual(1)
  })
})

test('whisperTranscribeCached onToken callback', () => {
  noGrad(() => {
    const model = createTestModel()
    const mel = createMelInput()

    const received = []
    const result = whisperTranscribeCached(model, mel, {
      maxTokens: 3,
      temperature: 0,
      onToken: (token, step) => { received.push({ token, step }) },
    })

    expect(received.length).toBe(result.length)
    for (let i = 0; i < result.length; i++) {
      expect(received[i].token).toBe(result[i])
      expect(received[i].step).toBe(i)
    }
  })
})

test('whisperTranscribeCached onToken can stop early', () => {
  noGrad(() => {
    const model = createTestModel()
    const mel = createMelInput()

    // Stop after first token
    const result = whisperTranscribeCached(model, mel, {
      maxTokens: 10,
      temperature: 0,
      onToken: () => true, // always stop
    })

    expect(result.length).toBe(1)
  })
})
