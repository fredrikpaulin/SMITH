import { test, expect } from 'bun:test'
import smith from '../../../src/index.js'
import {
  WHISPER_CONFIGS, createWhisperModel,
  createEncoderBlock, encoderBlock,
  createDecoderBlock, decoderBlock,
  whisperEncode, whisperDecode,
} from '../model.js'

const { variable, tensor, zeros, ones, noGrad, conv1d, sinusoidalPE, createCausalMask } = smith

// --- Conv1d tests (now using Smith core) ---

test('conv1d basic shapes', () => {
  noGrad(() => {
    const x = variable(tensor(Array.from({ length: 80 * 100 }, () => Math.random()), [80, 100]), { requiresGrad: false })
    const w = variable(tensor(Array.from({ length: 384 * 80 * 3 }, () => Math.random() * 0.01), [384, 80, 3]), { requiresGrad: false })
    const b = variable(zeros([384]), { requiresGrad: false })

    // stride=1, padding=1 → same length
    const out = conv1d(x, w, b, { stride: 1, padding: 1 })
    expect(out.data.shape).toEqual([384, 100])
  })
})

test('conv1d with stride 2', () => {
  noGrad(() => {
    const x = variable(tensor(Array.from({ length: 384 * 100 }, () => Math.random()), [384, 100]), { requiresGrad: false })
    const w = variable(tensor(Array.from({ length: 384 * 384 * 3 }, () => Math.random() * 0.01), [384, 384, 3]), { requiresGrad: false })
    const b = variable(zeros([384]), { requiresGrad: false })

    // stride=2, padding=1 → length/2
    const out = conv1d(x, w, b, { stride: 2, padding: 1 })
    expect(out.data.shape).toEqual([384, 50])
  })
})

// --- Sinusoidal PE tests (now using Smith core) ---

test('sinusoidal PE has correct shape and properties', () => {
  const pe = sinusoidalPE(1500, 384)
  expect(pe.shape).toEqual([1500, 384])
  expect(pe.data[0]).toBeCloseTo(0, 5)
  for (let i = 0; i < Math.min(1000, pe.data.length); i++) {
    expect(Math.abs(pe.data[i])).toBeLessThanOrEqual(1.001)
  }
})

// --- Causal mask ---

test('causal mask shape and values', () => {
  const mask = createCausalMask(4)
  expect(mask.shape).toEqual([4, 4])
  const d = mask.data
  expect(d[1]).toBe(-Infinity)
  expect(d[0]).toBe(0)
  expect(d[5]).toBe(0)
  expect(d[4]).toBe(0)
})

// --- Encoder block ---

test('encoder block forward', () => {
  noGrad(() => {
    const block = createEncoderBlock(64, 4, 256)
    const x = variable(tensor(Array.from({ length: 10 * 64 }, () => Math.random() * 0.1), [10, 64]), { requiresGrad: false })
    const out = encoderBlock(x, block)
    expect(out.data.shape).toEqual([10, 64])
  })
})

// --- Decoder block ---

test('decoder block forward', () => {
  noGrad(() => {
    const block = createDecoderBlock(64, 4, 256)
    const x = variable(tensor(Array.from({ length: 5 * 64 }, () => Math.random() * 0.1), [5, 64]), { requiresGrad: false })
    const enc = variable(tensor(Array.from({ length: 20 * 64 }, () => Math.random() * 0.1), [20, 64]), { requiresGrad: false })
    const mask = createCausalMask(5)
    const out = decoderBlock(x, block, enc, mask)
    expect(out.data.shape).toEqual([5, 64])
  })
})

// --- Config ---

test('WHISPER_CONFIGS has expected model sizes', () => {
  expect(WHISPER_CONFIGS.tiny.dim).toBe(384)
  expect(WHISPER_CONFIGS.tiny.encoderLayers).toBe(4)
  expect(WHISPER_CONFIGS.tiny.decoderLayers).toBe(4)
  expect(WHISPER_CONFIGS.large.dim).toBe(1280)
  expect(WHISPER_CONFIGS.large.nMels).toBe(128)
})

test('createWhisperModel tiny has correct structure', () => {
  const model = createWhisperModel(WHISPER_CONFIGS.tiny)
  expect(model.encoderBlocks.length).toBe(4)
  expect(model.decoderBlocks.length).toBe(4)
  expect(model.conv1W.data.shape).toEqual([384, 80, 3])
  expect(model.conv2W.data.shape).toEqual([384, 384, 3])
  expect(model.tokenEmbed.data.shape).toEqual([51865, 384])
  expect(model.decoderPE.data.shape).toEqual([448, 384])
  // Check encoder block uses Smith's MHA structure
  expect(model.encoderBlocks[0].mha.qProj.weight.data.shape).toEqual([384, 384])
  expect(model.encoderBlocks[0].mha.numHeads).toBe(6)
  // Check decoder block has both self-attn and cross-attn
  expect(model.decoderBlocks[0].selfAttn.numHeads).toBe(6)
  expect(model.decoderBlocks[0].crossAttn.numHeads).toBe(6)
})

// --- Integration: encode ---

test('whisperEncode produces correct output shape', () => {
  noGrad(() => {
    const config = { ...WHISPER_CONFIGS.tiny, encoderLayers: 1, decoderLayers: 1 }
    const model = createWhisperModel(config)

    const mel = new Float32Array(80 * 100)
    for (let i = 0; i < mel.length; i++) mel[i] = Math.random() * 0.1

    const out = whisperEncode(model, mel)
    expect(out.data.shape[0]).toBe(50) // conv2 stride=2 halves
    expect(out.data.shape[1]).toBe(384)
  })
})

// --- Integration: decode ---

test('whisperDecode produces logits', () => {
  noGrad(() => {
    const config = { ...WHISPER_CONFIGS.tiny, encoderLayers: 1, decoderLayers: 1 }
    const model = createWhisperModel(config)

    const encoderOut = variable(
      tensor(Array.from({ length: 50 * 384 }, () => Math.random() * 0.1), [50, 384]),
      { requiresGrad: false }
    )

    const tokens = [50258, 50259, 50359, 50363]
    const logits = whisperDecode(model, encoderOut, tokens)
    expect(logits.data.shape).toEqual([4, 51865])
  })
})
