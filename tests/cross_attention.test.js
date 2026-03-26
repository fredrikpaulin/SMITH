import { test, expect } from 'bun:test'
import smith from '../src/index.js'

const {
  variable, tensor, zeros, ones, noGrad, backward,
  createMultiHeadAttention, multiHeadCrossAttention, multiHeadCrossAttentionCached,
  createLinear, linear, linearParams,
  sinusoidalPE,
} = smith

// --- Cross-attention tests ---

test('multiHeadCrossAttention — basic forward shape', () => {
  noGrad(() => {
    const mha = createMultiHeadAttention(64, 4)
    const x = variable(tensor(Array.from({ length: 5 * 64 }, () => Math.random() * 0.1), [5, 64]))
    const kv = variable(tensor(Array.from({ length: 20 * 64 }, () => Math.random() * 0.1), [20, 64]))

    const { output } = multiHeadCrossAttention(x, kv, mha)
    expect(output.data.shape).toEqual([5, 64])
  })
})

test('multiHeadCrossAttention — different Q and KV lengths', () => {
  noGrad(() => {
    const mha = createMultiHeadAttention(128, 8)
    const decoder = variable(tensor(Array.from({ length: 3 * 128 }, () => Math.random() * 0.1), [3, 128]))
    const encoder = variable(tensor(Array.from({ length: 50 * 128 }, () => Math.random() * 0.1), [50, 128]))

    const { output } = multiHeadCrossAttention(decoder, encoder, mha)
    expect(output.data.shape).toEqual([3, 128])
  })
})

test('multiHeadCrossAttention — output changes with different KV', () => {
  noGrad(() => {
    const mha = createMultiHeadAttention(64, 4)
    const x = variable(tensor(Array.from({ length: 2 * 64 }, () => 0.1), [2, 64]))
    const kv1 = variable(tensor(Array.from({ length: 5 * 64 }, () => 0.1), [5, 64]))
    const kv2 = variable(tensor(Array.from({ length: 5 * 64 }, () => 0.9), [5, 64]))

    const { output: out1 } = multiHeadCrossAttention(x, kv1, mha)
    const { output: out2 } = multiHeadCrossAttention(x, kv2, mha)

    // Different KV sources should produce different outputs
    let same = true
    for (let i = 0; i < out1.data.data.length; i++) {
      if (Math.abs(out1.data.data[i] - out2.data.data[i]) > 1e-6) { same = false; break }
    }
    expect(same).toBe(false)
  })
})

test('multiHeadCrossAttention — self-attention matches when kv === x', () => {
  noGrad(() => {
    const mha = createMultiHeadAttention(64, 4)
    const x = variable(tensor(Array.from({ length: 5 * 64 }, () => Math.random() * 0.1), [5, 64]))

    // Cross-attention with kv=x should be equivalent to self-attention
    const { output: crossOut } = multiHeadCrossAttention(x, x, mha)
    const { output: selfOut } = smith.multiHeadAttention(x, mha)

    // Should be identical (no mask in either case)
    for (let i = 0; i < crossOut.data.data.length; i++) {
      expect(Math.abs(crossOut.data.data[i] - selfOut.data.data[i])).toBeLessThan(1e-4)
    }
  })
})

test('multiHeadCrossAttention with mask', () => {
  noGrad(() => {
    const mha = createMultiHeadAttention(64, 4)
    const x = variable(tensor(Array.from({ length: 3 * 64 }, () => Math.random() * 0.1), [3, 64]))
    const kv = variable(tensor(Array.from({ length: 10 * 64 }, () => Math.random() * 0.1), [10, 64]))

    // Create a mask that blocks some KV positions
    const maskData = new Float32Array(3 * 10)
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 10; j++) {
        maskData[i * 10 + j] = j > 5 ? -Infinity : 0
      }
    }
    const mask = tensor(Array.from(maskData), [3, 10])

    const { output } = multiHeadCrossAttention(x, kv, mha, mask)
    expect(output.data.shape).toEqual([3, 64])
  })
})

// --- Cached cross-attention tests ---

test('multiHeadCrossAttentionCached — forward shape', () => {
  noGrad(() => {
    const mha = createMultiHeadAttention(64, 4)
    const x = variable(tensor(Array.from({ length: 1 * 64 }, () => Math.random() * 0.1), [1, 64]))

    // Pre-compute encoder KV
    const encoder = variable(tensor(Array.from({ length: 20 * 64 }, () => Math.random() * 0.1), [20, 64]))
    const K = linear(encoder, mha.kProj)
    const V = linear(encoder, mha.vProj)
    const Kh = smith.transpose(smith.reshape(K, [20, 4, 16]), [1, 0, 2])
    const Vh = smith.transpose(smith.reshape(V, [20, 4, 16]), [1, 0, 2])
    const encoderKV = { k: Kh, v: Vh }

    const { output } = multiHeadCrossAttentionCached(x, encoderKV, mha)
    expect(output.data.shape).toEqual([1, 64])
  })
})

// --- Sinusoidal PE tests ---

test('sinusoidalPE shape', () => {
  const pe = sinusoidalPE(100, 64)
  expect(pe.shape).toEqual([100, 64])
})

test('sinusoidalPE values bounded', () => {
  const pe = sinusoidalPE(50, 128)
  for (let i = 0; i < pe.data.length; i++) {
    expect(Math.abs(pe.data[i])).toBeLessThanOrEqual(1.001)
  }
})

test('sinusoidalPE first position starts with sin(0)=0', () => {
  const pe = sinusoidalPE(10, 8)
  // pos=0, dim=0: sin(0) = 0
  expect(pe.data[0]).toBeCloseTo(0, 5)
  // pos=0, dim=1: cos(0) = 1
  expect(pe.data[1]).toBeCloseTo(1, 5)
})

test('sinusoidalPE different positions produce different embeddings', () => {
  const pe = sinusoidalPE(100, 64)
  let same = true
  const dim = 64
  for (let d = 0; d < dim; d++) {
    if (Math.abs(pe.data[0 * dim + d] - pe.data[1 * dim + d]) > 1e-6) { same = false; break }
  }
  expect(same).toBe(false)
})

test('sinusoidalPE low frequencies change slowly', () => {
  const pe = sinusoidalPE(100, 64)
  // High dim indices = low frequency → values change slowly between positions
  const dim = 64
  const highDimIdx = 62 // near end → low freq
  const lowDimIdx = 0   // near start → high freq
  const diffHigh = Math.abs(pe.data[0 * dim + highDimIdx] - pe.data[1 * dim + highDimIdx])
  const diffLow = Math.abs(pe.data[0 * dim + lowDimIdx] - pe.data[1 * dim + lowDimIdx])
  // Low-dim positions should change more between adjacent positions
  expect(diffLow).toBeGreaterThan(diffHigh)
})

test('sinusoidalPE Whisper encoder dimensions', () => {
  // Whisper tiny: 1500 positions, dim 384
  const pe = sinusoidalPE(1500, 384)
  expect(pe.shape).toEqual([1500, 384])
  expect(pe.data.length).toBe(1500 * 384)
})
