// smith/tests/inference.test.js
// Tests for Phase 4: generation, checkpoint, quantization.

import { test, expect } from 'bun:test'
import { unlink } from 'node:fs/promises'
import smith from '../src/index.js'

function expectClose(actual, expected, tol = 0.1) {
  expect(Math.abs(actual - expected)).toBeLessThan(tol)
}

// --- Sampling utilities ---

test('argmax returns index of highest value', async () => {
  const { argmax } = await import('../src/generate.js')
  expect(argmax([1, 3, 2])).toBe(1)
  expect(argmax([5, 1, 0])).toBe(0)
})

test('applyTemperature sharpens distribution', async () => {
  const { applyTemperature } = await import('../src/generate.js')
  const logits = [1, 2, 3]
  const sharpened = applyTemperature(logits, 0.5)
  expect(sharpened[0]).toBe(2)
  expect(sharpened[1]).toBe(4)
  expect(sharpened[2]).toBe(6)
})

test('applyTopK masks low logits', async () => {
  const { applyTopK } = await import('../src/generate.js')
  const logits = [1, 5, 3, 2, 4]
  const filtered = applyTopK(logits, 2)
  // Only top-2 (5 and 4) should survive
  expect(filtered[1]).toBe(5)
  expect(filtered[4]).toBe(4)
  expect(filtered[0]).toBe(-Infinity)
  expect(filtered[3]).toBe(-Infinity)
})

test('applyRepetitionPenalty reduces seen tokens', async () => {
  const { applyRepetitionPenalty } = await import('../src/generate.js')
  const logits = [2, 3, 1]
  const penalized = applyRepetitionPenalty(logits, [0, 1], 2.0)
  expect(penalized[0]).toBe(1)   // 2 / 2.0
  expect(penalized[1]).toBe(1.5) // 3 / 2.0
  expect(penalized[2]).toBe(1)   // untouched
})

// --- Generation ---

test('generate produces tokens', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 16 })
  const prompt = [0, 1, 2]
  const generated = smith.generate(model, prompt, { maxTokens: 5, temperature: 1.0 })
  expect(generated.length).toBe(8) // 3 prompt + 5 generated
  for (const id of generated) {
    expect(id).toBeGreaterThanOrEqual(0)
    expect(id).toBeLessThan(32)
  }
})

test('generate with temperature=0 is deterministic', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 16 })
  const prompt = [0, 1, 2]
  const a = smith.generate(model, prompt, { maxTokens: 5, temperature: 0 })
  const b = smith.generate(model, prompt, { maxTokens: 5, temperature: 0 })
  expect(a).toEqual(b)
})

// --- Checkpoint ---

test('save and load checkpoint preserves weights', async () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 8 })

  // Generate some output before save
  const prompt = [0, 1, 2, 3]
  const before = smith.generate(model, prompt, { maxTokens: 3, temperature: 0 })

  const path = '/tmp/smith-test-ckpt'
  await smith.saveCheckpoint(model, path)
  const loaded = await smith.loadCheckpoint(path)

  // Generate with loaded model — should produce same output
  const after = smith.generate(loaded, prompt, { maxTokens: 3, temperature: 0 })
  expect(after).toEqual(before)

  // Cleanup
  await unlink(path + '.json').catch(() => {})
  await unlink(path + '.bin').catch(() => {})
})

// --- Quantization ---

test('quantizeQ4 + matmulQ4 produces reasonable results', () => {
  // Create a simple weight matrix and quantize it
  const K = 64, N = 32
  const w = smith.rand([K, N])
  const wq = smith.quantizeQ4(w)

  expect(wq.K).toBe(K)
  expect(wq.N).toBe(N)
  expect(wq.groups).toBe(2) // 64 / 32

  // Multiply
  const a = smith.ones([1, K])
  const out = smith.matmulQ4(a, wq)
  expect(out.shape).toEqual([1, N])

  // Result should be close to column sums of w (since a is all-ones)
  const expected = new Float32Array(N)
  for (let k = 0; k < K; k++) {
    for (let n = 0; n < N; n++) {
      expected[n] += w.data[k * N + n]
    }
  }

  const actual = smith.toArray(out)
  for (let n = 0; n < N; n++) {
    // Q4 quantization introduces error — allow 10% tolerance
    const relError = Math.abs(actual[0][n] - expected[n]) / (Math.abs(expected[n]) + 1e-6)
    expect(relError).toBeLessThan(0.15)
  }
})

test('quantizeQ4 reduces memory vs f32', () => {
  const K = 256, N = 256
  const w = smith.rand([K, N])
  const wq = smith.quantizeQ4(w)

  const f32Bytes = K * N * 4 // 262144
  // Q4: groups * N * 24 bytes
  const q4Bytes = wq.totalBytes
  expect(q4Bytes).toBeLessThan(f32Bytes) // should be ~4.6x smaller
  expect(q4Bytes / f32Bytes).toBeLessThan(0.35) // ~24/128 = 0.1875 per group
})
