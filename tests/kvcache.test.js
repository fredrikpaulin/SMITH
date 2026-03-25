// smith/tests/kvcache.test.js
// Tests for Phase 5: KV cache — cached attention, cached forward, cached generation.

import { test, expect } from 'bun:test'
import smith from '../src/index.js'

// --- Cached attention produces correct output shape ---

test('multiHeadAttentionCached returns correct shape with no cache', () => {
  const mha = smith.createMultiHeadAttention(16, 2)
  const x = smith.variable(smith.rand([1, 16]), { requiresGrad: false })
  const { output, newCache } = smith.multiHeadAttentionCached(x, mha, null)

  expect(output.data.shape).toEqual([1, 16])
  expect(newCache.k.data.shape).toEqual([2, 1, 8]) // [numHeads, 1, headDim]
  expect(newCache.v.data.shape).toEqual([2, 1, 8])
})

test('multiHeadAttentionCached grows cache on subsequent calls', () => {
  const mha = smith.createMultiHeadAttention(16, 2)

  // First token
  const x1 = smith.variable(smith.rand([1, 16]), { requiresGrad: false })
  const { newCache: cache1 } = smith.multiHeadAttentionCached(x1, mha, null)
  expect(cache1.k.data.shape).toEqual([2, 1, 8])

  // Second token
  const x2 = smith.variable(smith.rand([1, 16]), { requiresGrad: false })
  const { newCache: cache2 } = smith.multiHeadAttentionCached(x2, mha, cache1)
  expect(cache2.k.data.shape).toEqual([2, 2, 8])
  expect(cache2.v.data.shape).toEqual([2, 2, 8])

  // Third token
  const x3 = smith.variable(smith.rand([1, 16]), { requiresGrad: false })
  const { newCache: cache3 } = smith.multiHeadAttentionCached(x3, mha, cache2)
  expect(cache3.k.data.shape).toEqual([2, 3, 8])
})

// --- Cached transformer block ---

test('transformerBlockCached preserves dimension and grows cache', () => {
  const block = smith.createTransformerBlock(16, 2)
  const x = smith.variable(smith.rand([1, 16]), { requiresGrad: false })

  const { output, newCache } = smith.transformerBlockCached(x, block, null)
  expect(output.data.shape).toEqual([1, 16])
  expect(newCache.k.data.shape[1]).toBe(1) // seqLen = 1

  const x2 = smith.variable(smith.rand([1, 16]), { requiresGrad: false })
  const { output: out2, newCache: cache2 } = smith.transformerBlockCached(x2, block, newCache)
  expect(out2.data.shape).toEqual([1, 16])
  expect(cache2.k.data.shape[1]).toBe(2) // seqLen = 2
})

// --- forwardCached ---

test('forwardCached produces correct logits shape', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 16 })

  smith.noGrad(() => {
    const { logits, newCaches } = smith.forwardCached(model, 0, 0, null)
    expect(logits.data.shape).toEqual([1, 32]) // [1, vocabSize]
    expect(newCaches.length).toBe(1) // 1 layer
    expect(newCaches[0].k.data.shape[1]).toBe(1)
  })
})

test('forwardCached builds cache over multiple positions', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 2, numHeads: 2, dim: 16, maxSeqLen: 16 })

  smith.noGrad(() => {
    const { newCaches: c1 } = smith.forwardCached(model, 0, 0, null)
    const { newCaches: c2 } = smith.forwardCached(model, 1, 1, c1)
    const { logits, newCaches: c3 } = smith.forwardCached(model, 2, 2, c2)

    expect(logits.data.shape).toEqual([1, 32])
    expect(c3.length).toBe(2) // 2 layers
    expect(c3[0].k.data.shape[1]).toBe(3) // 3 positions cached
    expect(c3[1].k.data.shape[1]).toBe(3)
  })
})

// --- generateCached ---

test('generateCached produces tokens', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 16 })
  const prompt = [0, 1, 2]
  const generated = smith.generateCached(model, prompt, { maxTokens: 5, temperature: 1.0 })
  expect(generated.length).toBe(8) // 3 prompt + 5 generated
  for (const id of generated) {
    expect(id).toBeGreaterThanOrEqual(0)
    expect(id).toBeLessThan(32)
  }
})

test('generateCached with temperature=0 is deterministic', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 16 })
  const prompt = [0, 1, 2]
  const a = smith.generateCached(model, prompt, { maxTokens: 5, temperature: 0 })
  const b = smith.generateCached(model, prompt, { maxTokens: 5, temperature: 0 })
  expect(a).toEqual(b)
})

// --- Equivalence: cached vs non-cached ---

test('cached and non-cached generation produce identical output at temperature=0', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 16 })
  const prompt = [0, 1, 2]

  const uncached = smith.generate(model, prompt, { maxTokens: 5, temperature: 0 })
  const cached = smith.generateCached(model, prompt, { maxTokens: 5, temperature: 0 })

  expect(cached).toEqual(uncached)
})

test('cached and non-cached produce identical output with 2-layer model', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 2, numHeads: 2, dim: 16, maxSeqLen: 16 })
  const prompt = [0, 1, 2, 3, 4]

  const uncached = smith.generate(model, prompt, { maxTokens: 4, temperature: 0 })
  const cached = smith.generateCached(model, prompt, { maxTokens: 4, temperature: 0 })

  expect(cached).toEqual(uncached)
})

// --- onToken callback ---

test('generateCached respects onToken callback for early stopping', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 16 })
  const prompt = [0, 1]
  let tokenCount = 0
  const generated = smith.generateCached(model, prompt, { maxTokens: 10, temperature: 1.0 }, {
    onToken: (token, idx) => {
      tokenCount++
      return tokenCount >= 3 // stop after 3 tokens
    }
  })
  expect(generated.length).toBeLessThanOrEqual(5) // 2 prompt + at most 3 generated
})
