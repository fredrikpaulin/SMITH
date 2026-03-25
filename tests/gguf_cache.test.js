// tests/gguf_cache.test.js
// Phase 12: KV Cache for GGUF Models
// Tests cache management, cached forward (prefill + decode), GQA head repetition,
// and generateGGUF entry point using a small synthetic Llama-style model.

import { test, expect, describe } from 'bun:test'
import * as T from '../src/tensor.js'
import * as A from '../src/autograd.js'
import {
  createGGUFCache, resetCache,
  cacheAppend, cachePrefill, cacheSlice, repeatKV,
  forwardLlamaCachedDecode, forwardLlamaCachedPrefill,
  generateGGUF,
} from '../src/gguf_cache.js'
import { createGGUFModel } from '../src/gguf_loader.js'
import { precomputeRoPE } from '../src/ops/rope.js'

const flat = t => new Float32Array(t.data.buffer, t.data.byteOffset, t.data.length)

// --- Helper: create a tiny Llama-like model for testing ---
function createTinyLlama() {
  const config = {
    arch: 'llama',
    vocabSize: 32,
    dim: 16,
    numLayers: 1,
    numHeads: 2,
    numKVHeads: 2,
    maxSeqLen: 32,
    ffnDim: 32,
    ropeFreqBase: 10000,
    normEps: 1e-5,
  }
  const model = createGGUFModel(config)

  // Initialize weights with small random values for non-degenerate outputs
  const fill = (v) => { v.data.data.set(Float32Array.from({ length: v.data.size }, () => (Math.random() - 0.5) * 0.1)) }
  fill(model.embedding.tokenWeight)
  for (const block of model.blocks) {
    fill(block.mha.qProj.weight)
    fill(block.mha.kProj.weight)
    fill(block.mha.vProj.weight)
    fill(block.mha.outProj.weight)
    fill(block.ffnGate.weight)
    fill(block.ffnUp.weight)
    fill(block.ffnDown.weight)
    // ln gammas should be ~1
    block.ln1Gamma.data.data.set(Float32Array.from({ length: config.dim }, () => 0.9 + Math.random() * 0.2))
    block.ln2Gamma.data.data.set(Float32Array.from({ length: config.dim }, () => 0.9 + Math.random() * 0.2))
  }
  model.lnFGamma.data.data.set(Float32Array.from({ length: config.dim }, () => 0.9 + Math.random() * 0.2))

  // lmHead
  fill(model.lmHead)
  model.weightTied = false

  return model
}

// --- Helper: create a tiny GQA model (fewer KV heads than Q heads) ---
function createTinyGQA() {
  const config = {
    arch: 'llama',
    vocabSize: 32,
    dim: 16,
    numLayers: 1,
    numHeads: 4,     // 4 Q heads
    numKVHeads: 2,   // 2 KV heads → GQA ratio 2
    maxSeqLen: 32,
    ffnDim: 32,
    ropeFreqBase: 10000,
    normEps: 1e-5,
  }
  const model = createGGUFModel(config)
  const headDim = config.dim / config.numHeads  // 4
  const kvDim = config.numKVHeads * headDim       // 8

  const fill = (v) => { v.data.data.set(Float32Array.from({ length: v.data.size }, () => (Math.random() - 0.5) * 0.1)) }
  fill(model.embedding.tokenWeight)
  for (const block of model.blocks) {
    fill(block.mha.qProj.weight)
    fill(block.mha.kProj.weight)
    fill(block.mha.vProj.weight)
    fill(block.mha.outProj.weight)
    fill(block.ffnGate.weight)
    fill(block.ffnUp.weight)
    fill(block.ffnDown.weight)
    block.ln1Gamma.data.data.set(Float32Array.from({ length: config.dim }, () => 0.9 + Math.random() * 0.2))
    block.ln2Gamma.data.data.set(Float32Array.from({ length: config.dim }, () => 0.9 + Math.random() * 0.2))
  }
  model.lnFGamma.data.data.set(Float32Array.from({ length: config.dim }, () => 0.9 + Math.random() * 0.2))
  fill(model.lmHead)
  model.weightTied = false

  return model
}


// === Cache Management ===

describe('createGGUFCache', () => {
  test('allocates correct shapes per layer', () => {
    const config = { numLayers: 3, numKVHeads: 4, numHeads: 8, maxSeqLen: 64, dim: 32 }
    const caches = createGGUFCache(config)
    expect(caches.length).toBe(3)
    for (const c of caches) {
      expect(c.k.shape).toEqual([4, 64, 4])  // [kvHeads, maxSeqLen, headDim=32/8]
      expect(c.v.shape).toEqual([4, 64, 4])
      expect(c.len).toBe(0)
    }
  })

  test('defaults to numHeads when numKVHeads is undefined', () => {
    const config = { numLayers: 1, numHeads: 4, maxSeqLen: 16, dim: 16 }
    const caches = createGGUFCache(config)
    expect(caches[0].k.shape).toEqual([4, 16, 4])  // uses numHeads=4
  })
})

describe('resetCache', () => {
  test('zeros all lengths without reallocating', () => {
    const config = { numLayers: 2, numKVHeads: 2, numHeads: 2, maxSeqLen: 16, dim: 8 }
    const caches = createGGUFCache(config)
    caches[0].len = 5
    caches[1].len = 10
    const kRef = caches[0].k  // keep reference
    resetCache(caches)
    expect(caches[0].len).toBe(0)
    expect(caches[1].len).toBe(0)
    expect(caches[0].k).toBe(kRef)  // same buffer object
  })
})

describe('cacheAppend', () => {
  test('writes single position and increments len', () => {
    const config = { numLayers: 1, numKVHeads: 2, numHeads: 2, maxSeqLen: 8, dim: 8 }
    const caches = createGGUFCache(config)
    const headDim = 4
    const kvHeads = 2

    // Create kNew, vNew: [2, 1, 4]
    const kNew = T.tensor(Float32Array.from({ length: kvHeads * headDim }, (_, i) => i + 1), [kvHeads, 1, headDim])
    const vNew = T.tensor(Float32Array.from({ length: kvHeads * headDim }, (_, i) => (i + 1) * 10), [kvHeads, 1, headDim])

    cacheAppend(caches[0], kNew, vNew)
    expect(caches[0].len).toBe(1)

    // Verify data at position 0
    const kData = flat(caches[0].k)
    // head 0, pos 0: values 1,2,3,4
    expect(kData[0]).toBeCloseTo(1)
    expect(kData[1]).toBeCloseTo(2)
    expect(kData[2]).toBeCloseTo(3)
    expect(kData[3]).toBeCloseTo(4)

    // Append another
    const kNew2 = T.tensor(Float32Array.from({ length: kvHeads * headDim }, (_, i) => -(i + 1)), [kvHeads, 1, headDim])
    const vNew2 = T.tensor(Float32Array.from({ length: kvHeads * headDim }, (_, i) => -(i + 1) * 10), [kvHeads, 1, headDim])
    cacheAppend(caches[0], kNew2, vNew2)
    expect(caches[0].len).toBe(2)

    // head 0, pos 1: values -1,-2,-3,-4
    expect(kData[4]).toBeCloseTo(-1)
    expect(kData[5]).toBeCloseTo(-2)
  })
})

describe('cachePrefill', () => {
  test('writes multiple positions at once', () => {
    const config = { numLayers: 1, numKVHeads: 2, numHeads: 2, maxSeqLen: 8, dim: 8 }
    const caches = createGGUFCache(config)
    const headDim = 4
    const kvHeads = 2
    const seqLen = 3

    // kNew: [2, 3, 4]
    const kNew = T.tensor(Float32Array.from({ length: kvHeads * seqLen * headDim }, (_, i) => i), [kvHeads, seqLen, headDim])
    const vNew = T.tensor(Float32Array.from({ length: kvHeads * seqLen * headDim }, (_, i) => i * 2), [kvHeads, seqLen, headDim])

    cachePrefill(caches[0], kNew, vNew, seqLen)
    expect(caches[0].len).toBe(3)

    // Verify: head 0 should have the first seqLen*headDim values
    const kData = flat(caches[0].k)
    for (let i = 0; i < seqLen * headDim; i++) {
      expect(kData[i]).toBeCloseTo(i)
    }
  })
})

describe('cacheSlice', () => {
  test('returns tensors sliced to current length', () => {
    const config = { numLayers: 1, numKVHeads: 2, numHeads: 2, maxSeqLen: 8, dim: 8 }
    const caches = createGGUFCache(config)
    const headDim = 4
    const kvHeads = 2
    const seqLen = 3

    const kNew = T.tensor(Float32Array.from({ length: kvHeads * seqLen * headDim }, (_, i) => i + 1), [kvHeads, seqLen, headDim])
    const vNew = T.tensor(Float32Array.from({ length: kvHeads * seqLen * headDim }, (_, i) => (i + 1) * 10), [kvHeads, seqLen, headDim])
    cachePrefill(caches[0], kNew, vNew, seqLen)

    const { k, v } = cacheSlice(caches[0])
    expect(k.shape).toEqual([2, 3, 4])
    expect(v.shape).toEqual([2, 3, 4])

    // Verify values match what was written
    const kSliceData = flat(k)
    for (let i = 0; i < kvHeads * seqLen * headDim; i++) {
      expect(kSliceData[i]).toBeCloseTo(i + 1)
    }
  })
})

describe('repeatKV', () => {
  test('no-op when kvHeads === numHeads', () => {
    const t = T.tensor(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2])
    const out = repeatKV(t, 2, 2)
    expect(out).toBe(t)  // identity, same object
  })

  test('repeats heads for GQA (2 kv → 4 q)', () => {
    // [2, 1, 3] — 2 kv heads, seqLen=1, headDim=3
    const data = new Float32Array([1, 2, 3, 4, 5, 6])
    const t = T.tensor(data, [2, 1, 3])
    const out = repeatKV(t, 4, 2)  // 4 Q heads, 2 KV heads → repeat 2x
    expect(out.shape).toEqual([4, 1, 3])

    const outData = flat(out)
    // head 0 and 1 should be copies of kv head 0
    expect(outData[0]).toBeCloseTo(1)
    expect(outData[3]).toBeCloseTo(1)
    // head 2 and 3 should be copies of kv head 1
    expect(outData[6]).toBeCloseTo(4)
    expect(outData[9]).toBeCloseTo(4)
  })

  test('repeats heads for GQA (1 kv → 4 q)', () => {
    const data = new Float32Array([10, 20])
    const t = T.tensor(data, [1, 1, 2])
    const out = repeatKV(t, 4, 1)
    expect(out.shape).toEqual([4, 1, 2])

    const outData = flat(out)
    for (let h = 0; h < 4; h++) {
      expect(outData[h * 2]).toBeCloseTo(10)
      expect(outData[h * 2 + 1]).toBeCloseTo(20)
    }
  })
})


// === Cached Forward Pass ===

describe('forwardLlamaCachedPrefill', () => {
  test('produces logits with correct shape', () => {
    const model = createTinyLlama()
    const caches = createGGUFCache(model.config)
    const tokenIds = [1, 5, 10]

    let result
    A.noGrad(() => {
      result = forwardLlamaCachedPrefill(model, tokenIds, caches)
    })

    // logits: [seqLen, vocabSize]
    expect(result.logits.data.shape).toEqual([3, 32])
    // Cache should be filled with 3 positions
    expect(caches[0].len).toBe(3)
  })

  test('logits contain finite values', () => {
    const model = createTinyLlama()
    const caches = createGGUFCache(model.config)

    let result
    A.noGrad(() => {
      result = forwardLlamaCachedPrefill(model, [0, 1, 2], caches)
    })

    const logitsData = flat(result.logits.data)
    for (let i = 0; i < logitsData.length; i++) {
      expect(Number.isFinite(logitsData[i])).toBe(true)
    }
  })
})

describe('forwardLlamaCachedDecode', () => {
  test('produces logits [1, vocabSize] and advances cache', () => {
    const model = createTinyLlama()
    const caches = createGGUFCache(model.config)

    // First prefill
    A.noGrad(() => {
      forwardLlamaCachedPrefill(model, [1, 2, 3], caches)
    })
    expect(caches[0].len).toBe(3)

    // Then decode one token
    let result
    A.noGrad(() => {
      result = forwardLlamaCachedDecode(model, 5, 3, caches)
    })

    expect(result.logits.data.shape).toEqual([1, 32])
    expect(caches[0].len).toBe(4)  // advanced by 1
  })

  test('multiple decode steps advance cache correctly', () => {
    const model = createTinyLlama()
    const caches = createGGUFCache(model.config)

    A.noGrad(() => {
      forwardLlamaCachedPrefill(model, [0], caches)
      forwardLlamaCachedDecode(model, 1, 1, caches)
      forwardLlamaCachedDecode(model, 2, 2, caches)
      forwardLlamaCachedDecode(model, 3, 3, caches)
    })

    expect(caches[0].len).toBe(4)
  })

  test('decode logits are finite', () => {
    const model = createTinyLlama()
    const caches = createGGUFCache(model.config)

    let result
    A.noGrad(() => {
      forwardLlamaCachedPrefill(model, [1, 2], caches)
      result = forwardLlamaCachedDecode(model, 3, 2, caches)
    })

    const data = flat(result.logits.data)
    for (let i = 0; i < data.length; i++) {
      expect(Number.isFinite(data[i])).toBe(true)
    }
  })
})


// === Prefill vs sequential decode consistency ===

describe('prefill vs decode consistency', () => {
  test('prefill logits match sequential single-token decode for last position', () => {
    const model = createTinyLlama()
    const tokens = [3, 7, 15]

    // Approach 1: prefill all at once
    const caches1 = createGGUFCache(model.config)
    let prefillLogits
    A.noGrad(() => {
      const result = forwardLlamaCachedPrefill(model, tokens, caches1)
      // Last token's logits
      const vocabSize = model.config.vocabSize
      const lastPos = tokens.length - 1
      prefillLogits = Array.from(result.logits.data.data.slice(lastPos * vocabSize, (lastPos + 1) * vocabSize))
    })

    // Approach 2: decode one token at a time
    const caches2 = createGGUFCache(model.config)
    let decodeLogits
    A.noGrad(() => {
      for (let i = 0; i < tokens.length; i++) {
        const result = forwardLlamaCachedDecode(model, tokens[i], i, caches2)
        if (i === tokens.length - 1) {
          decodeLogits = Array.from(result.logits.data.data.slice(0, model.config.vocabSize))
        }
      }
    })

    // They should be very close (not exact due to flash attention vs matmul+softmax)
    for (let i = 0; i < prefillLogits.length; i++) {
      expect(prefillLogits[i]).toBeCloseTo(decodeLogits[i], 1)  // within 0.1
    }
  })
})


// === GQA cached forward ===

describe('GQA cached forward', () => {
  test('prefill works with fewer KV heads', () => {
    const model = createTinyGQA()
    const caches = createGGUFCache(model.config)

    let result
    A.noGrad(() => {
      result = forwardLlamaCachedPrefill(model, [0, 1, 2], caches)
    })

    expect(result.logits.data.shape).toEqual([3, 32])
    expect(caches[0].len).toBe(3)
    // KV cache should use kvHeads=2, not numHeads=4
    expect(caches[0].k.shape[0]).toBe(2)
  })

  test('decode works with GQA', () => {
    const model = createTinyGQA()
    const caches = createGGUFCache(model.config)

    let result
    A.noGrad(() => {
      forwardLlamaCachedPrefill(model, [0, 1], caches)
      result = forwardLlamaCachedDecode(model, 2, 2, caches)
    })

    expect(result.logits.data.shape).toEqual([1, 32])
    expect(caches[0].len).toBe(3)
    const data = flat(result.logits.data)
    for (let i = 0; i < data.length; i++) {
      expect(Number.isFinite(data[i])).toBe(true)
    }
  })
})


// === generateGGUF ===

describe('generateGGUF', () => {
  test('generates tokens from prompt', () => {
    const model = createTinyLlama()
    const promptIds = [1, 2, 3]
    const result = generateGGUF(model, promptIds, { maxTokens: 5, temperature: 0 })

    expect(result.length).toBe(promptIds.length + 5)
    // First tokens should be the prompt
    expect(result[0]).toBe(1)
    expect(result[1]).toBe(2)
    expect(result[2]).toBe(3)
    // Generated tokens should be valid vocab indices
    for (let i = promptIds.length; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0)
      expect(result[i]).toBeLessThan(32)
    }
  })

  test('respects maxTokens', () => {
    const model = createTinyLlama()
    const result = generateGGUF(model, [0], { maxTokens: 3, temperature: 0 })
    expect(result.length).toBe(4)  // 1 prompt + 3 generated
  })

  test('stops on EOS token', () => {
    const model = createTinyLlama()
    // Force the model to always predict token 0 by zeroing lmHead then biasing token 0
    const vocabSize = model.config.vocabSize
    const dim = model.config.dim
    model.lmHead.data.data.fill(0)
    for (let d = 0; d < dim; d++) {
      model.lmHead.data.data[d * vocabSize + 0] = 1000  // token 0 logit overwhelms everything
    }

    const result = generateGGUF(model, [1], { maxTokens: 10, temperature: 0, eosToken: 0 })
    // Must stop before maxTokens and last token must be EOS
    expect(result.length).toBeLessThan(1 + 10)  // stopped early (prompt + fewer than maxTokens)
    expect(result[result.length - 1]).toBe(0)
  })

  test('calls onToken callback', () => {
    const model = createTinyLlama()
    const tokens = []
    generateGGUF(model, [1, 2], { maxTokens: 3, temperature: 0 }, {
      onToken: (tok, step) => { tokens.push({ tok, step }) },
    })
    expect(tokens.length).toBe(3)
    expect(tokens[0].step).toBe(1)
    expect(tokens[1].step).toBe(2)
    expect(tokens[2].step).toBe(3)
  })

  test('onToken can stop generation early', () => {
    const model = createTinyLlama()
    const result = generateGGUF(model, [1], { maxTokens: 10, temperature: 0 }, {
      onToken: (tok, step) => step >= 2,  // stop after 2 tokens
    })
    expect(result.length).toBe(3)  // 1 prompt + 2 generated
  })

  test('temperature=0 is deterministic', () => {
    const model = createTinyLlama()
    const r1 = generateGGUF(model, [1, 2, 3], { maxTokens: 5, temperature: 0 })
    const r2 = generateGGUF(model, [1, 2, 3], { maxTokens: 5, temperature: 0 })
    expect(r1).toEqual(r2)
  })

  test('works with GQA model', () => {
    const model = createTinyGQA()
    const result = generateGGUF(model, [1, 2], { maxTokens: 3, temperature: 0 })
    expect(result.length).toBe(5)  // 2 prompt + 3 generated
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0)
      expect(result[i]).toBeLessThan(32)
    }
  })
})


// === Weight-tied model ===

describe('weight-tied model', () => {
  test('forward works when lmHead shares embedding weights', () => {
    const model = createTinyLlama()
    model.weightTied = true
    // Point lmHead at embedding (simulating weight tying)
    model.lmHead.data = model.embedding.tokenWeight.data

    const caches = createGGUFCache(model.config)
    let result
    A.noGrad(() => {
      result = forwardLlamaCachedPrefill(model, [0, 1, 2], caches)
    })
    expect(result.logits.data.shape).toEqual([3, 32])

    const data = flat(result.logits.data)
    for (let i = 0; i < data.length; i++) {
      expect(Number.isFinite(data[i])).toBe(true)
    }
  })
})
