// tests/sampling.test.js
// Phase 24: GPU-Side Sampling
// Tests argmax, penalties, top-k, softmax+multinomial, and full gpuSample pipeline.
// All tests use real GPU tensors via T.* — requires Apple Silicon.

import { test, expect, describe } from 'bun:test'
import * as T from '../src/tensor.js'
import {
  gpuArgmax,
  gpuApplyRepPenalty,
  gpuApplyTemperature,
  gpuApplyTopK,
  cpuApplyTopP,
  gpuMultinomialSample,
  gpuSample,
} from '../src/ops/sampling.js'
import { softmax } from '../src/ops/softmax.js'
import {
  argmax as cpuArgmax,
  applyRepetitionPenalty as cpuRepPenalty,
  applyTemperature as cpuTemperature,
  applyTopK as cpuTopK,
  applyTopP as cpuTopP,
} from '../src/generate.js'

const f = t => Array.from(t.data)

// --- gpuArgmax ---

describe('gpuArgmax', () => {
  test('finds max in small array', () => {
    const t = T.tensor([1, 3, 2, 0.5], [4])
    expect(gpuArgmax(t)).toBe(1)
  })

  test('finds max at first position', () => {
    const t = T.tensor([10, 3, 2, 0.5], [4])
    expect(gpuArgmax(t)).toBe(0)
  })

  test('finds max at last position', () => {
    const t = T.tensor([1, 3, 2, 5], [4])
    expect(gpuArgmax(t)).toBe(3)
  })

  test('handles negative values', () => {
    const t = T.tensor([-5, -1, -3, -2], [4])
    expect(gpuArgmax(t)).toBe(1)
  })

  test('matches CPU argmax on random logits', () => {
    const data = Array.from({ length: 1024 }, () => Math.random() * 10 - 5)
    const t = T.tensor(data, [1024])
    const cpuResult = cpuArgmax(data)
    expect(gpuArgmax(t)).toBe(cpuResult)
  })

  test('works with large vocab (32K)', () => {
    const size = 32000
    const data = new Float32Array(size)
    for (let i = 0; i < size; i++) data[i] = Math.random() * 20 - 10
    // Plant a known max
    const knownMax = 15000
    data[knownMax] = 100
    const t = T.tensor(Array.from(data), [size])
    expect(gpuArgmax(t)).toBe(knownMax)
  })

  test('multi-threadgroup reduction (> 256 elements)', () => {
    const size = 512
    const data = new Float32Array(size)
    data[400] = 99
    const t = T.tensor(Array.from(data), [size])
    expect(gpuArgmax(t)).toBe(400)
  })
})

// --- gpuApplyRepPenalty ---

describe('gpuApplyRepPenalty', () => {
  test('penalizes positive logits by dividing', () => {
    const t = T.tensor([4, 8, 2, 6], [4])
    gpuApplyRepPenalty(t, [1, 3], 2.0)
    const result = f(t)
    expect(result[0]).toBeCloseTo(4, 4)    // untouched
    expect(result[1]).toBeCloseTo(4, 4)    // 8 / 2
    expect(result[2]).toBeCloseTo(2, 4)    // untouched
    expect(result[3]).toBeCloseTo(3, 4)    // 6 / 2
  })

  test('penalizes negative logits by multiplying', () => {
    const t = T.tensor([4, -8, 2, -6], [4])
    gpuApplyRepPenalty(t, [1, 3], 2.0)
    const result = f(t)
    expect(result[1]).toBeCloseTo(-16, 4)  // -8 * 2
    expect(result[3]).toBeCloseTo(-12, 4)  // -6 * 2
  })

  test('no-op when penalty <= 1', () => {
    const t = T.tensor([1, 2, 3], [3])
    gpuApplyRepPenalty(t, [0, 1], 1.0)
    expect(f(t)).toEqual([1, 2, 3])
  })

  test('no-op when no recent tokens', () => {
    const t = T.tensor([1, 2, 3], [3])
    gpuApplyRepPenalty(t, [], 2.0)
    expect(f(t)).toEqual([1, 2, 3])
  })

  test('matches CPU repetition penalty', () => {
    const data = [3, -2, 5, -1, 4]
    const recent = [0, 2, 4]
    const penalty = 1.5

    const cpuResult = cpuRepPenalty(data, recent, penalty)
    const t = T.tensor([...data], [5])
    gpuApplyRepPenalty(t, recent, penalty)
    const gpuResult = f(t)

    for (let i = 0; i < data.length; i++) {
      expect(gpuResult[i]).toBeCloseTo(cpuResult[i], 4)
    }
  })
})

// --- gpuApplyTemperature ---

describe('gpuApplyTemperature', () => {
  test('divides by temperature', () => {
    const t = T.tensor([2, 4, 6], [3])
    gpuApplyTemperature(t, 2.0)
    const result = f(t)
    expect(result[0]).toBeCloseTo(1, 4)
    expect(result[1]).toBeCloseTo(2, 4)
    expect(result[2]).toBeCloseTo(3, 4)
  })

  test('no-op at temperature=1', () => {
    const t = T.tensor([2, 4, 6], [3])
    gpuApplyTemperature(t, 1.0)
    expect(f(t)).toEqual([2, 4, 6])
  })

  test('low temperature sharpens distribution', () => {
    const t = T.tensor([1, 2, 3], [3])
    gpuApplyTemperature(t, 0.5)
    const result = f(t)
    expect(result[0]).toBeCloseTo(2, 4)
    expect(result[1]).toBeCloseTo(4, 4)
    expect(result[2]).toBeCloseTo(6, 4)
  })
})

// --- gpuApplyTopK ---

describe('gpuApplyTopK', () => {
  test('masks non-top-K to -Infinity', () => {
    const t = T.tensor([1, 5, 2, 4, 3], [5])
    gpuApplyTopK(t, 2)
    const result = f(t)
    // Top 2 are indices 1 (5) and 3 (4)
    expect(result[1]).toBe(5)
    expect(result[3]).toBe(4)
    expect(result[0]).toBe(-Infinity)
    expect(result[2]).toBe(-Infinity)
    expect(result[4]).toBe(-Infinity)
  })

  test('no-op when k >= size', () => {
    const t = T.tensor([1, 2, 3], [3])
    gpuApplyTopK(t, 5)
    expect(f(t)).toEqual([1, 2, 3])
  })

  test('no-op when k <= 0', () => {
    const t = T.tensor([1, 2, 3], [3])
    gpuApplyTopK(t, 0)
    expect(f(t)).toEqual([1, 2, 3])
  })

  test('top-1 keeps only the maximum', () => {
    const t = T.tensor([3, 1, 4, 1, 5], [5])
    gpuApplyTopK(t, 1)
    const result = f(t)
    expect(result[4]).toBe(5)
    for (let i = 0; i < 4; i++) expect(result[i]).toBe(-Infinity)
  })

  test('handles duplicate values', () => {
    const t = T.tensor([3, 5, 5, 1, 2], [5])
    gpuApplyTopK(t, 2)
    const result = f(t)
    // Both 5s should survive (threshold is 5)
    expect(result[1]).toBe(5)
    expect(result[2]).toBe(5)
  })

  test('matches CPU top-K for random logits', () => {
    const data = Array.from({ length: 100 }, () => Math.random() * 10 - 5)
    const k = 10

    const cpuResult = cpuTopK([...data], k)
    const t = T.tensor([...data], [100])
    gpuApplyTopK(t, k)
    const gpuResult = f(t)

    // Same positions should be -Infinity
    for (let i = 0; i < data.length; i++) {
      if (cpuResult[i] === -Infinity) {
        expect(gpuResult[i]).toBe(-Infinity)
      } else {
        expect(gpuResult[i]).toBeCloseTo(cpuResult[i], 4)
      }
    }
  })
})

// --- cpuApplyTopP ---

describe('cpuApplyTopP', () => {
  test('keeps tokens until cumulative prob >= p', () => {
    // Create a simple probability distribution
    const t = T.tensor([0.5, 0.3, 0.15, 0.05], [4])
    cpuApplyTopP(t, 0.8)
    const result = f(t)
    // Should keep indices 0 (0.5) and 1 (0.3) = cumulative 0.8
    expect(result[0]).toBeGreaterThan(0)
    expect(result[1]).toBeGreaterThan(0)
    expect(result[2]).toBe(0)
    expect(result[3]).toBe(0)
  })

  test('no-op when p >= 1', () => {
    const t = T.tensor([0.25, 0.25, 0.25, 0.25], [4])
    cpuApplyTopP(t, 1.0)
    const result = f(t)
    for (const v of result) expect(v).toBeCloseTo(0.25, 4)
  })

  test('renormalizes after masking', () => {
    const t = T.tensor([0.5, 0.3, 0.15, 0.05], [4])
    cpuApplyTopP(t, 0.8)
    const result = f(t)
    const sum = result.reduce((a, b) => a + b, 0)
    expect(sum).toBeCloseTo(1.0, 4)
  })
})

// --- gpuMultinomialSample ---

describe('gpuMultinomialSample', () => {
  test('samples according to probability', () => {
    // Probability concentrated on index 2
    const t = T.tensor([0, 0, 1, 0], [4])
    expect(gpuMultinomialSample(t, 0.5)).toBe(2)
  })

  test('samples first token with low random value', () => {
    const t = T.tensor([0.5, 0.3, 0.2], [3])
    expect(gpuMultinomialSample(t, 0.1)).toBe(0)
  })

  test('samples last token with high random value', () => {
    const t = T.tensor([0.3, 0.3, 0.4], [3])
    expect(gpuMultinomialSample(t, 0.99)).toBe(2)
  })

  test('respects cumulative boundaries', () => {
    // probs = [0.5, 0.3, 0.2]. Cumulative: [0.5, 0.8, 1.0]
    const t = T.tensor([0.5, 0.3, 0.2], [3])
    expect(gpuMultinomialSample(t, 0.49)).toBe(0)
    // Need fresh tensor since we might modify in-place
    const t2 = T.tensor([0.5, 0.3, 0.2], [3])
    expect(gpuMultinomialSample(t2, 0.51)).toBe(1)
    const t3 = T.tensor([0.5, 0.3, 0.2], [3])
    expect(gpuMultinomialSample(t3, 0.81)).toBe(2)
  })
})

// --- gpuSample (full pipeline) ---

describe('gpuSample', () => {
  test('greedy (temperature=0) matches gpuArgmax', () => {
    const data = [1, 5, 2, 4, 3]
    const t1 = T.tensor([...data], [5])
    const t2 = T.tensor([...data], [5])
    const greedy = gpuSample(t1, { temperature: 0 })
    const argmaxResult = gpuArgmax(t2)
    expect(greedy).toBe(argmaxResult)
    expect(greedy).toBe(1) // index of max value (5)
  })

  test('greedy with repetition penalty', () => {
    // Without penalty, argmax is index 1 (value 5)
    const t1 = T.tensor([1, 5, 4, 3, 2], [5])
    const noPenalty = gpuSample(t1, { temperature: 0 })
    expect(noPenalty).toBe(1)

    // With penalty on index 1, argmax should shift to index 2 (value 4)
    const t2 = T.tensor([1, 5, 4, 3, 2], [5])
    const withPenalty = gpuSample(t2, {
      temperature: 0,
      repetitionPenalty: 10.0,
      recentTokens: [1],
    })
    expect(withPenalty).toBe(2)
  })

  test('stochastic sampling returns valid token index', () => {
    const size = 100
    const data = Array.from({ length: size }, () => Math.random())
    const t = T.tensor(data, [size])
    const token = gpuSample(t, { temperature: 1.0 })
    expect(token).toBeGreaterThanOrEqual(0)
    expect(token).toBeLessThan(size)
  })

  test('top-K restricts sampling to top values', () => {
    // Make one value dominant — with top-K=1, must always pick it
    const data = new Float32Array(50).fill(0)
    data[25] = 100
    const t = T.tensor(Array.from(data), [50])
    const token = gpuSample(t, { temperature: 1.0, topK: 1 })
    expect(token).toBe(25)
  })

  test('temperature=0 with top-K still works (greedy)', () => {
    const t = T.tensor([1, 5, 2, 4, 3], [5])
    const token = gpuSample(t, { temperature: 0, topK: 3 })
    // Greedy skips top-K, just does argmax after penalties
    expect(token).toBe(1)
  })
})

// --- Integration: generateGGUF with gpuSampling ---

import { generateGGUF } from '../src/gguf_cache.js'
import { createGGUFModel } from '../src/gguf_loader.js'
import { precomputeRoPE } from '../src/ops/rope.js'

describe('generateGGUF with gpuSampling', () => {
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
      normEps: 1e-5,
    }
    const headDim = config.dim / config.numHeads
    const model = createGGUFModel(config)
    // Random init
    const fill = t => { for (let i = 0; i < t.data.length; i++) t.data[i] = (Math.random() - 0.5) * 0.1 }
    fill(model.embedding.tokenWeight.data)
    for (const b of model.blocks) {
      fill(b.ln1Gamma.data); b.ln1Gamma.data.data.fill(1)
      fill(b.mha.qProj.weight.data); fill(b.mha.kProj.weight.data)
      fill(b.mha.vProj.weight.data); fill(b.mha.outProj.weight.data)
      fill(b.ln2Gamma.data); b.ln2Gamma.data.data.fill(1)
      fill(b.ffnGate.weight.data); fill(b.ffnUp.weight.data); fill(b.ffnDown.weight.data)
    }
    fill(model.lnFGamma.data); model.lnFGamma.data.data.fill(1)
    fill(model.lmHead.data)
    model.rope = precomputeRoPE(headDim, config.maxSeqLen)
    model.normEps = config.normEps
    model.weightTied = false
    return model
  }

  test('GPU sampling produces same result as CPU at temperature=0', () => {
    const model = createTinyLlama()
    const cpuResult = generateGGUF(model, [1, 2, 3], {
      maxTokens: 5,
      temperature: 0,
      gpuSampling: false,
    })
    const gpuResult = generateGGUF(model, [1, 2, 3], {
      maxTokens: 5,
      temperature: 0,
      gpuSampling: true,
    })
    expect(gpuResult).toEqual(cpuResult)
  })

  test('GPU sampling respects maxTokens', () => {
    const model = createTinyLlama()
    const result = generateGGUF(model, [0], {
      maxTokens: 3,
      temperature: 0,
      gpuSampling: true,
    })
    expect(result.length).toBe(4)  // 1 prompt + 3 generated
  })

  test('GPU sampling respects eosToken', () => {
    const model = createTinyLlama()
    // Probe to find the model's actual prediction
    const probe = generateGGUF(model, [1], {
      maxTokens: 1,
      temperature: 0,
      gpuSampling: true,
    })
    const eosToken = probe[probe.length - 1]

    const result = generateGGUF(model, [1], {
      maxTokens: 10,
      temperature: 0,
      gpuSampling: true,
      eosToken,
    })
    expect(result.length).toBe(2)
    expect(result[result.length - 1]).toBe(eosToken)
  })

  test('GPU sampling with stochastic config produces valid tokens', () => {
    const model = createTinyLlama()
    const result = generateGGUF(model, [1, 2], {
      maxTokens: 5,
      temperature: 0.8,
      topK: 10,
      topP: 0.9,
      gpuSampling: true,
    })
    expect(result.length).toBe(7)  // 2 prompt + 5 generated
    for (let i = 2; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0)
      expect(result[i]).toBeLessThan(32)
    }
  })
})
