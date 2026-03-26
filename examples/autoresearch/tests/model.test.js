// examples/autoresearch/tests/model.test.js
// Tests for autoresearch GPT model (forward, backward, optimizer integration).
// Run: bun test examples/autoresearch/tests/

import { test, expect } from 'bun:test'
import smith from '../../../src/index.js'
import { createModel, initWeights, forward, setupOptimizer, countModelParams, allParams } from '../model.js'

const flat = t => Array.from(t.data)

// Small model for fast tests
const smallConfig = {
  seqLen: 32,
  vocabSize: 64,
  nLayer: 2,
  nHead: 2,
  nKVHead: 2,
  nEmbd: 64,
  windowPattern: 'SL',
}

test('createModel returns correct structure', () => {
  const model = createModel(smallConfig)
  expect(model.config.nLayer).toBe(2)
  expect(model.config.nEmbd).toBe(64)
  expect(model.blocks.length).toBe(2)
  expect(model.wte.data.shape).toEqual([64, 64])  // [vocabSize, nEmbd]
  expect(model.lmHead.data.shape).toEqual([64, 64])  // [nEmbd, vocabSize]
  expect(model.residLambdas.data.shape).toEqual([2])
  expect(model.x0Lambdas.data.shape).toEqual([2])
})

test('initWeights sets values', () => {
  const model = createModel(smallConfig)
  initWeights(model)

  // Residual lambdas should be 1.0
  const rl = flat(model.residLambdas.data)
  expect(rl[0]).toBe(1.0)
  expect(rl[1]).toBe(1.0)

  // x0 lambdas should be 0.1
  const xl = flat(model.x0Lambdas.data)
  expect(Math.abs(xl[0] - 0.1)).toBeLessThan(1e-6)

  // cProj should be zeros
  const proj = flat(model.blocks[0].cProj.data)
  expect(proj.every(v => v === 0)).toBe(true)

  // wte should have non-zero values
  const wte = flat(model.wte.data)
  expect(wte.some(v => v !== 0)).toBe(true)
})

test('countModelParams returns positive number', () => {
  const model = createModel(smallConfig)
  const n = countModelParams(model)
  expect(n).toBeGreaterThan(0)
  // Small model should be well under 1M params
  expect(n).toBeLessThan(1000000)
})

test('forward produces logits of correct shape', () => {
  const model = createModel(smallConfig)
  initWeights(model)

  const tokens = Array.from({ length: 16 }, () => Math.floor(Math.random() * 64))
  const { logits } = forward(model, tokens)

  expect(logits.data.shape).toEqual([16, 64])  // [T, vocabSize]

  // Values should be finite
  const vals = flat(logits.data)
  for (const v of vals) expect(isFinite(v)).toBe(true)
})

test('forward with targets returns loss', () => {
  const model = createModel(smallConfig)
  initWeights(model)

  const tokens = Array.from({ length: 16 }, () => Math.floor(Math.random() * 64))
  const targets = Array.from({ length: 16 }, () => Math.floor(Math.random() * 64))

  const { logits, loss } = forward(model, tokens, targets)

  expect(logits.data.shape).toEqual([16, 64])
  expect(loss.data.shape).toEqual([])  // scalar

  const lossVal = loss.data.data[0]
  expect(isFinite(lossVal)).toBe(true)
  expect(lossVal).toBeGreaterThan(0)

  // Loss should be around -log(1/64) ≈ 4.16 for random weights
  expect(lossVal).toBeLessThan(10)
})

test('backward produces gradients', () => {
  const model = createModel(smallConfig)
  initWeights(model)

  const tokens = Array.from({ length: 8 }, () => Math.floor(Math.random() * 64))
  const targets = Array.from({ length: 8 }, () => Math.floor(Math.random() * 64))

  const { loss } = forward(model, tokens, targets)
  smith.backward(loss)

  // wte should have gradients
  expect(model.wte.grad).not.toBeNull()
  const wteGrad = flat(model.wte.grad)
  expect(wteGrad.some(v => v !== 0)).toBe(true)

  // Output projections get gradients (init'd to zero, so upstream weights
  // like cQ/cFc don't get grads until projections become non-zero after a step)
  expect(model.blocks[0].cProj.grad).not.toBeNull()
  const cpGrad = flat(model.blocks[0].cProj.grad)
  expect(cpGrad.some(v => v !== 0)).toBe(true)

  expect(model.blocks[0].cMlpProj.grad).not.toBeNull()
  const mpGrad = flat(model.blocks[0].cMlpProj.grad)
  expect(mpGrad.some(v => v !== 0)).toBe(true)

  // lmHead gets gradients directly from loss
  expect(model.lmHead.grad).not.toBeNull()
  const lmGrad = flat(model.lmHead.grad)
  expect(lmGrad.some(v => v !== 0)).toBe(true)

  // All gradients should be finite
  const params = allParams(model)
  for (const p of params) {
    if (p.grad) {
      const g = flat(p.grad)
      for (const v of g) expect(isFinite(v)).toBe(true)
    }
  }
})

test('MuonAdamW optimizer step produces finite params', () => {
  const model = createModel(smallConfig)
  initWeights(model)

  const tokens = Array.from({ length: 8 }, () => Math.floor(Math.random() * 64))
  const targets = Array.from({ length: 8 }, () => Math.floor(Math.random() * 64))

  const opt = setupOptimizer(model, { matrixLr: 0.01, weightDecay: 0.0 })

  // Forward + backward
  const { loss } = forward(model, tokens, targets)
  smith.backward(loss)

  // Optimizer step
  smith.muonAdamWStep(opt)

  // All params should be finite
  const params = allParams(model)
  for (const p of params) {
    const vals = flat(p.data)
    for (const v of vals) expect(isFinite(v)).toBe(true)
  }
})

test('training reduces loss over steps', () => {
  const model = createModel(smallConfig)
  initWeights(model)
  const opt = setupOptimizer(model, { matrixLr: 0.01, weightDecay: 0.0 })
  const params = allParams(model)

  // Fixed data for consistent test
  const tokens = Array.from({ length: 16 }, (_, i) => i % 64)
  const targets = Array.from({ length: 16 }, (_, i) => (i + 1) % 64)

  let firstLoss = null
  let lastLoss = null

  for (let step = 0; step < 10; step++) {
    const { loss } = forward(model, tokens, targets)
    const lossVal = loss.data.data[0]

    if (step === 0) firstLoss = lossVal
    lastLoss = lossVal

    smith.backward(loss)
    smith.muonAdamWStep(opt)
    smith.zeroGrad(params)
  }

  // Loss should decrease (not necessarily monotonically, but overall trend)
  expect(lastLoss).toBeLessThan(firstLoss)
})

test('value embeddings are present on correct layers', () => {
  const model = createModel(smallConfig)

  // With nLayer=2: hasVE(0,2) = 0%2 == 1%2 → false, hasVE(1,2) = 1%2 == 1%2 → true
  expect(model.blocks[0].veEmbed).toBeUndefined()
  expect(model.blocks[1].veEmbed).toBeDefined()
  expect(model.blocks[1].veGate).toBeDefined()
})

test('soft-capping bounds logits', () => {
  const model = createModel(smallConfig)
  initWeights(model)

  const tokens = Array.from({ length: 8 }, () => Math.floor(Math.random() * 64))
  const { logits } = forward(model, tokens)

  // Logit soft-capping: 15 * tanh(logits/15) bounds output to [-15, 15]
  const vals = flat(logits.data)
  for (const v of vals) {
    expect(v).toBeGreaterThan(-15.01)
    expect(v).toBeLessThan(15.01)
  }
})

test('window sizes follow SSSL pattern', () => {
  const cfg = { ...smallConfig, nLayer: 4, windowPattern: 'SSSL' }
  const model = createModel(cfg)

  // S=half, S=half, S=half, L=full — but last always full
  const half = Math.floor(cfg.seqLen / 2)
  expect(model.windowSizes[0]).toBe(half)
  expect(model.windowSizes[1]).toBe(half)
  expect(model.windowSizes[2]).toBe(half)
  expect(model.windowSizes[3]).toBe(cfg.seqLen)  // last always full
})

// --- After one step, projections are non-zero and ALL params get gradients ---

test('full gradient flow after one optimizer step', () => {
  const model = createModel(smallConfig)
  initWeights(model)
  const opt = setupOptimizer(model, { matrixLr: 0.01, weightDecay: 0.0 })
  const params = allParams(model)

  const tokens = Array.from({ length: 8 }, (_, i) => i % 64)
  const targets = Array.from({ length: 8 }, (_, i) => (i + 1) % 64)

  // Step 0: projections are zero, only some params get grads
  const { loss: loss0 } = forward(model, tokens, targets)
  smith.backward(loss0)
  smith.muonAdamWStep(opt)
  smith.zeroGrad(params)

  // Step 1: projections are now non-zero — all params should get grads
  const { loss: loss1 } = forward(model, tokens, targets)
  smith.backward(loss1)

  // Every param with requiresGrad should now have non-zero gradient
  for (const p of params) {
    if (p.grad) {
      const g = flat(p.grad)
      const hasNonZero = g.some(v => v !== 0)
      expect(hasNonZero).toBe(true)
      for (const v of g) expect(isFinite(v)).toBe(true)
    }
  }
})

// --- GQA: fewer KV heads than Q heads ---

test('GQA forward and backward (nKVHead < nHead)', () => {
  const gqaConfig = {
    seqLen: 32,
    vocabSize: 64,
    nLayer: 2,
    nHead: 4,
    nKVHead: 2,     // 2 KV heads shared across 4 Q heads
    nEmbd: 64,
    windowPattern: 'SL',
  }
  const model = createModel(gqaConfig)
  initWeights(model)

  // Forward should work with GQA
  const tokens = Array.from({ length: 8 }, () => Math.floor(Math.random() * 64))
  const targets = Array.from({ length: 8 }, () => Math.floor(Math.random() * 64))
  const { logits, loss } = forward(model, tokens, targets)

  expect(logits.data.shape).toEqual([8, 64])
  const lossVal = loss.data.data[0]
  expect(isFinite(lossVal)).toBe(true)
  expect(lossVal).toBeGreaterThan(0)

  // Backward should produce finite gradients
  smith.backward(loss)
  const params = allParams(model)
  for (const p of params) {
    if (p.grad) {
      const g = flat(p.grad)
      for (const v of g) expect(isFinite(v)).toBe(true)
    }
  }

  // Verify shapes: cK/cV should use kvDim = nKVHead * headDim = 2 * 16 = 32
  const headDim = 64 / 4  // nEmbd / nHead = 16
  expect(model.blocks[0].cK.data.shape).toEqual([64, 32])  // [nEmbd, nKVHead*headDim]
  expect(model.blocks[0].cQ.data.shape).toEqual([64, 64])  // [nEmbd, nHead*headDim]
})

// --- Edge case: T=1 (single token) ---

test('forward with single token (T=1)', () => {
  const model = createModel(smallConfig)
  initWeights(model)

  const { logits } = forward(model, [42])
  expect(logits.data.shape).toEqual([1, 64])

  const vals = flat(logits.data)
  for (const v of vals) expect(isFinite(v)).toBe(true)
})

// --- Edge case: T=seqLen (full sequence length) ---

test('forward with full sequence length (T=seqLen)', () => {
  const model = createModel(smallConfig)
  initWeights(model)

  const tokens = Array.from({ length: smallConfig.seqLen }, () => Math.floor(Math.random() * 64))
  const targets = Array.from({ length: smallConfig.seqLen }, () => Math.floor(Math.random() * 64))
  const { logits, loss } = forward(model, tokens, targets)

  expect(logits.data.shape).toEqual([smallConfig.seqLen, 64])
  expect(isFinite(loss.data.data[0])).toBe(true)

  // Backward at full seqLen should work
  smith.backward(loss)
  const params = allParams(model)
  for (const p of params) {
    if (p.grad) {
      const g = flat(p.grad)
      for (const v of g) expect(isFinite(v)).toBe(true)
    }
  }
})
