// smith/tests/transformer.test.js
// Tests for Phase 3: softmax, layernorm, crossEntropy, transformer forward/backward.

import { test, expect } from 'bun:test'
import smith from '../src/index.js'

function expectClose(actual, expected, tol = 1e-3) {
  if (Array.isArray(expected)) {
    for (let i = 0; i < expected.length; i++) expectClose(actual[i], expected[i], tol)
  } else {
    expect(Math.abs(actual - expected)).toBeLessThan(tol)
  }
}

// --- Softmax tests ---

test('softmax sums to 1', () => {
  const t = smith.tensor([1, 2, 3, 4], [1, 4])
  const v = smith.variable(t)
  const s = smith.softmax(v, -1)
  const arr = smith.toArray(s.data)
  const sum = arr[0].reduce((a, b) => a + b, 0)
  expect(Math.abs(sum - 1)).toBeLessThan(1e-5)
})

test('softmax correct values', () => {
  const t = smith.tensor([0, 0, 0], [1, 3])
  const v = smith.variable(t)
  const s = smith.softmax(v, -1)
  const arr = smith.toArray(s.data)
  // uniform: each should be 1/3
  expectClose(arr, [[1/3, 1/3, 1/3]], 1e-5)
})

test('softmax numerical stability (large values)', () => {
  const t = smith.tensor([1000, 1001, 1002], [1, 3])
  const v = smith.variable(t)
  const s = smith.softmax(v, -1)
  const arr = smith.toArray(s.data)
  const sum = arr[0].reduce((a, b) => a + b, 0)
  expect(Math.abs(sum - 1)).toBeLessThan(1e-4)
  // Should be same as softmax([0,1,2])
  expect(arr[0][2]).toBeGreaterThan(arr[0][1])
  expect(arr[0][1]).toBeGreaterThan(arr[0][0])
})

test('softmax backward', () => {
  const t = smith.tensor([1, 2, 3], [1, 3])
  const v = smith.variable(t, { requiresGrad: true })
  const s = smith.softmax(v, -1)
  const loss = smith.sum(s)
  smith.backward(loss)
  // Sum of softmax is always 1, so gradient w.r.t. input should be ~0
  const g = smith.toArray(v.grad)
  for (const row of g) for (const val of (Array.isArray(row) ? row : [row])) {
    expect(Math.abs(val)).toBeLessThan(1e-5)
  }
})

// --- Layernorm tests ---

test('layernorm output has zero mean and unit variance', () => {
  const x = smith.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])
  const gamma = smith.tensor([1, 1, 1, 1], [4])
  const beta = smith.tensor([0, 0, 0, 0], [4])
  const vx = smith.variable(x, { requiresGrad: true })
  const vg = smith.variable(gamma, { requiresGrad: true })
  const vb = smith.variable(beta, { requiresGrad: true })

  const out = smith.layernorm(vx, vg, vb)
  const arr = smith.toArray(out.data)

  // Check each row has ~zero mean
  for (const row of arr) {
    const mean = row.reduce((a, b) => a + b, 0) / row.length
    expect(Math.abs(mean)).toBeLessThan(1e-4)
    // Check ~unit variance
    const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length
    expect(Math.abs(variance - 1)).toBeLessThan(0.1)
  }
})

test('layernorm backward propagates gradients', () => {
  const x = smith.tensor([1, 2, 3, 4], [1, 4])
  const gamma = smith.tensor([1, 1, 1, 1], [4])
  const beta = smith.tensor([0, 0, 0, 0], [4])
  const vx = smith.variable(x, { requiresGrad: true })
  const vg = smith.variable(gamma, { requiresGrad: true })
  const vb = smith.variable(beta, { requiresGrad: true })

  const out = smith.layernorm(vx, vg, vb)
  const loss = smith.sum(out)
  smith.backward(loss)

  // All three should have gradients
  expect(vx.grad).not.toBeNull()
  expect(vg.grad).not.toBeNull()
  expect(vb.grad).not.toBeNull()

  // Beta grad should be ones (sum backward through identity)
  const bgArr = smith.toArray(vb.grad)
  expectClose(bgArr, [1, 1, 1, 1], 1e-4)
})

// --- CrossEntropy tests ---

test('crossEntropy computes correct loss', () => {
  // logits = [[0, 0, 10]] with target = [2] should give low loss
  const logits = smith.tensor([0, 0, 10], [1, 3])
  const vl = smith.variable(logits, { requiresGrad: true })
  const loss = smith.crossEntropy(vl, [2])
  const lossVal = smith.toArray(loss.data)
  expect(lossVal).toBeLessThan(0.01) // very confident prediction
})

test('crossEntropy gradient: softmax minus one_hot', () => {
  // With uniform logits and target=1, gradient should be softmax - [0,1,0]
  const logits = smith.tensor([0, 0, 0], [1, 3])
  const vl = smith.variable(logits, { requiresGrad: true })
  const loss = smith.crossEntropy(vl, [1])
  smith.backward(loss)

  const g = smith.toArray(vl.grad)
  // softmax([0,0,0]) = [1/3, 1/3, 1/3], minus [0,1,0] = [1/3, -2/3, 1/3]
  expectClose(g, [[1/3, -2/3, 1/3]], 1e-4)
})

// --- Tokenizer tests ---

test('tokenizer encode/decode roundtrip', () => {
  const text = 'hello world hello world'
  const tok = smith.tokenizer.train(text, 260)
  const ids = smith.tokenizer.encode(text, tok.merges)
  const decoded = smith.tokenizer.decode(ids, tok.vocab)
  expect(decoded).toBe(text)
})

test('tokenizer compression: vocab > 256 reduces token count', () => {
  const text = 'aaabbbaaabbbaaabbb'
  const tok = smith.tokenizer.train(text, 260)
  const ids = smith.tokenizer.encode(text, tok.merges)
  // Merged tokens should produce fewer IDs than raw bytes
  expect(ids.length).toBeLessThan(text.length)
})

// --- End-to-end: tiny GPT forward + backward ---

test('tiny GPT forward produces logits of correct shape', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 8 })
  const { logits } = smith.forward(model, [0, 1, 2, 3])
  // logits: [seqLen, vocabSize]
  expect(logits.data.shape).toEqual([4, 32])
})

// --- Finite-difference gradient checks for layernorm and softmax ---

function numGradCheck(makeLoss, x, tol = 1e-3) {
  const eps = 1e-4
  smith.zeroGrad([x])
  const loss = makeLoss(x)
  smith.backward(loss)
  const analyticGrad = Array.from(x.grad.data)

  const xData = x.data.data
  for (let i = 0; i < xData.length; i++) {
    const orig = xData[i]
    xData[i] = orig + eps
    const lossPlus = smith.noGrad(() => smith.toArray(makeLoss(x).data))
    xData[i] = orig - eps
    const lossMinus = smith.noGrad(() => smith.toArray(makeLoss(x).data))
    xData[i] = orig

    const numGrad = (lossPlus - lossMinus) / (2 * eps)
    expect(Math.abs(analyticGrad[i] - numGrad)).toBeLessThan(tol)
  }
}

test('finite-diff: layernorm input gradient (4x8)', () => {
  const rows = 4, cols = 8
  const xData = Array.from({ length: rows * cols }, () => Math.random() * 2 - 1)
  const x = smith.variable(smith.tensor(xData, [rows, cols]), { requiresGrad: true })
  const gamma = smith.variable(smith.tensor(Array.from({ length: cols }, () => 1 + Math.random() * 0.5), [cols]))
  const beta = smith.variable(smith.tensor(Array.from({ length: cols }, () => Math.random() * 0.1), [cols]))
  numGradCheck(v => smith.sum(smith.layernorm(v, gamma, beta)), x, 5e-3)
})

test('finite-diff: layernorm gamma gradient (4x8)', () => {
  const rows = 4, cols = 8
  const xData = Array.from({ length: rows * cols }, () => Math.random() * 2 - 1)
  const x = smith.variable(smith.tensor(xData, [rows, cols]))
  const gamma = smith.variable(smith.tensor(Array.from({ length: cols }, () => 1 + Math.random() * 0.5), [cols]), { requiresGrad: true })
  const beta = smith.variable(smith.tensor(Array.from({ length: cols }, () => Math.random() * 0.1), [cols]))
  numGradCheck(v => smith.sum(smith.layernorm(x, v, beta)), gamma, 5e-3)
})

test('finite-diff: softmax backward (4x16)', () => {
  const rows = 4, cols = 16
  const xData = Array.from({ length: rows * cols }, () => Math.random() * 3 - 1)
  const x = smith.variable(smith.tensor(xData, [rows, cols]), { requiresGrad: true })
  // Use a weighted sum so gradient isn't trivially zero
  const weights = smith.variable(smith.tensor(Array.from({ length: rows * cols }, () => Math.random()), [rows, cols]))
  numGradCheck(v => smith.sum(smith.mul(smith.softmax(v, -1), weights)), x, 5e-3)
})

test('finite-diff: cross-entropy gradient (8x32)', () => {
  const batch = 8, vocab = 32
  const xData = Array.from({ length: batch * vocab }, () => Math.random() * 4 - 2)
  const targets = Array.from({ length: batch }, () => Math.floor(Math.random() * vocab))
  const x = smith.variable(smith.tensor(xData, [batch, vocab]), { requiresGrad: true })
  numGradCheck(v => smith.crossEntropy(v, targets), x, 5e-3)
})

test('tiny GPT training step reduces loss', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 8 })
  const params = smith.modelParams(model)
  const opt = smith.createAdamW(params, { lr: 0.01, weightDecay: 0 })

  function trainStep(input, target) {
    smith.zeroGrad(params)
    const { logits } = smith.forward(model, input)
    const loss = smith.crossEntropy(logits, target)
    smith.backward(loss)
    smith.clipGradNorm(params, 1.0)
    smith.adamwStep(opt)
    return smith.toArray(loss.data)
  }

  const input = [0, 1, 2, 3]
  const target = [1, 2, 3, 4]

  const losses = []
  for (let i = 0; i < 10; i++) {
    losses.push(trainStep(input, target))
  }

  // Loss should decrease over 10 steps on the same data
  expect(losses[losses.length - 1]).toBeLessThan(losses[0])
})
