// smith/tests/flash_attention_gqa_window.test.js
// Tests for Phase 26: GQA + Sliding Window Flash Attention

import { test, expect } from 'bun:test'
import smith from '../src/index.js'

const T = { tensor: smith.tensor, zeros: smith.zeros, ones: smith.ones, rand: smith.rand, toArray: smith.toArray }
const A = {
  variable: smith.variable, param: smith.param, backward: smith.backward,
  zeroGrad: smith.zeroGrad, noGrad: smith.noGrad,
  matmul: smith.matmul, transpose: smith.transpose, scale: smith.scale,
  add: smith.add, softmax: smith.softmax, reshape: smith.reshape,
  flashAttention: smith.flashAttention, sum: smith.sum,
}

function expectClose(actual, expected, tol = 1e-3) {
  if (Array.isArray(expected)) {
    for (let i = 0; i < expected.length; i++) expectClose(actual[i], expected[i], tol)
  } else {
    expect(Math.abs(actual - expected)).toBeLessThan(tol)
  }
}

// --- Backward compat: boolean causal arg still works ---

test('backward compat: boolean causal=true', () => {
  const Q = A.variable(T.rand([2, 8, 4]), { requiresGrad: true })
  const K = A.variable(T.rand([2, 8, 4]), { requiresGrad: true })
  const V = A.variable(T.rand([2, 8, 4]), { requiresGrad: true })
  let out
  A.noGrad(() => { out = A.flashAttention(Q, K, V, true) })
  expect(out.data.shape).toEqual([2, 8, 4])
})

test('backward compat: boolean causal=false', () => {
  const Q = A.variable(T.rand([2, 8, 4]), { requiresGrad: true })
  const K = A.variable(T.rand([2, 8, 4]), { requiresGrad: true })
  const V = A.variable(T.rand([2, 8, 4]), { requiresGrad: true })
  let out
  A.noGrad(() => { out = A.flashAttention(Q, K, V, false) })
  expect(out.data.shape).toEqual([2, 8, 4])
})

test('opts object: causal=true produces same result as boolean', () => {
  const data = T.rand([2, 8, 4])
  const Q1 = A.variable(data, { requiresGrad: false })
  const K1 = A.variable(data, { requiresGrad: false })
  const V1 = A.variable(data, { requiresGrad: false })
  const Q2 = A.variable(data, { requiresGrad: false })
  const K2 = A.variable(data, { requiresGrad: false })
  const V2 = A.variable(data, { requiresGrad: false })

  let out1, out2
  A.noGrad(() => {
    out1 = A.flashAttention(Q1, K1, V1, true)
    out2 = A.flashAttention(Q2, K2, V2, { causal: true })
  })
  const a1 = T.toArray(out1.data)
  const a2 = T.toArray(out2.data)
  // Should be identical (same params)
  for (let i = 0; i < out1.data.data.length; i++) {
    expect(Math.abs(out1.data.data[i] - out2.data.data[i])).toBeLessThan(1e-6)
  }
})

// --- Sliding window tests ---

test('sliding window: output shape is correct', () => {
  const Q = A.variable(T.rand([2, 16, 8]), { requiresGrad: false })
  const K = A.variable(T.rand([2, 16, 8]), { requiresGrad: false })
  const V = A.variable(T.rand([2, 16, 8]), { requiresGrad: false })
  let out
  A.noGrad(() => {
    out = A.flashAttention(Q, K, V, { causal: true, windowSize: 4 })
  })
  expect(out.data.shape).toEqual([2, 16, 8])
})

test('sliding window: windowSize=0 equals full context', () => {
  const qd = T.rand([1, 16, 4])
  const kd = T.rand([1, 16, 4])
  const vd = T.rand([1, 16, 4])
  let outFull, outWindow0
  A.noGrad(() => {
    outFull = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd), { causal: true })
    outWindow0 = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd), { causal: true, windowSize: 0 })
  })
  for (let i = 0; i < outFull.data.data.length; i++) {
    expect(Math.abs(outFull.data.data[i] - outWindow0.data.data[i])).toBeLessThan(1e-6)
  }
})

test('sliding window: large window equals full context', () => {
  const qd = T.rand([1, 16, 4])
  const kd = T.rand([1, 16, 4])
  const vd = T.rand([1, 16, 4])
  let outFull, outLargeWindow
  A.noGrad(() => {
    outFull = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd), { causal: true })
    outLargeWindow = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd), { causal: true, windowSize: 1000 })
  })
  for (let i = 0; i < outFull.data.data.length; i++) {
    expect(Math.abs(outFull.data.data[i] - outLargeWindow.data.data[i])).toBeLessThan(1e-6)
  }
})

test('sliding window: small window isolates distant positions', () => {
  // With window=2 at position 7, it can only see positions 5-7
  // Compare output at position 7 with window vs full causal
  // They should differ (full causal sees 0-7, window sees 5-7)
  const N = 8, d = 4
  const qd = T.rand([1, N, d])
  const kd = T.rand([1, N, d])
  const vd = T.rand([1, N, d])

  let outFull, outWindow
  A.noGrad(() => {
    outFull = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd),
      { causal: true })
    outWindow = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd),
      { causal: true, windowSize: 2 })
  })

  // Position 7 with full causal attends to 0-7, with window=2 attends to 5-7
  // These should be different (the window cuts off early positions)
  const pos7offset = 7 * d
  let maxDiff = 0
  for (let j = 0; j < d; j++) {
    const diff = Math.abs(outFull.data.data[pos7offset + j] - outWindow.data.data[pos7offset + j])
    maxDiff = Math.max(maxDiff, diff)
  }
  // There should be a noticeable difference at distant positions
  expect(maxDiff).toBeGreaterThan(1e-4)

  // Position 0 should be identical (only attends to itself in both cases)
  for (let j = 0; j < d; j++) {
    expect(Math.abs(outFull.data.data[j] - outWindow.data.data[j])).toBeLessThan(1e-5)
  }
})

test('sliding window: position 1 with window=2 still sees position 0', () => {
  const N = 8, d = 4
  const qd = T.rand([1, N, d])
  const kd = T.rand([1, N, d])
  const vd = T.rand([1, N, d])

  let outFull, outWindow
  A.noGrad(() => {
    outFull = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd), { causal: true })
    outWindow = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd), { causal: true, windowSize: 2 })
  })

  // Position 1 with window=2: can attend to positions 0 and 1 (distance <= 2 from row=1)
  // With full context causal: also attends to 0 and 1.
  // So outputs at position 1 should be identical.
  const pos1offset = 1 * d
  for (let j = 0; j < d; j++) {
    expect(Math.abs(outFull.data.data[pos1offset + j] - outWindow.data.data[pos1offset + j])).toBeLessThan(1e-5)
  }
})

// --- Sliding window backward ---

test('sliding window: backward produces gradients', () => {
  const Q = A.variable(T.rand([1, 8, 4]), { requiresGrad: true })
  const K = A.variable(T.rand([1, 8, 4]), { requiresGrad: true })
  const V = A.variable(T.rand([1, 8, 4]), { requiresGrad: true })

  const out = A.flashAttention(Q, K, V, { causal: true, windowSize: 3 })
  const loss = A.sum(out)
  A.backward(loss)

  expect(Q.grad).not.toBeNull()
  expect(K.grad).not.toBeNull()
  expect(V.grad).not.toBeNull()
  // Grads should be non-zero
  let qGradSum = 0
  for (let i = 0; i < Q.grad.data.length; i++) qGradSum += Math.abs(Q.grad.data[i])
  expect(qGradSum).toBeGreaterThan(0)
})

// --- GQA tests ---

test('GQA forward: output shape with numKVHeads < numQHeads', () => {
  const numQHeads = 4, numKVHeads = 2, N = 8, d = 4
  const Q = A.variable(T.rand([numQHeads, N, d]), { requiresGrad: false })
  const K = A.variable(T.rand([numKVHeads, N, d]), { requiresGrad: false })
  const V = A.variable(T.rand([numKVHeads, N, d]), { requiresGrad: false })

  let out
  A.noGrad(() => {
    out = A.flashAttention(Q, K, V, { causal: true, numKVHeads })
  })
  expect(out.data.shape).toEqual([numQHeads, N, d])
})

test('GQA: numKVHeads == numQHeads equals standard MHA', () => {
  const numHeads = 2, N = 8, d = 4
  const qd = T.rand([numHeads, N, d])
  const kd = T.rand([numHeads, N, d])
  const vd = T.rand([numHeads, N, d])

  let outMHA, outGQA
  A.noGrad(() => {
    outMHA = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd), { causal: true })
    outGQA = A.flashAttention(
      A.variable(qd), A.variable(kd), A.variable(vd), { causal: true, numKVHeads: numHeads })
  })
  for (let i = 0; i < outMHA.data.data.length; i++) {
    expect(Math.abs(outMHA.data.data[i] - outGQA.data.data[i])).toBeLessThan(1e-6)
  }
})

test('GQA: Q heads sharing same KV head produce related outputs', () => {
  // With 4 Q heads and 2 KV heads: heads 0,1 share KV 0; heads 2,3 share KV 1
  const numQHeads = 4, numKVHeads = 2, N = 4, d = 4
  const Q = A.variable(T.rand([numQHeads, N, d]))
  const K = A.variable(T.rand([numKVHeads, N, d]))
  const V = A.variable(T.rand([numKVHeads, N, d]))

  let out
  A.noGrad(() => {
    out = A.flashAttention(Q, K, V, { causal: true, numKVHeads })
  })

  // Output should exist and have correct shape
  expect(out.data.shape).toEqual([numQHeads, N, d])
  // Outputs of heads 0,1 should use the same K,V (from KV head 0)
  // but different Q, so they'll differ. Just verify non-NaN.
  for (let i = 0; i < out.data.data.length; i++) {
    expect(isNaN(out.data.data[i])).toBe(false)
  }
})

test('GQA backward: produces correctly shaped gradients', () => {
  const numQHeads = 4, numKVHeads = 2, N = 8, d = 4
  const Q = A.variable(T.rand([numQHeads, N, d]), { requiresGrad: true })
  const K = A.variable(T.rand([numKVHeads, N, d]), { requiresGrad: true })
  const V = A.variable(T.rand([numKVHeads, N, d]), { requiresGrad: true })

  const out = A.flashAttention(Q, K, V, { causal: true, numKVHeads })
  const loss = A.sum(out)
  A.backward(loss)

  expect(Q.grad.shape).toEqual([numQHeads, N, d])
  expect(K.grad.shape).toEqual([numKVHeads, N, d])
  expect(V.grad.shape).toEqual([numKVHeads, N, d])
})

// --- Combined GQA + sliding window ---

test('GQA + sliding window: combined forward', () => {
  const numQHeads = 4, numKVHeads = 2, N = 16, d = 4
  const Q = A.variable(T.rand([numQHeads, N, d]))
  const K = A.variable(T.rand([numKVHeads, N, d]))
  const V = A.variable(T.rand([numKVHeads, N, d]))

  let out
  A.noGrad(() => {
    out = A.flashAttention(Q, K, V, { causal: true, numKVHeads, windowSize: 4 })
  })
  expect(out.data.shape).toEqual([numQHeads, N, d])
  for (let i = 0; i < out.data.data.length; i++) {
    expect(isNaN(out.data.data[i])).toBe(false)
  }
})

test('GQA + sliding window: backward produces gradients', () => {
  const numQHeads = 4, numKVHeads = 2, N = 16, d = 4
  const Q = A.variable(T.rand([numQHeads, N, d]), { requiresGrad: true })
  const K = A.variable(T.rand([numKVHeads, N, d]), { requiresGrad: true })
  const V = A.variable(T.rand([numKVHeads, N, d]), { requiresGrad: true })

  const out = A.flashAttention(Q, K, V, { causal: true, numKVHeads, windowSize: 4 })
  const loss = A.sum(out)
  A.backward(loss)

  expect(Q.grad).not.toBeNull()
  expect(K.grad).not.toBeNull()
  expect(V.grad).not.toBeNull()
  expect(Q.grad.shape).toEqual([numQHeads, N, d])
  expect(K.grad.shape).toEqual([numKVHeads, N, d])
  expect(V.grad.shape).toEqual([numKVHeads, N, d])
})
