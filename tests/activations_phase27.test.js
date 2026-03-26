// smith/tests/activations_phase27.test.js
// Tests for Phase 27: tanh, sigmoid, reluSquared activations + autograd

import { test, expect } from 'bun:test'
import smith from '../src/index.js'

const T = { tensor: smith.tensor, zeros: smith.zeros, ones: smith.ones, rand: smith.rand, toArray: smith.toArray }
const A = {
  variable: smith.variable, param: smith.param, backward: smith.backward,
  zeroGrad: smith.zeroGrad, noGrad: smith.noGrad,
  add: smith.add, mul: smith.mul, scale: smith.scale, sum: smith.sum,
  tanh: smith.tanh, sigmoid: smith.sigmoid, reluSquared: smith.reluSquared,
  relu: smith.relu,
}

function expectClose(actual, expected, tol = 1e-3) {
  if (Array.isArray(expected)) {
    expect(actual.length).toBe(expected.length)
    for (let i = 0; i < expected.length; i++) expectClose(actual[i], expected[i], tol)
  } else {
    expect(Math.abs(actual - expected)).toBeLessThan(tol)
  }
}

// --- Tanh ---

test('tanh forward: known values', () => {
  const input = A.variable(T.tensor([-2, -1, 0, 1, 2], [5]))
  let out
  A.noGrad(() => { out = A.tanh(input) })
  const result = T.toArray(out.data)
  expectClose(result, [Math.tanh(-2), Math.tanh(-1), 0, Math.tanh(1), Math.tanh(2)])
})

test('tanh forward: zeros gives zeros', () => {
  const input = A.variable(T.zeros([4]))
  let out
  A.noGrad(() => { out = A.tanh(input) })
  const result = T.toArray(out.data)
  expectClose(result, [0, 0, 0, 0])
})

test('tanh backward: gradient check', () => {
  const a = A.variable(T.tensor([0.5, -0.5, 1.0, -1.0], [4]), { requiresGrad: true })
  const out = A.tanh(a)
  const loss = A.sum(out)
  A.backward(loss)
  const g = T.toArray(a.grad)
  // d/dx tanh(x) = 1 - tanh(x)^2
  for (let i = 0; i < 4; i++) {
    const x = [0.5, -0.5, 1.0, -1.0][i]
    const t = Math.tanh(x)
    expectClose(g[i], 1 - t * t, 1e-3)
  }
})

test('tanh backward: chained', () => {
  const a = A.variable(T.tensor([1, 2, 3], [3]), { requiresGrad: true })
  const out = A.scale(A.tanh(a), 2.0)
  const loss = A.sum(out)
  A.backward(loss)
  expect(a.grad).not.toBeNull()
  const g = T.toArray(a.grad)
  for (let i = 0; i < 3; i++) {
    const x = [1, 2, 3][i]
    const t = Math.tanh(x)
    expectClose(g[i], 2.0 * (1 - t * t), 1e-3)
  }
})

// --- Sigmoid ---

test('sigmoid forward: known values', () => {
  const input = A.variable(T.tensor([-2, -1, 0, 1, 2], [5]))
  let out
  A.noGrad(() => { out = A.sigmoid(input) })
  const result = T.toArray(out.data)
  const sig = x => 1 / (1 + Math.exp(-x))
  expectClose(result, [-2, -1, 0, 1, 2].map(sig))
})

test('sigmoid forward: zero gives 0.5', () => {
  const input = A.variable(T.tensor([0], [1]))
  let out
  A.noGrad(() => { out = A.sigmoid(input) })
  expectClose(T.toArray(out.data), [0.5])
})

test('sigmoid backward: gradient check', () => {
  const a = A.variable(T.tensor([0, 1, -1], [3]), { requiresGrad: true })
  const out = A.sigmoid(a)
  const loss = A.sum(out)
  A.backward(loss)
  const g = T.toArray(a.grad)
  const sig = x => 1 / (1 + Math.exp(-x))
  for (let i = 0; i < 3; i++) {
    const x = [0, 1, -1][i]
    const s = sig(x)
    expectClose(g[i], s * (1 - s), 1e-3)
  }
})

// --- ReluSquared ---

test('reluSquared forward: known values', () => {
  const input = A.variable(T.tensor([-2, -1, 0, 1, 2, 3], [6]))
  let out
  A.noGrad(() => { out = A.reluSquared(input) })
  const result = T.toArray(out.data)
  expectClose(result, [0, 0, 0, 1, 4, 9])
})

test('reluSquared forward: negative inputs are zero', () => {
  const input = A.variable(T.tensor([-5, -0.1, -100], [3]))
  let out
  A.noGrad(() => { out = A.reluSquared(input) })
  const result = T.toArray(out.data)
  expectClose(result, [0, 0, 0])
})

test('reluSquared backward: gradient check', () => {
  const a = A.variable(T.tensor([-1, 0, 1, 2, 3], [5]), { requiresGrad: true })
  const out = A.reluSquared(a)
  const loss = A.sum(out)
  A.backward(loss)
  const g = T.toArray(a.grad)
  // d/dx relu(x)^2 = 2*relu(x) = 2*max(0,x)
  expectClose(g, [0, 0, 2, 4, 6])
})

test('reluSquared backward: chained with scale', () => {
  const a = A.variable(T.tensor([1, 2, -1], [3]), { requiresGrad: true })
  const out = A.scale(A.reluSquared(a), 0.5)
  const loss = A.sum(out)
  A.backward(loss)
  const g = T.toArray(a.grad)
  // d/dx 0.5 * relu(x)^2 = 0.5 * 2 * relu(x) = max(0, x)
  expectClose(g, [1, 2, 0])
})

test('reluSquared matches relu then square', () => {
  // relu(x)^2 should match manual relu + mul(x, x)
  const vals = T.tensor([1, -1, 2, -2, 0.5], [5])
  let outFused
  A.noGrad(() => { outFused = A.reluSquared(A.variable(vals)) })
  let outManual
  A.noGrad(() => {
    const r = A.relu(A.variable(vals))
    outManual = A.mul(r, r)
  })
  const f = T.toArray(outFused.data)
  const m = T.toArray(outManual.data)
  for (let i = 0; i < 5; i++) expectClose(f[i], m[i], 1e-6)
})

// --- Soft-capping (composed from tanh + scale) ---

test('soft-capping: softcap * tanh(x / softcap)', () => {
  const softcap = 15
  const x = A.variable(T.tensor([0, 10, 30, -20, 100], [5]))
  let out
  A.noGrad(() => {
    const scaled = A.scale(x, 1 / softcap)
    const t = A.tanh(scaled)
    out = A.scale(t, softcap)
  })
  const result = T.toArray(out.data)
  for (let i = 0; i < 5; i++) {
    const v = [0, 10, 30, -20, 100][i]
    expectClose(result[i], softcap * Math.tanh(v / softcap), 1e-3)
  }
})

test('soft-capping: clamps large logits', () => {
  const softcap = 15
  const x = A.variable(T.tensor([100, -100], [2]))
  let out
  A.noGrad(() => {
    out = A.scale(A.tanh(A.scale(x, 1 / softcap)), softcap)
  })
  const result = T.toArray(out.data)
  // tanh(1000/15) ≈ 1, so output ≈ 15
  expectClose(result[0], 15, 0.01)
  expectClose(result[1], -15, 0.01)
})

test('soft-capping backward', () => {
  const softcap = 15
  const a = A.variable(T.tensor([5, 10, -5], [3]), { requiresGrad: true })
  const scaled = A.scale(a, 1 / softcap)
  const t = A.tanh(scaled)
  const out = A.scale(t, softcap)
  const loss = A.sum(out)
  A.backward(loss)
  const g = T.toArray(a.grad)
  // d/dx softcap * tanh(x/softcap) = 1 - tanh(x/softcap)^2
  for (let i = 0; i < 3; i++) {
    const x = [5, 10, -5][i]
    const t = Math.tanh(x / softcap)
    expectClose(g[i], 1 - t * t, 1e-3)
  }
})

// --- 2D tensor support ---

test('tanh on 2D tensor', () => {
  const input = A.variable(T.tensor([1, -1, 0, 2], [2, 2]))
  let out
  A.noGrad(() => { out = A.tanh(input) })
  expect(out.data.shape).toEqual([2, 2])
  const flat = out.data.data
  expectClose(flat[0], Math.tanh(1))
  expectClose(flat[1], Math.tanh(-1))
})

test('sigmoid on 2D tensor', () => {
  const input = A.variable(T.tensor([0, 1, -1, 2], [2, 2]))
  let out
  A.noGrad(() => { out = A.sigmoid(input) })
  expect(out.data.shape).toEqual([2, 2])
})

test('reluSquared on 2D tensor', () => {
  const input = A.variable(T.tensor([1, -1, 2, -2], [2, 2]))
  let out
  A.noGrad(() => { out = A.reluSquared(input) })
  expect(out.data.shape).toEqual([2, 2])
  expectClose(out.data.data[0], 1)
  expectClose(out.data.data[1], 0)
  expectClose(out.data.data[2], 4)
  expectClose(out.data.data[3], 0)
})
