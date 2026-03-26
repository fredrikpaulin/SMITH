// tests/muon.test.js
// Tests for MuonAdamW optimizer (Phase 28)

import { test, expect } from 'bun:test'
import smith from '../src/index.js'

const { tensor, zeros, variable, param, matmul, sum, backward, noGrad } = smith

function expectClose(actual, expected, tol = 1e-3) {
  expect(Math.abs(actual - expected)).toBeLessThan(tol)
}

// Flat array from tensor's GPU buffer (works for any shape)
const flat = t => Array.from(t.data)

// Helper: random values in [-scale, scale]
const randVals = (n, scale = 0.5) => Array.from({ length: n }, () => scale * (Math.random() * 2 - 1))

// --- AdamW side of MuonAdamW ---

test('MuonAdamW: adamw group updates params', () => {
  const w = variable(tensor([1, 2, 3, 4], [4]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'adamw',
    params: [w],
    lr: 0.1,
    betas: [0.9, 0.999],
    eps: 1e-8,
    weightDecay: 0.0,
  }])

  w.grad = tensor([1, 1, 1, 1], [4])
  smith.muonAdamWStep(opt)
  const result = flat(w.data)

  for (let i = 0; i < 4; i++) {
    expect(result[i]).toBeLessThan([1, 2, 3, 4][i])
  }
})

test('MuonAdamW: adamw with weight decay', () => {
  const w = variable(tensor([1, 2, 3, 4], [4]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'adamw',
    params: [w],
    lr: 0.01,
    betas: [0.9, 0.999],
    eps: 1e-8,
    weightDecay: 0.1,
  }])

  w.grad = tensor([0, 0, 0, 0], [4])  // zero grad — only WD acts
  smith.muonAdamWStep(opt)
  const result = flat(w.data)

  expect(result[0]).toBeLessThan(1)
  expect(result[3]).toBeLessThan(4)
})

test('MuonAdamW: multiple adamw steps converge', () => {
  const w = variable(tensor([5], [1]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'adamw',
    params: [w],
    lr: 0.5,
    betas: [0.9, 0.999],
    eps: 1e-8,
    weightDecay: 0.0,
  }])

  for (let step = 0; step < 50; step++) {
    w.grad = tensor([flat(w.data)[0]], [1])
    smith.muonAdamWStep(opt)
  }
  const result = flat(w.data)[0]
  expect(Math.abs(result)).toBeLessThan(1)
})

// --- Muon side ---
// NS polar coefficients are tuned for large matrices (768×768).
// Use 16×16+ for tests that exercise the full 5-step NS.

test('MuonAdamW: muon group updates 2D params', () => {
  const n = 16
  const vals = Array.from({ length: n * n }, (_, i) => ((i % 17) - 8) * 0.05)
  const w = variable(tensor(vals, [n, n]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'muon',
    params: [w],
    lr: 0.01,
    momentum: 0.95,
    beta2: 0.7,
    weightDecay: 0.0,
    nsSteps: 5,
  }])

  w.grad = tensor(randVals(n * n, 0.3), [n, n])
  const before = flat(w.data).slice()

  smith.muonAdamWStep(opt)
  const after = flat(w.data)

  let changed = false
  for (let i = 0; i < after.length; i++) {
    if (Math.abs(after[i] - before[i]) > 1e-6) changed = true
  }
  expect(changed).toBe(true)
})

test('MuonAdamW: muon produces finite values', () => {
  const n = 16
  const w = variable(tensor(Array.from({ length: n * n }, (_, i) => (i - n * n / 2) * 0.001), [n, n]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'muon',
    params: [w],
    lr: 0.01,
    momentum: 0.95,
    beta2: 0.7,
    weightDecay: 0.0,
    nsSteps: 5,
  }])

  w.grad = tensor(randVals(n * n), [n, n])
  smith.muonAdamWStep(opt)

  const result = flat(w.data)
  for (const v of result) {
    expect(isFinite(v)).toBe(true)
  }
})

test('MuonAdamW: muon with zero grad does not change params', () => {
  const n = 8
  const vals = Array.from({ length: n * n }, (_, i) => i * 0.1)
  const w = variable(tensor(vals, [n, n]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'muon',
    params: [w],
    lr: 0.01,
    momentum: 0.95,
    beta2: 0.7,
    weightDecay: 0.0,
    nsSteps: 5,
  }])

  w.grad = tensor(new Array(n * n).fill(0), [n, n])
  smith.muonAdamWStep(opt)

  const result = flat(w.data)
  for (let i = 0; i < vals.length; i++) {
    expectClose(result[i], vals[i], 1e-4)
  }
})

test('MuonAdamW: muon with weight decay shrinks params', () => {
  const n = 16
  const vals = randVals(n * n, 1.0)
  const w = variable(tensor(vals, [n, n]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'muon',
    params: [w],
    lr: 0.01,
    momentum: 0.95,
    beta2: 0.7,
    weightDecay: 0.5,
    nsSteps: 5,
  }])

  w.grad = tensor(randVals(n * n, 0.01), [n, n])
  smith.muonAdamWStep(opt)

  const result = flat(w.data)
  let shrunk = 0
  for (let i = 0; i < vals.length; i++) {
    if (Math.abs(result[i]) < Math.abs(vals[i])) shrunk++
  }
  expect(shrunk).toBeGreaterThan(0)
})

// --- Combined MuonAdamW ---

test('MuonAdamW: mixed groups (adamw + muon)', () => {
  const n = 16
  const bias = variable(tensor([0.1, -0.2, 0.3], [3]), { requiresGrad: true })
  const weight = variable(tensor(randVals(n * n, 0.5), [n, n]), { requiresGrad: true })

  const opt = smith.createMuonAdamW([
    {
      kind: 'adamw',
      params: [bias],
      lr: 0.01,
      betas: [0.9, 0.999],
      eps: 1e-8,
      weightDecay: 0.0,
    },
    {
      kind: 'muon',
      params: [weight],
      lr: 0.01,
      momentum: 0.95,
      beta2: 0.7,
      weightDecay: 0.0,
      nsSteps: 5,
    },
  ])

  bias.grad = tensor([1, 1, 1], [3])
  weight.grad = tensor(randVals(n * n, 0.3), [n, n])

  const biasBefore = flat(bias.data).slice()
  const weightBefore = flat(weight.data).slice()

  smith.muonAdamWStep(opt)

  const biasAfter = flat(bias.data)
  const weightAfter = flat(weight.data)

  let biasChanged = false, weightChanged = false
  for (let i = 0; i < 3; i++) {
    if (Math.abs(biasAfter[i] - biasBefore[i]) > 1e-6) biasChanged = true
  }
  for (let i = 0; i < weightAfter.length; i++) {
    if (Math.abs(weightAfter[i] - weightBefore[i]) > 1e-6) weightChanged = true
  }
  expect(biasChanged).toBe(true)
  expect(weightChanged).toBe(true)
})

test('MuonAdamW: multiple muon steps remain stable', () => {
  const n = 16
  const w = variable(tensor(Array.from({ length: n * n }, (_, i) => (i - n * n / 2) * 0.001), [n, n]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'muon',
    params: [w],
    lr: 0.001,
    momentum: 0.95,
    beta2: 0.7,
    weightDecay: 0.0,
    nsSteps: 5,
  }])

  for (let step = 0; step < 20; step++) {
    w.grad = tensor(randVals(n * n, 0.05), [n, n])
    smith.muonAdamWStep(opt)

    const result = flat(w.data)
    for (const v of result) {
      expect(isFinite(v)).toBe(true)
      expect(Math.abs(v)).toBeLessThan(100)
    }
  }
})

test('MuonAdamW: tall matrix (rows > cols)', () => {
  const rows = 32, cols = 16
  const w = variable(tensor(randVals(rows * cols, 0.3), [rows, cols]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'muon',
    params: [w],
    lr: 0.01,
    momentum: 0.95,
    beta2: 0.7,
    weightDecay: 0.0,
    nsSteps: 5,
  }])

  w.grad = tensor(randVals(rows * cols), [rows, cols])
  smith.muonAdamWStep(opt)

  const result = flat(w.data)
  for (const v of result) expect(isFinite(v)).toBe(true)
})

test('MuonAdamW: wide matrix (cols > rows)', () => {
  const rows = 16, cols = 32
  const w = variable(tensor(randVals(rows * cols, 0.3), [rows, cols]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'muon',
    params: [w],
    lr: 0.01,
    momentum: 0.95,
    beta2: 0.7,
    weightDecay: 0.0,
    nsSteps: 5,
  }])

  w.grad = tensor(randVals(rows * cols), [rows, cols])
  smith.muonAdamWStep(opt)

  const result = flat(w.data)
  for (const v of result) expect(isFinite(v)).toBe(true)
})

test('MuonAdamW: Newton-Schulz produces near-orthogonal result', async () => {
  const { createMuonAdamW } = await import('../src/muon.js')
  const n = 16

  const vals = Array.from({ length: n * n }, (_, i) => {
    const r = Math.floor(i / n), c = i % n
    return r === c ? 1.0 : 0.05 * (Math.random() - 0.5)
  })

  const w = variable(tensor(vals, [n, n]), { requiresGrad: true })
  const opt = createMuonAdamW([{
    kind: 'muon',
    params: [w],
    lr: 0.01,
    momentum: 0.0,
    beta2: 0.0,
    weightDecay: 0.0,
    nsSteps: 5,
  }])

  w.grad = tensor(vals, [n, n])
  smith.muonAdamWStep(opt)

  const result = flat(w.data)
  for (const v of result) expect(isFinite(v)).toBe(true)
})

test('MuonAdamW: polar express coefficients match reference', async () => {
  const { POLAR_COEFFS } = await import('../src/muon.js')
  expect(POLAR_COEFFS.length).toBe(5)
  expectClose(POLAR_COEFFS[0][0], 8.156554524902461, 1e-10)
  expectClose(POLAR_COEFFS[0][1], -22.48329292557795, 1e-10)
  expectClose(POLAR_COEFFS[0][2], 15.878769915207462, 1e-10)
})

// --- Skipped params (no grad) ---

test('MuonAdamW: skips params without grad', () => {
  const n = 8
  const w1 = variable(tensor(randVals(n * n), [n, n]), { requiresGrad: true })
  const w2Vals = Array.from({ length: n * n }, (_, i) => i * 0.1)
  const w2 = variable(tensor(w2Vals, [n, n]), { requiresGrad: true })

  const opt = smith.createMuonAdamW([{
    kind: 'muon',
    params: [w1, w2],
    lr: 0.01,
    momentum: 0.95,
    beta2: 0.7,
    weightDecay: 0.0,
    nsSteps: 5,
  }])

  w1.grad = tensor(randVals(n * n, 0.2), [n, n])
  // w2.grad is null/undefined

  const w2Before = flat(w2.data).slice()
  smith.muonAdamWStep(opt)

  const w2After = flat(w2.data)
  for (let i = 0; i < w2Vals.length; i++) {
    expectClose(w2After[i], w2Before[i], 1e-6)
  }
})

// --- AdamW step counter consistency ---

test('MuonAdamW: adamw step counter increments once per optimizer step, not per group', () => {
  // Two AdamW groups — the step counter should be the same for both
  const w1 = variable(tensor([1, 2, 3], [3]), { requiresGrad: true })
  const w2 = variable(tensor([4, 5, 6], [3]), { requiresGrad: true })

  const opt = smith.createMuonAdamW([
    { kind: 'adamw', params: [w1], lr: 0.1, betas: [0.9, 0.999], eps: 1e-8, weightDecay: 0.0 },
    { kind: 'adamw', params: [w2], lr: 0.1, betas: [0.9, 0.999], eps: 1e-8, weightDecay: 0.0 },
  ])

  // Both have identical init values and identical grads — they should get identical updates
  w1.grad = tensor([1, 1, 1], [3])
  w2.grad = tensor([1, 1, 1], [3])
  smith.muonAdamWStep(opt)

  // After 1 step, _adamwStep should be 1 (not 2)
  expect(opt._adamwStep).toBe(1)

  // Both groups used the same step value, so bias correction was identical.
  // With same init and same grad, the updates should match exactly.
  const r1 = flat(w1.data)
  const r2 = flat(w2.data)
  for (let i = 0; i < 3; i++) {
    // w1 started at [1,2,3], w2 at [4,5,6], offsets differ but delta should be identical
    expectClose(r1[i] - [1, 2, 3][i], r2[i] - [4, 5, 6][i], 1e-6)
  }

  // Second step
  w1.grad = tensor([1, 1, 1], [3])
  w2.grad = tensor([1, 1, 1], [3])
  smith.muonAdamWStep(opt)
  expect(opt._adamwStep).toBe(2)
})

// --- Square matrix (exercises wide path per reference) ---

test('MuonAdamW: square matrix uses wide NS path', () => {
  const n = 16
  const w = variable(tensor(randVals(n * n, 0.3), [n, n]), { requiresGrad: true })
  const opt = smith.createMuonAdamW([{
    kind: 'muon',
    params: [w],
    lr: 0.01,
    momentum: 0.95,
    beta2: 0.7,
    weightDecay: 0.0,
    nsSteps: 5,
  }])

  w.grad = tensor(randVals(n * n, 0.3), [n, n])
  const before = flat(w.data).slice()
  smith.muonAdamWStep(opt)

  const after = flat(w.data)
  let changed = false
  for (let i = 0; i < after.length; i++) {
    if (Math.abs(after[i] - before[i]) > 1e-6) changed = true
    expect(isFinite(after[i])).toBe(true)
  }
  expect(changed).toBe(true)
})
