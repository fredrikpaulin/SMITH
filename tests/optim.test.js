// smith/tests/optim.test.js
// Tests for AdamW optimizer, LR schedule, grad clipping, and end-to-end MLP training.

import { test, expect } from 'bun:test'
import smith from '../src/index.js'

// --- Unit tests ---

test('createSchedule + getLr: warmup then cosine', () => {
  const sched = smith.createSchedule({ warmupSteps: 4, totalSteps: 20, maxLr: 0.01, minLr: 0.001 })

  // Warmup: linear ramp
  const lr0 = smith.getLr(sched, 0) // (0+1)/4 * 0.01 = 0.0025
  const lr3 = smith.getLr(sched, 3) // (3+1)/4 * 0.01 = 0.01
  expect(Math.abs(lr0 - 0.0025)).toBeLessThan(1e-8)
  expect(Math.abs(lr3 - 0.01)).toBeLessThan(1e-8)

  // After warmup: cosine decay
  const lrMid = smith.getLr(sched, 12) // halfway through decay
  expect(lrMid).toBeLessThan(0.01)
  expect(lrMid).toBeGreaterThan(0.001)

  // End: should approach minLr
  const lrEnd = smith.getLr(sched, 20)
  expect(Math.abs(lrEnd - 0.001)).toBeLessThan(1e-6)
})

test('clipGradNorm scales gradients', () => {
  // Create a param with a known gradient
  const w = smith.param([4], () => smith.ones([4]))
  w.grad = smith.tensor([3, 4, 0, 0], [4]) // norm = 5

  const norm = smith.clipGradNorm([w], 2.5) // maxNorm = 2.5, scale = 0.5
  expect(Math.abs(norm - 5)).toBeLessThan(1e-4)

  const g = smith.toArray(w.grad)
  expect(Math.abs(g[0] - 1.5)).toBeLessThan(1e-4) // 3 * 0.5
  expect(Math.abs(g[1] - 2.0)).toBeLessThan(1e-4) // 4 * 0.5
})

test('clipGradNorm no-op when below threshold', () => {
  const w = smith.param([4], () => smith.ones([4]))
  w.grad = smith.tensor([0.1, 0.1, 0.1, 0.1], [4]) // norm ≈ 0.2

  const norm = smith.clipGradNorm([w], 1.0)
  expect(norm).toBeLessThan(1.0)

  // Gradients unchanged
  const g = smith.toArray(w.grad)
  expect(Math.abs(g[0] - 0.1)).toBeLessThan(1e-6)
})

test('adamwStep updates weights', () => {
  const w = smith.param([4], () => smith.full([4], 1.0))
  w.grad = smith.full(w.data.shape, 0.1)

  const opt = smith.createAdamW([w], { lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0 })
  smith.adamwStep(opt)

  // After 1 step, weights should have decreased
  const vals = smith.toArray(w.data)
  for (const v of vals) expect(v).toBeLessThan(1.0)
})

// --- End-to-end MLP training test ---
// Tiny 2-layer MLP learns to approximate y = sum(x) on random vectors.
// If the loss drops substantially over 50 steps, the full pipeline works:
// forward → backward → grad clipping → AdamW GPU step → repeat.

test('MLP training: loss decreases over 50 steps', () => {
  const HIDDEN = 8
  const IN = 4

  // Parameters
  const w1 = smith.param([IN, HIDDEN], () => {
    const t = smith.randn([IN, HIDDEN])
    // Scale init by 1/sqrt(IN)
    const s = 1 / Math.sqrt(IN)
    for (let i = 0; i < t.data.length; i++) t.data[i] *= s
    return t
  })
  const w2 = smith.param([HIDDEN, 1], () => {
    const t = smith.randn([HIDDEN, 1])
    const s = 1 / Math.sqrt(HIDDEN)
    for (let i = 0; i < t.data.length; i++) t.data[i] *= s
    return t
  })

  const params = [w1, w2]
  const opt = smith.createAdamW(params, { lr: 0.01, weightDecay: 0 })

  // Simple MSE: loss = (pred - target)^2
  function trainStep() {
    smith.zeroGrad(params)

    // Generate random input and target = sum(input)
    const xData = []
    let target = 0
    for (let i = 0; i < IN; i++) {
      const v = Math.random() * 2 - 1
      xData.push(v)
      target += v
    }

    const x = smith.variable(smith.tensor(xData, [1, IN]))
    const t = target

    // Forward: x @ w1 → relu → @ w2 → scalar
    const h = smith.relu(smith.matmul(x, w1))
    const pred = smith.matmul(h, w2) // [1, 1]

    // MSE: (pred - target)^2
    const tVar = smith.variable(smith.tensor([t], [1, 1]))
    const diff = smith.sub(pred, tVar)
    const loss = smith.sum(smith.mul(diff, diff))

    smith.backward(loss)
    smith.clipGradNorm(params, 1.0)
    smith.adamwStep(opt)

    return smith.toArray(loss.data)
  }

  // Collect losses
  const losses = []
  for (let step = 0; step < 200; step++) {
    losses.push(trainStep())
  }

  // Smooth comparison: average of first 20 vs last 20
  const firstAvg = losses.slice(0, 20).reduce((a, b) => a + b, 0) / 20
  const lastAvg = losses.slice(-20).reduce((a, b) => a + b, 0) / 20

  // Loss should decrease — the last 20 should be lower than the first 20
  expect(lastAvg).toBeLessThan(firstAvg)
})
