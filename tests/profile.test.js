// tests/profile.test.js
// Phase 16: Profiling and benchmarking

import { test, expect } from 'bun:test'
import * as T from '../src/tensor.js'
import * as A from '../src/autograd.js'
import * as device from '../src/device.js'
import {
  enableProfiling, disableProfiling, isProfilingEnabled,
  report, resetProfile,
  profile, benchmark,
  memorySnapshot,
} from '../src/profile.js'

// =====================================================================
// Profiler state management
// =====================================================================

test('profiling is disabled by default', () => {
  disableProfiling()
  expect(isProfilingEnabled()).toBe(false)
})

test('enableProfiling / disableProfiling toggle state', () => {
  disableProfiling()
  expect(isProfilingEnabled()).toBe(false)
  enableProfiling()
  expect(isProfilingEnabled()).toBe(true)
  disableProfiling()
  expect(isProfilingEnabled()).toBe(false)
})

test('resetProfile clears all stats', () => {
  enableProfiling()
  // Trigger some GPU work
  const a = T.rand([64, 64])
  const b = T.rand([64, 64])
  const va = A.variable(a, { requiresGrad: false })
  const vb = A.variable(b, { requiresGrad: false })
  A.noGrad(() => { A.matmul(va, vb) })

  const r1 = report()
  expect(r1.dispatches).toBeGreaterThan(0)

  resetProfile()
  const r2 = report()
  expect(r2.dispatches).toBe(0)
  expect(r2.totalGpuMs).toBe(0)
  expect(r2.kernels.length).toBe(0)
  disableProfiling()
})

// =====================================================================
// profile(fn)
// =====================================================================

test('profile() wraps a function and returns timing', () => {
  const p = profile(() => {
    const a = T.rand([128, 128])
    const b = T.rand([128, 128])
    const va = A.variable(a, { requiresGrad: false })
    const vb = A.variable(b, { requiresGrad: false })
    return A.noGrad(() => A.matmul(va, vb))
  })

  expect(p.result).toBeDefined()
  expect(p.result.data).toBeDefined()
  expect(p.cpuMs).toBeGreaterThanOrEqual(0)
  expect(p.gpuMs).toBeGreaterThanOrEqual(0)
  expect(p.dispatches).toBeGreaterThan(0)
  expect(p.kernels.length).toBeGreaterThan(0)
  expect(p.memory).toBeDefined()
  expect(p.memory.currentBytes).toBeGreaterThan(0)
})

test('profile() does not affect numerical results', () => {
  // Run without profiling
  const a = T.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
  const b = T.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1], [3, 3])

  const va1 = A.variable(a, { requiresGrad: false })
  const vb1 = A.variable(b, { requiresGrad: false })
  let resultWithout
  A.noGrad(() => { resultWithout = A.matmul(va1, vb1) })
  const dataWithout = Array.from(resultWithout.data.data.slice(0, 9))

  // Run with profiling
  const p = profile(() => {
    const va2 = A.variable(a, { requiresGrad: false })
    const vb2 = A.variable(b, { requiresGrad: false })
    let r
    A.noGrad(() => { r = A.matmul(va2, vb2) })
    return r
  })
  const dataWith = Array.from(p.result.data.data.slice(0, 9))

  // Results must match exactly
  for (let i = 0; i < 9; i++) {
    expect(dataWith[i]).toBe(dataWithout[i])
  }
})

test('profile() reports per-kernel stats', () => {
  const p = profile(() => {
    const a = T.rand([64, 64])
    const va = A.variable(a, { requiresGrad: false })
    A.noGrad(() => {
      A.relu(va)
      A.gelu(va)
    })
  })

  // Should have at least some kernel entries
  expect(p.kernels.length).toBeGreaterThan(0)
  for (const k of p.kernels) {
    expect(k.kernel).toBeDefined()
    expect(typeof k.kernel).toBe('string')
    expect(k.calls).toBeGreaterThan(0)
    expect(k.totalMs).toBeGreaterThanOrEqual(0)
    expect(k.avgMs).toBeGreaterThanOrEqual(0)
    expect(k.minMs).toBeGreaterThanOrEqual(0)
    expect(k.maxMs).toBeGreaterThanOrEqual(k.minMs)
    expect(k.pct).toBeGreaterThanOrEqual(0)
    expect(k.pct).toBeLessThanOrEqual(100.001)
  }

  // Percentages should sum to ~100%
  const totalPct = p.kernels.reduce((s, k) => s + k.pct, 0)
  expect(totalPct).toBeGreaterThan(99)
  expect(totalPct).toBeLessThan(100.1)
})

test('profile() tracks GPU time > 0 for real work', () => {
  const p = profile(() => {
    const a = T.rand([256, 256])
    const b = T.rand([256, 256])
    const va = A.variable(a, { requiresGrad: false })
    const vb = A.variable(b, { requiresGrad: false })
    A.noGrad(() => { A.matmul(va, vb) })
  })

  // GPU time should be > 0 for a 256x256 matmul
  expect(p.gpuMs).toBeGreaterThan(0)
})

test('profile() restores previous profiling state', () => {
  disableProfiling()
  expect(isProfilingEnabled()).toBe(false)

  profile(() => { T.rand([4, 4]) })

  // Should be disabled again after profile()
  expect(isProfilingEnabled()).toBe(false)

  enableProfiling()
  profile(() => { T.rand([4, 4]) })

  // Should remain enabled (was enabled before)
  expect(isProfilingEnabled()).toBe(true)
  disableProfiling()
})

// =====================================================================
// benchmark()
// =====================================================================

test('benchmark() runs warmup + iterations and returns stats', () => {
  const result = benchmark('matmul-64', () => {
    const a = T.rand([64, 64])
    const b = T.rand([64, 64])
    const va = A.variable(a, { requiresGrad: false })
    const vb = A.variable(b, { requiresGrad: false })
    A.noGrad(() => { A.matmul(va, vb) })
  }, { warmup: 2, iterations: 5 })

  expect(result.name).toBe('matmul-64')
  expect(result.iterations).toBe(5)

  // CPU stats
  expect(result.cpu.mean).toBeGreaterThan(0)
  expect(result.cpu.median).toBeGreaterThan(0)
  expect(result.cpu.min).toBeGreaterThan(0)
  expect(result.cpu.max).toBeGreaterThanOrEqual(result.cpu.min)
  expect(result.cpu.p95).toBeGreaterThanOrEqual(result.cpu.min)
  expect(result.cpu.stddev).toBeGreaterThanOrEqual(0)

  // GPU stats
  expect(result.gpu.mean).toBeGreaterThanOrEqual(0)
  expect(result.gpu.median).toBeGreaterThanOrEqual(0)
  expect(result.gpu.min).toBeGreaterThanOrEqual(0)
  expect(result.gpu.max).toBeGreaterThanOrEqual(result.gpu.min)
})

test('benchmark() with default options works', () => {
  const result = benchmark('add-small', () => {
    const a = T.rand([32])
    const b = T.rand([32])
    const va = A.variable(a, { requiresGrad: false })
    const vb = A.variable(b, { requiresGrad: false })
    A.noGrad(() => { A.add(va, vb) })
  })

  expect(result.iterations).toBe(10)  // default
  expect(result.cpu).toBeDefined()
  expect(result.gpu).toBeDefined()
})

// =====================================================================
// Memory tracking
// =====================================================================

test('memorySnapshot returns allocated bytes', () => {
  const snap = memorySnapshot()
  expect(snap.allocatedBytes).toBeGreaterThan(0)
  expect(typeof snap.allocatedBytes).toBe('number')
})

test('memory increases after allocations', () => {
  const before = memorySnapshot().allocatedBytes
  // Allocate a large tensor
  const big = T.rand([1024, 1024])  // 4MB
  const after = memorySnapshot().allocatedBytes
  // Should have increased (at least ~4MB)
  expect(after).toBeGreaterThan(before)
})

test('profile() reports memory stats', () => {
  const p = profile(() => {
    // Allocate some tensors inside
    const a = T.rand([512, 512])
    const b = T.rand([512, 512])
    const va = A.variable(a, { requiresGrad: false })
    const vb = A.variable(b, { requiresGrad: false })
    A.noGrad(() => { A.matmul(va, vb) })
  })

  expect(p.memory.startBytes).toBeGreaterThan(0)
  expect(p.memory.peakBytes).toBeGreaterThanOrEqual(p.memory.startBytes)
  expect(p.memory.currentBytes).toBeGreaterThan(0)
})

// =====================================================================
// Inline profiling (enable/disable around manual work)
// =====================================================================

test('inline profiling collects stats across multiple dispatches', () => {
  enableProfiling()
  resetProfile()

  // Multiple ops
  const a = T.rand([64, 64])
  const b = T.rand([64, 64])
  const va = A.variable(a, { requiresGrad: false })
  const vb = A.variable(b, { requiresGrad: false })

  A.noGrad(() => {
    A.matmul(va, vb)
    A.relu(va)
    A.add(va, vb)
  })

  const r = report()
  expect(r.dispatches).toBeGreaterThanOrEqual(3)
  expect(r.kernels.length).toBeGreaterThanOrEqual(2) // at least matmul + relu + add
  expect(r.totalGpuMs).toBeGreaterThanOrEqual(0)

  disableProfiling()
})

test('report() sorts kernels by total time descending', () => {
  const p = profile(() => {
    const a = T.rand([128, 128])
    const b = T.rand([128, 128])
    const va = A.variable(a, { requiresGrad: false })
    const vb = A.variable(b, { requiresGrad: false })
    A.noGrad(() => {
      A.matmul(va, vb)
      A.relu(va)
    })
  })

  if (p.kernels.length >= 2) {
    for (let i = 1; i < p.kernels.length; i++) {
      expect(p.kernels[i - 1].totalMs).toBeGreaterThanOrEqual(p.kernels[i].totalMs)
    }
  }
})

// =====================================================================
// Observer effect: profiling shouldn't break things
// =====================================================================

test('profiling does not break backward pass', () => {
  enableProfiling()
  resetProfile()

  const x = A.variable(T.randn([4, 4]), { requiresGrad: true })
  const w = A.variable(T.randn([4, 4]), { requiresGrad: true })
  const out = A.matmul(x, w)
  const loss = A.sum(out)
  A.backward(loss)

  expect(x.grad).not.toBeNull()
  expect(w.grad).not.toBeNull()

  // Gradients should be finite
  for (let i = 0; i < x.grad.data.length; i++) {
    expect(isFinite(x.grad.data[i])).toBe(true)
  }

  const r = report()
  expect(r.dispatches).toBeGreaterThan(0)

  disableProfiling()
})

test('profiling does not break conv2d', () => {
  const p = profile(() => {
    const x = A.variable(T.randn([1, 3, 8, 8]), { requiresGrad: true })
    const w = A.variable(T.randn([4, 3, 3, 3]), { requiresGrad: true })
    const out = A.conv2d(x, w, null, { padding: 1 })
    const loss = A.sum(out)
    A.backward(loss)
    return { outShape: out.data.shape, hasGrad: x.grad !== null }
  })

  expect(p.result.outShape).toEqual([1, 4, 8, 8])
  expect(p.result.hasGrad).toBe(true)
  expect(p.dispatches).toBeGreaterThan(0)
})

// =====================================================================
// Edge cases
// =====================================================================

test('report() with no dispatches returns empty', () => {
  enableProfiling()
  resetProfile()
  const r = report()
  expect(r.dispatches).toBe(0)
  expect(r.totalGpuMs).toBe(0)
  expect(r.kernels).toEqual([])
  disableProfiling()
})

test('benchmark with 1 iteration works', () => {
  const result = benchmark('single', () => {
    T.rand([16, 16])
  }, { warmup: 1, iterations: 1 })

  expect(result.iterations).toBe(1)
  expect(result.cpu.mean).toBeGreaterThan(0)
  expect(result.cpu.median).toBeGreaterThan(0)
  expect(result.cpu.min).toBe(result.cpu.max)
})
