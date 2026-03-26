// tests/div_gather.test.js
// Phase 25: Autograd div, gather, scatter tests.
// Tests forward correctness and backward (gradient) correctness for all three ops.

import { test, expect, describe } from 'bun:test'
import smith from '../src/index.js'
import { gather, scatterAdd, scatter } from '../src/ops/gather.js'
import * as T from '../src/tensor.js'

function expectClose(actual, expected, tol = 1e-3) {
  if (Array.isArray(expected)) {
    expect(actual.length).toBe(expected.length)
    for (let i = 0; i < expected.length; i++) expectClose(actual[i], expected[i], tol)
  } else {
    expect(Math.abs(actual - expected)).toBeLessThan(tol)
  }
}

const f = t => Array.from(t.data)

// ============================================================
// div
// ============================================================

describe('div forward', () => {
  test('elementwise division', () => {
    const a = T.tensor([6, 10, 15], [3])
    const b = T.tensor([2, 5, 3], [3])
    const c = smith.div(
      smith.variable(a, { requiresGrad: false }),
      smith.variable(b, { requiresGrad: false })
    )
    expectClose(f(c.data), [3, 2, 5])
  })

  test('division by 1 is identity', () => {
    const a = T.tensor([7, 8, 9], [3])
    const b = T.tensor([1, 1, 1], [3])
    const c = smith.div(
      smith.variable(a),
      smith.variable(b)
    )
    expectClose(f(c.data), [7, 8, 9])
  })
})

describe('div backward', () => {
  test('gradients: dA = dOut/b, dB = -dOut*a/b²', () => {
    const a = smith.variable(smith.tensor([6, 10], [2]), { requiresGrad: true })
    const b = smith.variable(smith.tensor([2, 5], [2]), { requiresGrad: true })
    const c = smith.div(a, b)
    const loss = smith.sum(c)
    smith.backward(loss)

    // dA = 1/b = [0.5, 0.2]
    expectClose(smith.toArray(a.grad), [0.5, 0.2])
    // dB = -a/b² = [-6/4, -10/25] = [-1.5, -0.4]
    expectClose(smith.toArray(b.grad), [-1.5, -0.4])
  })

  test('gradient chain through div', () => {
    const a = smith.variable(smith.tensor([4, 9], [2]), { requiresGrad: true })
    const b = smith.variable(smith.tensor([2, 3], [2]), { requiresGrad: true })
    const c = smith.div(a, b)           // [2, 3]
    const d = smith.scale(c, 2.0)       // [4, 6]
    const loss = smith.sum(d)           // 10
    smith.backward(loss)

    // dL/dc = [2, 2] (from scale)
    // dA = dL/dc / b = [2/2, 2/3] = [1, 0.6667]
    expectClose(smith.toArray(a.grad), [1.0, 0.6667])
    // dB = -dL/dc * a / b² = [-2*4/4, -2*9/9] = [-2, -2]
    expectClose(smith.toArray(b.grad), [-2.0, -2.0])
  })
})

// ============================================================
// gather (raw GPU op)
// ============================================================

describe('gather forward (GPU)', () => {
  test('1D gather', () => {
    // input: [10, 20, 30, 40, 50], indices: [1, 3, 0]
    const input = T.tensor([10, 20, 30, 40, 50], [5])
    const out = gather(input, 0, [1, 3, 0])
    expectClose(f(out), [20, 40, 10])
    expect(out.shape).toEqual([3])
  })

  test('2D gather along axis 0', () => {
    // input: [[1,2],[3,4],[5,6]] shape [3,2], gather rows [2, 0]
    const input = T.tensor([1, 2, 3, 4, 5, 6], [3, 2])
    const out = gather(input, 0, [2, 0])
    expect(out.shape).toEqual([2, 2])
    expectClose(f(out), [5, 6, 1, 2])
  })

  test('2D gather along axis 1', () => {
    // input: [[10,20,30],[40,50,60]] shape [2,3], gather cols [2, 0]
    const input = T.tensor([10, 20, 30, 40, 50, 60], [2, 3])
    const out = gather(input, 1, [2, 0])
    expect(out.shape).toEqual([2, 2])
    expectClose(f(out), [30, 10, 60, 40])
  })

  test('duplicate indices', () => {
    const input = T.tensor([10, 20, 30], [3])
    const out = gather(input, 0, [1, 1, 1])
    expectClose(f(out), [20, 20, 20])
  })

  test('single index', () => {
    const input = T.tensor([10, 20, 30], [3])
    const out = gather(input, 0, [2])
    expect(out.shape).toEqual([1])
    expectClose(f(out), [30])
  })
})

// ============================================================
// scatterAdd (raw GPU op)
// ============================================================

describe('scatterAdd (GPU)', () => {
  test('1D scatter add', () => {
    const dst = T.zeros([5])
    const src = T.tensor([100, 200, 300], [3])
    scatterAdd(dst, 0, [0, 2, 4], src)
    expectClose(f(dst), [100, 0, 200, 0, 300])
  })

  test('duplicate indices accumulate', () => {
    const dst = T.zeros([3])
    const src = T.tensor([10, 20, 30], [3])
    scatterAdd(dst, 0, [1, 1, 1], src)
    expectClose(f(dst), [0, 60, 0])
  })

  test('2D scatter add along axis 0', () => {
    // dst [3,2] = zeros, src [2,2], indices [1, 0] → rows 1 and 0
    const dst = T.zeros([3, 2])
    const src = T.tensor([10, 20, 30, 40], [2, 2])
    scatterAdd(dst, 0, [1, 0], src)
    // row 0 += [30,40], row 1 += [10,20], row 2 unchanged
    expectClose(f(dst), [30, 40, 10, 20, 0, 0])
  })
})

// ============================================================
// scatter (raw GPU op)
// ============================================================

describe('scatter forward (GPU)', () => {
  test('1D scatter overwrites', () => {
    const input = T.tensor([1, 2, 3, 4, 5], [5])
    const src = T.tensor([99, 88], [2])
    const out = scatter(input, 0, [1, 3], src)
    expectClose(f(out), [1, 99, 3, 88, 5])
  })

  test('scatter preserves un-indexed positions', () => {
    const input = T.tensor([10, 20, 30], [3])
    const src = T.tensor([77], [1])
    const out = scatter(input, 0, [0], src)
    expectClose(f(out), [77, 20, 30])
  })
})

// ============================================================
// gather/scatter autograd
// ============================================================

describe('gather autograd', () => {
  test('gather forward through autograd', () => {
    const w = smith.variable(smith.tensor([10, 20, 30, 40, 50], [5]), { requiresGrad: true })
    const out = smith.gather(w, 0, [1, 3])
    expectClose(smith.toArray(out.data), [20, 40])
  })

  test('gather backward = scatter-add of gradient', () => {
    const w = smith.variable(smith.tensor([10, 20, 30, 40, 50], [5]), { requiresGrad: true })
    const out = smith.gather(w, 0, [1, 3])
    const loss = smith.sum(out)
    smith.backward(loss)
    // grad flows to indices 1 and 3 only
    expectClose(smith.toArray(w.grad), [0, 1, 0, 1, 0])
  })

  test('gather backward with duplicate indices accumulates', () => {
    const w = smith.variable(smith.tensor([10, 20, 30], [3]), { requiresGrad: true })
    const out = smith.gather(w, 0, [1, 1, 1])
    const loss = smith.sum(out)
    smith.backward(loss)
    // index 1 gathered 3 times → grad accumulates to 3
    expectClose(smith.toArray(w.grad), [0, 3, 0])
  })

  test('gather backward 2D', () => {
    // input [3,2], gather rows [0, 2] → output [2,2]
    const w = smith.variable(smith.tensor([1, 2, 3, 4, 5, 6], [3, 2]), { requiresGrad: true })
    const out = smith.gather(w, 0, [0, 2])
    const loss = smith.sum(out)
    smith.backward(loss)
    // rows 0 and 2 get grad 1, row 1 gets 0
    expectClose(smith.toArray(w.grad), [[1, 1], [0, 0], [1, 1]])
  })
})

describe('scatter autograd', () => {
  test('scatter forward through autograd', () => {
    const base = smith.variable(smith.tensor([1, 2, 3, 4, 5], [5]), { requiresGrad: true })
    const src = smith.variable(smith.tensor([99, 88], [2]), { requiresGrad: true })
    const out = smith.scatter(base, 0, [1, 3], src)
    expectClose(smith.toArray(out.data), [1, 99, 3, 88, 5])
  })

  test('scatter backward: src gets gathered grad', () => {
    const base = smith.variable(smith.tensor([1, 2, 3, 4, 5], [5]), { requiresGrad: true })
    const src = smith.variable(smith.tensor([99, 88], [2]), { requiresGrad: true })
    const out = smith.scatter(base, 0, [1, 3], src)
    const loss = smith.sum(out)
    smith.backward(loss)
    // dSrc = gather(grad, indices) = [1, 1] (grad is all ones from sum)
    expectClose(smith.toArray(src.grad), [1, 1])
    // dBase = grad with scattered positions zeroed = [1, 0, 1, 0, 1]
    expectClose(smith.toArray(base.grad), [1, 0, 1, 0, 1])
  })
})

// ============================================================
// Round-trip: gather then scatter-add recovers original (for unique indices)
// ============================================================

describe('round-trip', () => {
  test('gather + scatter_add round trip', () => {
    const input = T.tensor([10, 20, 30, 40, 50], [5])
    const indices = [0, 2, 4]
    const gathered = gather(input, 0, indices)
    const dst = T.zeros([5])
    scatterAdd(dst, 0, indices, gathered)
    // Only gathered positions are filled
    expectClose(f(dst), [10, 0, 30, 0, 50])
  })

  test('gather all indices = identity', () => {
    const input = T.tensor([10, 20, 30], [3])
    const out = gather(input, 0, [0, 1, 2])
    expectClose(f(out), [10, 20, 30])
  })
})

// ============================================================
// Combined: div + gather in one autograd pipeline
// ============================================================

// ============================================================
// Larger-scale GPU tests
// ============================================================

describe('large scale GPU', () => {
  test('gather + scatter_add round trip (256 elements)', () => {
    const N = 256
    const data = Array.from({ length: N }, () => Math.random() * 100)
    const input = T.tensor(data, [N])
    // Gather all odd indices, scatter them back
    const indices = Array.from({ length: N / 2 }, (_, i) => i * 2 + 1)
    const gathered = gather(input, 0, indices)
    expect(gathered.shape).toEqual([N / 2])

    const dst = T.zeros([N])
    scatterAdd(dst, 0, indices, gathered)
    const result = f(dst)
    for (let i = 0; i < N; i++) {
      if (i % 2 === 1) {
        expect(Math.abs(result[i] - data[i])).toBeLessThan(1e-3)
      } else {
        expect(result[i]).toBe(0)
      }
    }
  })

  test('2D gather + scatter_add round trip (64x8)', () => {
    const rows = 64, cols = 8
    const data = Array.from({ length: rows * cols }, () => Math.random() * 10)
    const input = T.tensor(data, [rows, cols])
    // Gather every 4th row
    const indices = Array.from({ length: rows / 4 }, (_, i) => i * 4)
    const gathered = gather(input, 0, indices)
    expect(gathered.shape).toEqual([rows / 4, cols])

    const dst = T.zeros([rows, cols])
    scatterAdd(dst, 0, indices, gathered)
    const result = f(dst)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c
        if (r % 4 === 0) {
          expect(Math.abs(result[idx] - data[idx])).toBeLessThan(1e-3)
        } else {
          expect(result[idx]).toBe(0)
        }
      }
    }
  })

  test('div finite-diff gradient (64 elements)', () => {
    const N = 64, eps = 1e-4
    const aData = Array.from({ length: N }, () => Math.random() * 5 + 1)
    const bData = Array.from({ length: N }, () => Math.random() * 3 + 0.5) // avoid near-zero
    const a = smith.variable(smith.tensor(aData, [N]), { requiresGrad: true })
    const b = smith.variable(smith.tensor(bData, [N]))

    smith.zeroGrad([a])
    const loss = smith.sum(smith.div(a, b))
    smith.backward(loss)
    const analyticGrad = Array.from(a.grad.data)

    // Spot-check 10 positions
    const ad = a.data.data
    for (let trial = 0; trial < 10; trial++) {
      const i = Math.floor(Math.random() * N)
      const orig = ad[i]
      ad[i] = orig + eps
      const lPlus = smith.noGrad(() => smith.toArray(smith.sum(smith.div(a, b)).data))
      ad[i] = orig - eps
      const lMinus = smith.noGrad(() => smith.toArray(smith.sum(smith.div(a, b)).data))
      ad[i] = orig
      const numGrad = (lPlus - lMinus) / (2 * eps)
      expect(Math.abs(analyticGrad[i] - numGrad)).toBeLessThan(0.1)
    }
  })

  test('gather autograd backward (128 elements, 32 indices)', () => {
    const N = 128, K = 32
    const data = Array.from({ length: N }, () => Math.random() * 10)
    const indices = Array.from({ length: K }, () => Math.floor(Math.random() * N))
    const w = smith.variable(smith.tensor(data, [N]), { requiresGrad: true })
    const out = smith.gather(w, 0, indices)
    const loss = smith.sum(out)
    smith.backward(loss)

    // Each index i should accumulate count(i in indices) as its gradient
    const grad = Array.from(w.grad.data)
    const counts = new Float32Array(N)
    for (const idx of indices) counts[idx]++
    for (let i = 0; i < N; i++) {
      expect(Math.abs(grad[i] - counts[i])).toBeLessThan(1e-3)
    }
  })
})

describe('combined pipeline', () => {
  test('div then gather backward', () => {
    const a = smith.variable(smith.tensor([6, 10, 15, 20], [4]), { requiresGrad: true })
    const b = smith.variable(smith.tensor([2, 5, 3, 4], [4]), { requiresGrad: true })
    const c = smith.div(a, b)             // [3, 2, 5, 5]
    const d = smith.gather(c, 0, [0, 2])  // [3, 5]
    const loss = smith.sum(d)             // 8
    smith.backward(loss)

    // d's grad = [1, 1]
    // c's grad from gather backward: [1, 0, 1, 0] (only indices 0, 2)
    // dA = c_grad / b = [1/2, 0, 1/3, 0] = [0.5, 0, 0.333, 0]
    expectClose(smith.toArray(a.grad), [0.5, 0, 0.3333, 0])
  })
})
