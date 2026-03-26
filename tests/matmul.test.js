// smith/tests/matmul.test.js
import { test, expect } from 'bun:test'
import smith from '../src/index.js'

function expectClose(actual, expected, tol = 1e-4) {
  if (Array.isArray(expected)) {
    for (let i = 0; i < expected.length; i++) {
      expectClose(actual[i], expected[i], tol)
    }
  } else {
    expect(Math.abs(actual - expected)).toBeLessThan(tol)
  }
}

test('2x2 matmul', () => {
  const a = smith.tensor([1, 2, 3, 4], [2, 2])
  const b = smith.tensor([5, 6, 7, 8], [2, 2])
  const va = smith.variable(a)
  const vb = smith.variable(b)
  const vc = smith.matmul(va, vb)
  const result = smith.toArray(vc.data)
  // [1*5+2*7, 1*6+2*8] = [19, 22]
  // [3*5+4*7, 3*6+4*8] = [43, 50]
  expectClose(result, [[19, 22], [43, 50]])
})

test('matmul identity', () => {
  const a = smith.tensor([1, 2, 3, 4], [2, 2])
  const eye = smith.tensor([1, 0, 0, 1], [2, 2])
  const va = smith.variable(a)
  const ve = smith.variable(eye)
  const vc = smith.matmul(va, ve)
  expectClose(smith.toArray(vc.data), [[1, 2], [3, 4]])
})

test('non-square matmul', () => {
  // [2, 3] @ [3, 2] = [2, 2]
  const a = smith.tensor([1, 2, 3, 4, 5, 6], [2, 3])
  const b = smith.tensor([7, 8, 9, 10, 11, 12], [3, 2])
  const va = smith.variable(a)
  const vb = smith.variable(b)
  const vc = smith.matmul(va, vb)
  expect(vc.data.shape).toEqual([2, 2])
  const result = smith.toArray(vc.data)
  // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
  // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
  expectClose(result, [[58, 64], [139, 154]])
})

test('matmul backward (gradient check)', () => {
  const a = smith.tensor([1, 2, 3, 4], [2, 2])
  const b = smith.tensor([5, 6, 7, 8], [2, 2])
  const va = smith.variable(a, { requiresGrad: true })
  const vb = smith.variable(b, { requiresGrad: true })

  const vc = smith.matmul(va, vb)
  // Sum all elements to get a scalar loss
  const loss = smith.sum(vc)
  smith.backward(loss)

  // dL/dA = grad @ B^T, where grad = ones(2,2)
  // B^T = [[5,7],[6,8]]
  // dL/dA = [[5+6, 7+8], [5+6, 7+8]] = [[11, 15], [11, 15]]
  expectClose(smith.toArray(va.grad), [[11, 15], [11, 15]])

  // dL/dB = A^T @ grad
  // A^T = [[1,3],[2,4]]
  // dL/dB = [[1+3, 1+3], [2+4, 2+4]] = [[4, 4], [6, 6]]
  expectClose(smith.toArray(vb.grad), [[4, 4], [6, 6]])
})

// --- Larger GPU-verified matmul tests ---

test('16x16 matmul: A @ I = A', () => {
  const N = 16
  const aData = Array.from({ length: N * N }, () => Math.random() * 2 - 1)
  const eyeData = new Float32Array(N * N)
  for (let i = 0; i < N; i++) eyeData[i * N + i] = 1
  const va = smith.variable(smith.tensor(aData, [N, N]))
  const ve = smith.variable(smith.tensor(Array.from(eyeData), [N, N]))
  const result = smith.matmul(va, ve)
  const resultArr = Array.from(result.data.data)
  for (let i = 0; i < N * N; i++) {
    expect(Math.abs(resultArr[i] - aData[i])).toBeLessThan(1e-3)
  }
})

test('non-square matmul: [32x64] @ [64x16]', () => {
  const M = 32, K = 64, N = 16
  const aData = Array.from({ length: M * K }, () => Math.random() - 0.5)
  const bData = Array.from({ length: K * N }, () => Math.random() - 0.5)
  const va = smith.variable(smith.tensor(aData, [M, K]))
  const vb = smith.variable(smith.tensor(bData, [K, N]))
  const vc = smith.matmul(va, vb)
  expect(vc.data.shape).toEqual([M, N])

  // Verify one random row/col dot product on CPU
  const result = Array.from(vc.data.data)
  const r = 7, c = 3
  let expected = 0
  for (let k = 0; k < K; k++) expected += aData[r * K + k] * bData[k * N + c]
  expect(Math.abs(result[r * N + c] - expected)).toBeLessThan(1e-3)
})

test('matmul associativity: (A@B)@C ≈ A@(B@C)', () => {
  const N = 32
  const rand = () => Array.from({ length: N * N }, () => Math.random() - 0.5)
  const vA = smith.variable(smith.tensor(rand(), [N, N]))
  const vB = smith.variable(smith.tensor(rand(), [N, N]))
  const vC = smith.variable(smith.tensor(rand(), [N, N]))

  const left = smith.matmul(smith.matmul(vA, vB), vC)   // (A@B)@C
  const right = smith.matmul(vA, smith.matmul(vB, vC))  // A@(B@C)

  const l = Array.from(left.data.data)
  const r = Array.from(right.data.data)
  for (let i = 0; i < l.length; i++) {
    expect(Math.abs(l[i] - r[i])).toBeLessThan(1e-2) // f32 accumulation diverges slightly
  }
})

test('matmul backward finite-diff (16x16)', () => {
  const N = 16
  const eps = 1e-4
  const aData = Array.from({ length: N * N }, () => Math.random() - 0.5)
  const bData = Array.from({ length: N * N }, () => Math.random() - 0.5)
  const a = smith.variable(smith.tensor(aData, [N, N]), { requiresGrad: true })
  const b = smith.variable(smith.tensor(bData, [N, N]))

  // Analytical gradient
  const loss = smith.sum(smith.matmul(a, b))
  smith.backward(loss)
  const analyticGrad = Array.from(a.grad.data)

  // Spot-check 10 random positions via finite differences
  const xData = a.data.data
  for (let trial = 0; trial < 10; trial++) {
    const i = Math.floor(Math.random() * N * N)
    const orig = xData[i]
    xData[i] = orig + eps
    const lPlus = smith.noGrad(() => smith.toArray(smith.sum(smith.matmul(a, b)).data))
    xData[i] = orig - eps
    const lMinus = smith.noGrad(() => smith.toArray(smith.sum(smith.matmul(a, b)).data))
    xData[i] = orig
    const numGrad = (lPlus - lMinus) / (2 * eps)
    expect(Math.abs(analyticGrad[i] - numGrad)).toBeLessThan(0.01)
  }
})
