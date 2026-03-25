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
