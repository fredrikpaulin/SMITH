// smith/tests/autograd.test.js
import { test, expect } from 'bun:test'
import smith from '../src/index.js'

function expectClose(actual, expected, tol = 1e-4) {
  if (Array.isArray(expected)) {
    for (let i = 0; i < expected.length; i++) expectClose(actual[i], expected[i], tol)
  } else {
    expect(Math.abs(actual - expected)).toBeLessThan(tol)
  }
}

test('add backward', () => {
  const a = smith.variable(smith.tensor([1, 2, 3], [3]), { requiresGrad: true })
  const b = smith.variable(smith.tensor([4, 5, 6], [3]), { requiresGrad: true })
  const c = smith.add(a, b)
  const loss = smith.sum(c)
  smith.backward(loss)
  expectClose(smith.toArray(a.grad), [1, 1, 1])
  expectClose(smith.toArray(b.grad), [1, 1, 1])
})

test('mul backward', () => {
  const a = smith.variable(smith.tensor([2, 3], [2]), { requiresGrad: true })
  const b = smith.variable(smith.tensor([4, 5], [2]), { requiresGrad: true })
  const c = smith.mul(a, b)
  const loss = smith.sum(c)
  smith.backward(loss)
  // dL/da = b, dL/db = a
  expectClose(smith.toArray(a.grad), [4, 5])
  expectClose(smith.toArray(b.grad), [2, 3])
})

test('scale backward', () => {
  const a = smith.variable(smith.tensor([1, 2, 3], [3]), { requiresGrad: true })
  const b = smith.scale(a, 3.0)
  const loss = smith.sum(b)
  smith.backward(loss)
  expectClose(smith.toArray(a.grad), [3, 3, 3])
})

test('relu backward', () => {
  const a = smith.variable(smith.tensor([-1, 0, 1, 2], [4]), { requiresGrad: true })
  const b = smith.relu(a)
  const loss = smith.sum(b)
  smith.backward(loss)
  // relu'(x) = x > 0 ? 1 : 0
  expectClose(smith.toArray(a.grad), [0, 0, 1, 1])
})

test('gelu forward', () => {
  const a = smith.variable(smith.tensor([0, 1, -1], [3]))
  const b = smith.gelu(a)
  const result = smith.toArray(b.data)
  // GELU(0) ≈ 0, GELU(1) ≈ 0.8413, GELU(-1) ≈ -0.1587
  expectClose(result[0], 0, 0.01)
  expectClose(result[1], 0.8413, 0.01)
  expectClose(result[2], -0.1587, 0.01)
})

test('neg backward', () => {
  const a = smith.variable(smith.tensor([1, 2, 3], [3]), { requiresGrad: true })
  const b = smith.neg(a)
  const loss = smith.sum(b)
  smith.backward(loss)
  expectClose(smith.toArray(a.grad), [-1, -1, -1])
})

test('noGrad skips graph construction', () => {
  const a = smith.variable(smith.tensor([1, 2], [2]), { requiresGrad: true })
  let c
  smith.noGrad(() => {
    c = smith.add(a, smith.variable(smith.tensor([3, 4], [2])))
  })
  expect(c._backward).toBeNull()
})

test('chain rule through multiple ops', () => {
  // f(x) = sum(relu(x * 2 + 1))
  const x = smith.variable(smith.tensor([1, -1, 0.5], [3]), { requiresGrad: true })
  const two = smith.variable(smith.tensor([2, 2, 2], [3]))
  const one = smith.variable(smith.tensor([1, 1, 1], [3]))

  const scaled = smith.mul(x, two)      // [2, -2, 1]
  const shifted = smith.add(scaled, one) // [3, -1, 2]
  const activated = smith.relu(shifted)  // [3, 0, 2]
  const loss = smith.sum(activated)      // 5

  smith.backward(loss)

  // d(loss)/d(x) = 2 * relu'(2x + 1)
  // x=1:  2*1+1=3 > 0, grad = 2
  // x=-1: 2*(-1)+1=-1 <= 0, grad = 0
  // x=0.5: 2*0.5+1=2 > 0, grad = 2
  expectClose(smith.toArray(x.grad), [2, 0, 2])
})

test('gradient accumulation (shared variable)', () => {
  // y = x + x (x used twice)
  const x = smith.variable(smith.tensor([1, 2], [2]), { requiresGrad: true })
  const y = smith.add(x, x)
  const loss = smith.sum(y)
  smith.backward(loss)
  // Each add contributes grad=1 for both a and b, and a=b=x, so grad=2
  expectClose(smith.toArray(x.grad), [2, 2])
})

test('zeroGrad clears gradients', () => {
  const x = smith.variable(smith.tensor([1, 2], [2]), { requiresGrad: true })
  const y = smith.sum(smith.scale(x, 3))
  smith.backward(y)
  expect(x.grad).not.toBeNull()
  smith.zeroGrad([x])
  expect(x.grad).toBeNull()
})

// --- Finite-difference gradient verification (GPU) ---

// Numerically check dL/dx by perturbing each element of x and measuring loss change.
// This verifies the GPU backward pass against the GPU forward pass — no CPU mock needed.
function numGradCheck(makeLoss, x, tol = 1e-3) {
  const eps = 1e-4
  // Analytical gradient
  smith.zeroGrad([x])
  const loss = makeLoss(x)
  smith.backward(loss)
  const analyticGrad = Array.from(x.grad.data)

  // Numerical gradient: for each element, perturb ±eps and measure Δloss
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

test('finite-diff: relu backward (64 elements)', () => {
  // Avoid values near 0 where relu's discontinuity breaks finite differences
  const data = Array.from({ length: 64 }, () => {
    let v; do { v = Math.random() * 4 - 2 } while (Math.abs(v) < 0.1)
    return v
  })
  const x = smith.variable(smith.tensor(data, [64]), { requiresGrad: true })
  numGradCheck(v => smith.sum(smith.relu(v)), x, 0.05)
})

test('finite-diff: gelu backward (64 elements)', () => {
  const data = Array.from({ length: 64 }, () => Math.random() * 4 - 2)
  const x = smith.variable(smith.tensor(data, [64]), { requiresGrad: true })
  numGradCheck(v => smith.sum(smith.gelu(v)), x, 0.05)
})

test('finite-diff: mul chain backward (32 elements)', () => {
  const data = Array.from({ length: 32 }, () => Math.random() * 2 + 0.5)
  const x = smith.variable(smith.tensor(data, [32]), { requiresGrad: true })
  const c = smith.variable(smith.tensor(data.map(() => Math.random() * 2), [32]))
  numGradCheck(v => smith.sum(smith.mul(v, c)), x, 0.05)
})

test('finite-diff: matmul backward (16x16)', () => {
  const N = 16
  const aData = Array.from({ length: N * N }, () => Math.random() - 0.5)
  const bData = Array.from({ length: N * N }, () => Math.random() - 0.5)
  const a = smith.variable(smith.tensor(aData, [N, N]), { requiresGrad: true })
  const b = smith.variable(smith.tensor(bData, [N, N]))
  numGradCheck(v => smith.sum(smith.matmul(v, b)), a, 0.01)
})

test('finite-diff: scale + add chain (64 elements)', () => {
  const data = Array.from({ length: 64 }, () => Math.random() * 3 - 1)
  const x = smith.variable(smith.tensor(data, [64]), { requiresGrad: true })
  const bias = smith.variable(smith.tensor(data.map(() => Math.random()), [64]))
  // relu discontinuity + f32 accumulation over 64 elements → needs generous tolerance
  numGradCheck(v => smith.sum(smith.relu(smith.add(smith.scale(v, 2.5), bias))), x, 0.1)
})
