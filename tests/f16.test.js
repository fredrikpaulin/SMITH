// smith/tests/f16.test.js
// Phase 8: Mixed precision f16 tests
import { test, expect } from 'bun:test'
import smith from '../src/index.js'

function expectClose(actual, expected, tol = 1e-2) {
  if (Array.isArray(expected)) {
    for (let i = 0; i < expected.length; i++) expectClose(actual[i], expected[i], tol)
  } else {
    expect(Math.abs(actual - expected)).toBeLessThan(tol)
  }
}

// --- Cast ---

test('cast f32 → f16 → f32 roundtrip', () => {
  const a = smith.tensor([1.5, -2.25, 0, 3.14], [4])
  const h = smith.cast(a, 'f16')
  expect(h.dtype).toBe('f16')
  expect(h.shape).toEqual([4])
  const back = smith.cast(h, 'f32')
  expect(back.dtype).toBe('f32')
  // f16 has limited precision, but these values are representable
  expectClose(smith.toArray(back), [1.5, -2.25, 0, 3.14], 0.01)
})

test('cast same dtype is identity', () => {
  const a = smith.tensor([1, 2, 3], [3])
  const b = smith.cast(a, 'f32')
  expect(b).toBe(a) // same object, no copy
})

// --- f16 tensor creation ---

test('create f16 tensor', () => {
  const t = smith.tensor([1.0, 2.0, 3.0, 4.0], [2, 2], 'f16')
  expect(t.dtype).toBe('f16')
  expect(t.shape).toEqual([2, 2])
  expectClose(smith.toArray(t), [[1, 2], [3, 4]], 0.01)
})

test('zeros f16', () => {
  const t = smith.zeros([4], 'f16')
  expect(t.dtype).toBe('f16')
  expectClose(smith.toArray(t), [0, 0, 0, 0])
})

test('ones f16', () => {
  const t = smith.ones([3], 'f16')
  expect(t.dtype).toBe('f16')
  expectClose(smith.toArray(t), [1, 1, 1])
})

// --- f16 elementwise ops ---

test('f16 add', () => {
  const a = smith.tensor([1, 2, 3, 4], [4], 'f16')
  const b = smith.tensor([5, 6, 7, 8], [4], 'f16')
  const va = smith.variable(a)
  const vb = smith.variable(b)
  const vc = smith.add(va, vb)
  expect(vc.data.dtype).toBe('f16')
  expectClose(smith.toArray(vc.data), [6, 8, 10, 12])
})

test('f16 sub', () => {
  const a = smith.tensor([5, 6, 7, 8], [4], 'f16')
  const b = smith.tensor([1, 2, 3, 4], [4], 'f16')
  const va = smith.variable(a)
  const vb = smith.variable(b)
  const vc = smith.sub(va, vb)
  expect(vc.data.dtype).toBe('f16')
  expectClose(smith.toArray(vc.data), [4, 4, 4, 4])
})

test('f16 mul', () => {
  const a = smith.tensor([1, 2, 3, 4], [4], 'f16')
  const b = smith.tensor([2, 3, 4, 5], [4], 'f16')
  const va = smith.variable(a)
  const vb = smith.variable(b)
  const vc = smith.mul(va, vb)
  expect(vc.data.dtype).toBe('f16')
  expectClose(smith.toArray(vc.data), [2, 6, 12, 20])
})

// --- f16 matmul ---

test('f16 2x2 matmul', () => {
  const a = smith.tensor([1, 2, 3, 4], [2, 2], 'f16')
  const b = smith.tensor([5, 6, 7, 8], [2, 2], 'f16')
  const va = smith.variable(a)
  const vb = smith.variable(b)
  const vc = smith.matmul(va, vb)
  expect(vc.data.dtype).toBe('f16')
  // [1*5+2*7, 1*6+2*8] = [19, 22]
  // [3*5+4*7, 3*6+4*8] = [43, 50]
  expectClose(smith.toArray(vc.data), [[19, 22], [43, 50]])
})

test('f16 matmul matches f32 within tolerance', () => {
  const N = 16
  const vals = Array.from({ length: N * N }, () => Math.random() * 2 - 1)
  const af32 = smith.tensor(vals, [N, N], 'f32')
  const bf32 = smith.tensor(vals, [N, N], 'f32')
  const af16 = smith.tensor(vals, [N, N], 'f16')
  const bf16 = smith.tensor(vals, [N, N], 'f16')

  const va32 = smith.variable(af32)
  const vb32 = smith.variable(bf32)
  const va16 = smith.variable(af16)
  const vb16 = smith.variable(bf16)

  const c32 = smith.toArray(smith.matmul(va32, vb32).data).flat()
  const c16 = smith.toArray(smith.matmul(va16, vb16).data).flat()

  let maxDiff = 0
  for (let i = 0; i < c32.length; i++) {
    const diff = Math.abs(c32[i] - c16[i])
    if (diff > maxDiff) maxDiff = diff
  }
  // f16 matmul should match f32 within ~1e-2 for small matrices
  expect(maxDiff).toBeLessThan(0.05)
})

// --- f16 activations ---

test('f16 relu', () => {
  const a = smith.tensor([-1, 0, 1, 2], [4], 'f16')
  const va = smith.variable(a)
  const vr = smith.relu(va)
  expect(vr.data.dtype).toBe('f16')
  expectClose(smith.toArray(vr.data), [0, 0, 1, 2])
})

test('f16 gelu', () => {
  const a = smith.tensor([0, 1, -1, 2], [4], 'f16')
  const va = smith.variable(a)
  const vg = smith.gelu(va)
  expect(vg.data.dtype).toBe('f16')
  const result = smith.toArray(vg.data)
  // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159, GELU(2) ≈ 1.955
  expectClose(result[0], 0, 0.05)
  expectClose(result[1], 0.841, 0.05)
  expectClose(result[2], -0.159, 0.05)
  expectClose(result[3], 1.955, 0.05)
})

// --- f16 softmax ---

test('f16 softmax', () => {
  const a = smith.tensor([1, 2, 3, 4], [1, 4], 'f16')
  const va = smith.variable(a)
  const vs = smith.softmax(va)
  expect(vs.data.dtype).toBe('f16')
  const result = smith.toArray(vs.data).flat()
  // Sum should be ~1.0
  const sum = result.reduce((a, b) => a + b, 0)
  expectClose(sum, 1.0, 0.02)
  // Values should be monotonically increasing
  for (let i = 1; i < result.length; i++) {
    expect(result[i]).toBeGreaterThan(result[i - 1])
  }
})

// --- f16 layernorm ---

test('f16 layernorm', () => {
  const a = smith.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 4], 'f16')
  const gamma = smith.tensor([1, 1, 1, 1], [4], 'f16')
  const beta = smith.tensor([0, 0, 0, 0], [4], 'f16')
  const va = smith.variable(a)
  const vg = smith.variable(gamma)
  const vb = smith.variable(beta)
  const vln = smith.layernorm(va, vg, vb)
  expect(vln.data.dtype).toBe('f16')
  const result = smith.toArray(vln.data)
  // Each row should have mean ≈ 0 and std ≈ 1
  for (const row of result) {
    const mean = row.reduce((a, b) => a + b, 0) / row.length
    expectClose(mean, 0, 0.1)
  }
})

// --- f16 reduce ---

test('f16 sum', () => {
  const a = smith.tensor([1, 2, 3, 4], [2, 2], 'f16')
  const va = smith.variable(a)
  const vs = smith.sum(va, 1)
  expect(vs.data.dtype).toBe('f16')
  expectClose(smith.toArray(vs.data), [3, 7])
})

// --- f16Mode toggle ---

test('f16Mode toggle', () => {
  expect(smith.f16Mode()).toBe(false)
  smith.f16Mode(true)
  expect(smith.f16Mode()).toBe(true)
  expect(smith.defaultDtype()).toBe('f16')
  smith.f16Mode(false)
  expect(smith.f16Mode()).toBe(false)
  expect(smith.defaultDtype()).toBe('f32')
})

// --- Loss scaler ---

test('loss scaler basic operation', () => {
  const scaler = smith.createLossScaler({ initScale: 1024, growthInterval: 2 })
  expect(scaler.getScale()).toBe(1024)

  // Successful step
  scaler.update(true)
  expect(scaler.getScale()).toBe(1024) // not enough good steps yet

  // Second successful step triggers growth
  scaler.update(true)
  expect(scaler.getScale()).toBe(2048) // doubled

  // NaN step halves
  scaler.update(false)
  expect(scaler.getScale()).toBe(1024) // halved back
})

test('loss scaler NaN detection', () => {
  const scaler = smith.createLossScaler({ initScale: 256 })
  const grads = [{ data: new Float32Array([1.0, 2.0, 3.0]) }]
  const ok = scaler.unscale(grads)
  expect(ok).toBe(true)
  // Values should be divided by scale
  expectClose(grads[0].data[0], 1 / 256, 1e-6)

  const nanGrads = [{ data: new Float32Array([1.0, NaN, 3.0]) }]
  const notOk = scaler.unscale(nanGrads)
  expect(notOk).toBe(false)
})

test('loss scaler min scale floor', () => {
  const scaler = smith.createLossScaler({ initScale: 2, minScale: 1 })
  scaler.update(false) // 2 → 1
  expect(scaler.getScale()).toBe(1)
  scaler.update(false) // stays at 1 (floor)
  expect(scaler.getScale()).toBe(1)
})
