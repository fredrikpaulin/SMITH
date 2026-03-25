// smith/tests/dtype.test.js
import { test, expect } from 'bun:test'
import { toFloat16, fromFloat16, float32ToFloat16, float16ToFloat32 } from '../src/dtype.js'

test('f16 roundtrip for typical values', () => {
  const values = [0, 1, -1, 0.5, -0.5, 3.14, 100, -100, 0.001]
  for (const v of values) {
    const h = toFloat16(v)
    const back = fromFloat16(h)
    // f16 has limited precision — accept ~0.1% error for normal values
    expect(Math.abs(back - v)).toBeLessThan(Math.abs(v) * 0.01 + 0.001)
  }
})

test('f16 handles zero', () => {
  expect(fromFloat16(toFloat16(0))).toBe(0)
  expect(fromFloat16(toFloat16(-0))).toBe(-0)
})

test('f16 handles infinity', () => {
  expect(fromFloat16(toFloat16(Infinity))).toBe(Infinity)
  expect(fromFloat16(toFloat16(-Infinity))).toBe(-Infinity)
})

test('f16 handles NaN', () => {
  expect(isNaN(fromFloat16(toFloat16(NaN)))).toBe(true)
})

test('f16 overflow to infinity', () => {
  // Max f16 is ~65504. Values above overflow.
  expect(fromFloat16(toFloat16(100000))).toBe(Infinity)
})

test('bulk f16 conversion', () => {
  const f32 = new Float32Array([1, 2, 3, -1, 0])
  const f16 = float32ToFloat16(f32)
  expect(f16.length).toBe(5)
  const back = float16ToFloat32(f16)
  for (let i = 0; i < f32.length; i++) {
    expect(Math.abs(back[i] - f32[i])).toBeLessThan(0.01)
  }
})
