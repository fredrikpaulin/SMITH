// smith/tests/tensor.test.js
import { test, expect } from 'bun:test'
import smith from '../src/index.js'

test('create tensor from array', () => {
  const t = smith.tensor([1, 2, 3, 4, 5, 6], [2, 3])
  expect(t.shape).toEqual([2, 3])
  expect(t.size).toBe(6)
  expect(t.dtype).toBe('f32')
  expect(smith.toArray(t)).toEqual([[1, 2, 3], [4, 5, 6]])
})

test('zeros', () => {
  const t = smith.zeros([3, 3])
  expect(t.shape).toEqual([3, 3])
  const arr = smith.toArray(t)
  for (const row of arr) for (const v of row) expect(v).toBe(0)
})

test('ones', () => {
  const t = smith.ones([2, 2])
  expect(smith.toArray(t)).toEqual([[1, 1], [1, 1]])
})

test('full', () => {
  const t = smith.full([2], 3.14)
  const arr = smith.toArray(t)
  expect(arr[0]).toBeCloseTo(3.14, 4)
  expect(arr[1]).toBeCloseTo(3.14, 4)
})

test('rand values are in [0, 1)', () => {
  const t = smith.rand([100])
  const arr = smith.toArray(t)
  for (const v of arr) {
    expect(v).toBeGreaterThanOrEqual(0)
    expect(v).toBeLessThan(1)
  }
})

test('randn produces reasonable distribution', () => {
  const t = smith.randn([1000])
  const arr = smith.toArray(t)
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length
  expect(Math.abs(mean)).toBeLessThan(0.2) // should be near 0
})

test('scalar', () => {
  const t = smith.scalar(42)
  expect(t.shape).toEqual([])
  expect(t.size).toBe(1)
  expect(smith.toArray(t)).toBe(42)
})

test('shape utilities', async () => {
  const { computeStrides, shapeSize, broadcastShapes } = await import('../src/tensor.js')
  expect(computeStrides([2, 3, 4])).toEqual([12, 4, 1])
  expect(shapeSize([2, 3, 4])).toBe(24)
  expect(broadcastShapes([1, 3], [2, 1])).toEqual([2, 3])
  expect(broadcastShapes([5], [2, 5])).toEqual([2, 5])
})
