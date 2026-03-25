// smith/tests/pool.test.js
import { test, expect } from 'bun:test'
import smith from '../src/index.js'
import { poolStats, poolDrain } from '../src/pool.js'

test('pool stats start at zero hits', () => {
  poolDrain()
  const stats = poolStats()
  expect(stats.hits).toBe(0)
  expect(stats.misses).toBe(0)
})

test('tensor creation uses the pool', () => {
  poolDrain()
  const t1 = smith.zeros([64])
  const t2 = smith.zeros([64])
  const stats = poolStats()
  // Both are fresh allocations (misses)
  expect(stats.misses).toBeGreaterThanOrEqual(2)
})

test('pool recycles released buffers', async () => {
  poolDrain()
  const { release } = await import('../src/tensor.js')

  const t1 = smith.zeros([64])
  release(t1) // return buffer to pool

  const before = poolStats()
  const t2 = smith.zeros([64]) // should hit the pool
  const after = poolStats()

  expect(after.hits).toBeGreaterThan(before.hits)
})

test('pool drain clears everything', () => {
  smith.zeros([128])
  smith.zeros([256])
  poolDrain()
  const stats = poolStats()
  expect(stats.shared.count).toBe(0)
  expect(stats.private.count).toBe(0)
})
