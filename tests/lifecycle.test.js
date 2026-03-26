// tests/lifecycle.test.js
// Phase 22: Tensor Lifecycle Management
// Tests dispose, retain, scoped cleanup (using), withNoAlloc, and poolStats integration.
// Requires Metal GPU — uses real tensor allocation.

import { test, expect, describe, beforeEach } from 'bun:test'
import * as T from '../src/tensor.js'
import { poolStats, poolDrain } from '../src/pool.js'
import {
  dispose, retain, isDisposed,
  using, usingAsync, withNoAlloc,
  activeScopeDepth,
} from '../src/lifecycle.js'

beforeEach(() => {
  poolDrain()
})

describe('dispose', () => {
  test('makes tensor buffer and data null', () => {
    const t = T.zeros([4])
    expect(t.buffer).not.toBeNull()
    expect(t.data).not.toBeNull()
    dispose(t)
    expect(t.buffer).toBeNull()
    expect(t.data).toBeNull()
  })

  test('marks tensor as disposed', () => {
    const t = T.zeros([4])
    expect(isDisposed(t)).toBe(false)
    dispose(t)
    expect(isDisposed(t)).toBe(true)
  })

  test('double dispose is safe', () => {
    const t = T.zeros([4])
    dispose(t)
    dispose(t) // should not throw
    expect(isDisposed(t)).toBe(true)
  })

  test('dispose(null) is safe', () => {
    dispose(null)
    dispose(undefined)
  })

  test('returns buffer to pool', () => {
    poolDrain()
    const t = T.zeros([16])
    dispose(t)
    const stats = poolStats()
    expect(stats.shared.count).toBeGreaterThan(0)
  })
})

describe('retain', () => {
  test('retained tensor survives one dispose call', () => {
    const t = T.zeros([4])
    retain(t)
    dispose(t) // decrements ref count, but still > 0
    expect(isDisposed(t)).toBe(false)
    expect(t.buffer).not.toBeNull()
    expect(t.data).not.toBeNull()
    // Clean up
    dispose(t)
  })

  test('retained tensor freed after matching dispose calls', () => {
    const t = T.zeros([4])
    retain(t) // refCount: 2
    dispose(t) // refCount: 1, still alive
    expect(t.buffer).not.toBeNull()
    dispose(t) // refCount: 0, freed
    expect(t.buffer).toBeNull()
    expect(isDisposed(t)).toBe(true)
  })

  test('multiple retains require multiple disposes', () => {
    const t = T.zeros([4])
    retain(t) // refCount: 2
    retain(t) // refCount: 3
    dispose(t) // 2
    dispose(t) // 1
    expect(t.buffer).not.toBeNull()
    dispose(t) // 0, freed
    expect(t.buffer).toBeNull()
  })

  test('returns the tensor for chaining', () => {
    const t = T.zeros([4])
    const r = retain(t)
    expect(r).toBe(t)
    dispose(t)
    dispose(t)
  })
})

describe('using', () => {
  test('cleans up tensors allocated inside scope', () => {
    const tensors = []
    using(() => {
      tensors.push(T.zeros([8]))
      tensors.push(T.zeros([16]))
      tensors.push(T.zeros([32]))
    })
    for (const t of tensors) {
      expect(isDisposed(t)).toBe(true)
      expect(t.buffer).toBeNull()
    }
  })

  test('returns the function result', () => {
    const result = using(() => 42)
    expect(result).toBe(42)
  })

  test('returned tensor from scope is disposed', () => {
    const result = using(() => T.zeros([4]))
    expect(isDisposed(result)).toBe(true)
  })

  test('retained tensor survives scope exit', () => {
    const result = using(() => {
      const t = T.zeros([4])
      retain(t)
      return t
    })
    expect(isDisposed(result)).toBe(false)
    expect(result.buffer).not.toBeNull()
    // Clean up
    dispose(result)
    expect(isDisposed(result)).toBe(true)
  })

  test('nested scopes clean up independently', () => {
    const outer = []
    const inner = []
    using(() => {
      outer.push(T.zeros([4]))
      using(() => {
        inner.push(T.zeros([8]))
      })
      expect(isDisposed(inner[0])).toBe(true)
      expect(isDisposed(outer[0])).toBe(false)
    })
    expect(isDisposed(outer[0])).toBe(true)
  })

  test('cleans up even on exception', () => {
    const tensors = []
    try {
      using(() => {
        tensors.push(T.zeros([4]))
        tensors.push(T.zeros([8]))
        throw new Error('test error')
      })
    } catch (e) {
      expect(e.message).toBe('test error')
    }
    for (const t of tensors) {
      expect(isDisposed(t)).toBe(true)
    }
  })

  test('tensors allocated outside scope are not affected', () => {
    const outside = T.zeros([4])
    using(() => {
      T.zeros([8])
    })
    expect(isDisposed(outside)).toBe(false)
    expect(outside.buffer).not.toBeNull()
    dispose(outside)
  })

  test('scope depth tracking', () => {
    expect(activeScopeDepth()).toBe(0)
    using(() => {
      expect(activeScopeDepth()).toBe(1)
      using(() => {
        expect(activeScopeDepth()).toBe(2)
      })
      expect(activeScopeDepth()).toBe(1)
    })
    expect(activeScopeDepth()).toBe(0)
  })
})

describe('usingAsync', () => {
  test('cleans up tensors from async scope', async () => {
    const tensors = []
    await usingAsync(async () => {
      tensors.push(T.zeros([4]))
      await new Promise(r => setTimeout(r, 1))
      tensors.push(T.zeros([8]))
    })
    for (const t of tensors) {
      expect(isDisposed(t)).toBe(true)
    }
  })

  test('returns async result', async () => {
    const result = await usingAsync(async () => {
      await new Promise(r => setTimeout(r, 1))
      return 99
    })
    expect(result).toBe(99)
  })

  test('cleans up on async exception', async () => {
    const tensors = []
    try {
      await usingAsync(async () => {
        tensors.push(T.zeros([4]))
        await new Promise(r => setTimeout(r, 1))
        throw new Error('async boom')
      })
    } catch (e) {
      expect(e.message).toBe('async boom')
    }
    expect(tensors[0]._disposed).toBe(true)
  })
})

describe('withNoAlloc', () => {
  test('throws on tensor allocation', () => {
    expect(() => {
      withNoAlloc(() => {
        T.zeros([4])
      })
    }).toThrow('Unexpected tensor allocation inside withNoAlloc()')
  })

  test('allows code that does not allocate', () => {
    const t = T.zeros([4])
    const result = withNoAlloc(() => t.data[0])
    expect(result).toBe(0)
    dispose(t)
  })

  test('restores state after exception', () => {
    try {
      withNoAlloc(() => { T.zeros([4]) })
    } catch (e) { /* expected */ }
    // Should be able to allocate again
    const t = T.zeros([4])
    expect(t.buffer).not.toBeNull()
    dispose(t)
  })
})

describe('poolStats integration', () => {
  test('totalAllocated tracks cumulative bytes', () => {
    poolDrain()
    const before = poolStats()
    expect(before.totalAllocated).toBe(0)

    const t1 = T.zeros([64]) // 64 * 4 = 256 bytes → bucket 256
    const stats1 = poolStats()
    expect(stats1.totalAllocated).toBe(256)

    // Dispose and reallocate same size — pool hit, no new allocation
    dispose(t1)
    const t2 = T.zeros([64])
    const stats2 = poolStats()
    expect(stats2.totalAllocated).toBe(256) // no increase — recycled
    expect(stats2.hits).toBeGreaterThanOrEqual(1)

    dispose(t2)
  })

  test('hitRate increases with pool reuse', () => {
    poolDrain()
    for (let i = 0; i < 10; i++) {
      const t = T.zeros([64])
      dispose(t)
    }
    const stats = poolStats()
    // First allocation is a miss, remaining 9 are hits
    expect(stats.hits).toBe(9)
    expect(stats.misses).toBe(1)
    expect(stats.hitRate).toBeCloseTo(0.9, 1)
  })
})

describe('lifecycle with creation functions', () => {
  test('all tensor factories get lifecycle fields', () => {
    const fns = [
      () => T.zeros([4]),
      () => T.ones([4]),
      () => T.rand([4]),
      () => T.randn([4]),
      () => T.full([4], 5),
      () => T.tensor([1, 2, 3, 4], [4]),
      () => T.scalar(1),
    ]
    for (const fn of fns) {
      const t = fn()
      expect(t._refCount).toBe(1)
      expect(t._disposed).toBe(false)
      dispose(t)
      expect(t._disposed).toBe(true)
    }
  })

  test('T.release still works alongside lifecycle', () => {
    const t = T.zeros([4])
    T.release(t)
    expect(t.buffer).toBeNull()
    expect(t.data).toBeNull()
  })

  test('using() cleans up all tensor factory types', () => {
    const tensors = []
    using(() => {
      tensors.push(T.zeros([4]))
      tensors.push(T.ones([4]))
      tensors.push(T.rand([4]))
      tensors.push(T.randn([4]))
      tensors.push(T.full([4], 7))
    })
    for (const t of tensors) {
      expect(isDisposed(t)).toBe(true)
    }
  })

  test('many tensors in scope, only retained ones survive', () => {
    const alive = []
    const dead = []
    using(() => {
      for (let i = 0; i < 6; i++) {
        const t = T.zeros([4])
        if (i % 2 === 0) {
          retain(t)
          alive.push(t)
        } else {
          dead.push(t)
        }
      }
    })
    for (const t of alive) {
      expect(isDisposed(t)).toBe(false)
      dispose(t) // clean up
    }
    for (const t of dead) {
      expect(isDisposed(t)).toBe(true)
    }
  })
})
