// smith/src/pool.js
// GPU buffer memory pool. Recycles Metal buffers to avoid allocation overhead.
// Buffers are bucketed by size (rounded up to next power of 2) and storage mode.
// Pattern inspired by TinyFormer's optimized.js pool, adapted for GPU buffers.

import * as device from './device.js'

// Bucket sizes: round up to next power of 2 for efficient binning
function bucketSize(bytes) {
  if (bytes <= 256) return 256 // minimum 256 bytes
  let s = 1
  while (s < bytes) s <<= 1
  return s
}

// Separate pools for shared and private buffers
const sharedPool = new Map()  // bucketSize → [{ buffer, bytes }]
const privatePool = new Map()

let hits = 0
let misses = 0
const MAX_PER_BIN = 16 // cap per-bin to prevent memory bloat

function poolAlloc(bytes, mode = device.SHARED) {
  const bucket = bucketSize(bytes)
  const pool = mode === device.PRIVATE ? privatePool : sharedPool
  const bin = pool.get(bucket)

  if (bin && bin.length > 0) {
    hits++
    return bin.pop().buffer
  }

  misses++
  return device.alloc(bucket, mode)
}

function poolFree(buffer, bytes, mode = device.SHARED) {
  if (!buffer) return
  const bucket = bucketSize(bytes)
  const pool = mode === device.PRIVATE ? privatePool : sharedPool

  if (!pool.has(bucket)) pool.set(bucket, [])
  const bin = pool.get(bucket)

  if (bin.length < MAX_PER_BIN) {
    bin.push({ buffer, bytes: bucket })
  } else {
    // Pool is full for this size — actually release
    device.releaseBuffer(buffer)
  }
}

function poolStats() {
  let sharedCount = 0, privateCount = 0
  let sharedBytes = 0, privateBytes = 0
  for (const [size, bin] of sharedPool) {
    sharedCount += bin.length
    sharedBytes += size * bin.length
  }
  for (const [size, bin] of privatePool) {
    privateCount += bin.length
    privateBytes += size * bin.length
  }
  return {
    hits,
    misses,
    hitRate: hits / (hits + misses || 1),
    shared: { count: sharedCount, bytes: sharedBytes },
    private: { count: privateCount, bytes: privateBytes },
  }
}

function poolDrain() {
  for (const bin of sharedPool.values()) {
    for (const entry of bin) device.releaseBuffer(entry.buffer)
  }
  for (const bin of privatePool.values()) {
    for (const entry of bin) device.releaseBuffer(entry.buffer)
  }
  sharedPool.clear()
  privatePool.clear()
  hits = 0
  misses = 0
}

export {
  poolAlloc,
  poolFree,
  poolStats,
  poolDrain,
}
