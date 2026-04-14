// smith/src/ops/matmul.js
// Matrix multiplication via Metal compute shaders.
// Chooses between simple and tiled kernels based on matrix size.
// Supports batched matmul for attention heads etc.

import * as T from '../tensor.js'
import { run, matmulParams, batchMatmulParams, k } from '../dispatch.js'

// Threshold: use tiled kernel for matrices larger than this
const TILE_THRESHOLD = 64

// 2D matmul: C[M,N] = A[M,K] @ B[K,N]
function matmul2d(a_, b_) {
  // Shaders assume contiguous memory — resolve strided views
  const a = T.contiguous(a_)
  const b = T.contiguous(b_)
  const M = a.shape[a.shape.length - 2]
  const K = a.shape[a.shape.length - 1]
  const N = b.shape[b.shape.length - 1]

  if (K !== b.shape[b.shape.length - 2]) {
    throw new Error(`matmul: inner dims don't match: ${K} vs ${b.shape[b.shape.length - 2]}`)
  }

  const out = T.create([M, N], a.dtype)
  const params = matmulParams(M, N, K)

  if (M >= TILE_THRESHOLD && N >= TILE_THRESHOLD && K >= TILE_THRESHOLD) {
    // Tiled GEMM with threadgroup shared memory
    // Grid: one threadgroup per 32x32 tile of the output
    const groupsX = Math.ceil(N / 32)
    const groupsY = Math.ceil(M / 32)
    run(a.dtype === 'f16' ? 'matmul_f16' : 'matmul_f32', [
      { buffer: a.buffer, index: 0 },
      { buffer: b.buffer, index: 1 },
      { buffer: out.buffer, index: 2 },
    ],
    // Grid in threadgroups (not threads) — we use dispatch with threadgroups
    // Actually Metal dispatchThreads wants total threads, so:
    { x: groupsX * 64, y: groupsY }, // 64 threads per group (8x8)
    { x: 64, y: 1 },
    { data: params, index: 3 })
  } else {
    // Simple kernel: one thread per output element
    run(k('matmul_simple', a.dtype), [
      { buffer: a.buffer, index: 0 },
      { buffer: b.buffer, index: 1 },
      { buffer: out.buffer, index: 2 },
    ], { x: N, y: M }, { x: Math.min(N, 16), y: Math.min(M, 16) },
    { data: params, index: 3 })
  }

  // Release contiguous copies if they were allocated (different from input)
  if (a !== a_) T.release(a)
  if (b !== b_) T.release(b)

  return out
}

// Batched matmul: C[B,M,N] = A[B,M,K] @ B[B,K,N]
function matmulBatched(a_, b_, batchSize) {
  const a = T.contiguous(a_)
  const b = T.contiguous(b_)
  const M = a.shape[a.shape.length - 2]
  const K = a.shape[a.shape.length - 1]
  const N = b.shape[b.shape.length - 1]

  const outShape = [...a.shape.slice(0, -2), M, N]
  const out = T.create(outShape, a.dtype)
  const params = batchMatmulParams(M, N, K, batchSize)

  run(k('matmul_batched', a.dtype), [
    { buffer: a.buffer, index: 0 },
    { buffer: b.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], { x: N, y: M, z: batchSize },
  { x: Math.min(N, 16), y: Math.min(M, 16), z: 1 },
  { data: params, index: 3 })

  // Release contiguous copies if they were allocated (different from input)
  if (a !== a_) T.release(a)
  if (b !== b_) T.release(b)

  return out
}

// Public matmul: handles 2D and batched cases
function matmul(a, b) {
  if (a.shape.length < 2 || b.shape.length < 2) {
    throw new Error(`matmul requires at least 2D tensors, got ${a.shape.length}D and ${b.shape.length}D`)
  }

  // Pure 2D case
  if (a.shape.length === 2 && b.shape.length === 2) {
    return matmul2d(a, b)
  }

  // Batched case: broadcast batch dimensions
  const aBatch = a.shape.slice(0, -2)
  const bBatch = b.shape.slice(0, -2)
  const batchShape = T.broadcastShapes(aBatch, bBatch)
  const batchSize = T.shapeSize(batchShape)

  // For now: require batch dims to match exactly (no broadcast in batch dim)
  // TODO: handle broadcast batch dims by expanding before dispatch
  return matmulBatched(a, b, batchSize)
}

export { matmul, matmul2d, matmulBatched }
