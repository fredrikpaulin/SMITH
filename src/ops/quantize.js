// smith/src/ops/quantize.js
// 4-bit quantization: pack f32 weights to q4 format, dispatch q4 matmul.
// Group size: 32 elements. Each group = 16 bytes nibbles + 4 bytes scale + 4 bytes zero.

import * as T from '../tensor.js'
import * as device from '../device.js'
import { run } from '../dispatch.js'

const GROUP_SIZE = 32
const GROUP_BYTES = 24 // 16 nibbles + 4 scale + 4 zero

// Quantize a 2D weight tensor [K, N] to q4 format.
// Returns a raw GPU buffer and metadata.
function quantizeQ4(weight) {
  const [K, N] = weight.shape
  const groups = Math.ceil(K / GROUP_SIZE)
  const totalBytes = groups * N * GROUP_BYTES
  const packed = new Uint8Array(totalBytes)

  for (let g = 0; g < groups; g++) {
    const kStart = g * GROUP_SIZE
    const kEnd = Math.min(kStart + GROUP_SIZE, K)

    for (let col = 0; col < N; col++) {
      // Find min/max in this group for this column
      let minVal = Infinity, maxVal = -Infinity
      for (let k = kStart; k < kEnd; k++) {
        const v = weight.data[k * N + col]
        if (v < minVal) minVal = v
        if (v > maxVal) maxVal = v
      }

      // Compute scale and zero-point for 4-bit [0, 15] range
      const range = maxVal - minVal
      const scale = range > 0 ? range / 15 : 1
      const zero = range > 0 ? -minVal / scale : 0

      const groupIdx = g * N + col
      const offset = groupIdx * GROUP_BYTES

      // Pack nibbles
      for (let k = 0; k < GROUP_SIZE; k++) {
        const kIdx = kStart + k
        const v = kIdx < K ? weight.data[kIdx * N + col] : 0
        const q = Math.round(Math.min(15, Math.max(0, v / scale + zero)))
        const byteIdx = offset + (k >> 1)
        if (k & 1) {
          packed[byteIdx] |= (q << 4)
        } else {
          packed[byteIdx] = q
        }
      }

      // Write scale and zero as f32
      const scaleView = new Float32Array(packed.buffer, offset + 16, 1)
      const zeroView = new Float32Array(packed.buffer, offset + 20, 1)
      scaleView[0] = scale
      zeroView[0] = zero
    }
  }

  // Upload to GPU buffer
  const buffer = device.alloc(totalBytes, device.SHARED)
  const view = device.viewBuffer(buffer, totalBytes, Uint8Array)
  view.set(packed)

  return { buffer, K, N, groups, totalBytes }
}

// Q4 matmul: C[M,N] = A[M,K](f32) @ B[K,N](q4)
function matmulQ4(a, bQuant) {
  const M = a.shape[0]
  const { N, K, groups, buffer: bBuffer } = bQuant

  if (a.shape[1] !== K) {
    throw new Error(`matmulQ4: inner dims don't match: ${a.shape[1]} vs ${K}`)
  }

  const out = T.create([M, N], a.dtype)
  const params = new Uint32Array([M, N, K, groups])

  run('matmul_q4', [
    { buffer: a.buffer, index: 0 },
    { buffer: bBuffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], { x: N, y: M }, { x: Math.min(N, 16), y: Math.min(M, 16) },
  { data: params, index: 3 })

  return out
}

// --- Q8 matmul ---
// Q8_0 block layout: 2 bytes fp16 scale + 32 bytes int8 = 34 bytes per block
const Q8_BLOCK_SIZE = 32
const Q8_BLOCK_BYTES = 34

// Q8 matmul: C[M,N] = A[M,K](f32) @ B[K,N](q8)
// bQuant: { buffer, K, N, groups }
function matmulQ8(a, bQuant) {
  const M = a.shape[0]
  const { N, K, groups, buffer: bBuffer } = bQuant

  if (a.shape[1] !== K) {
    throw new Error(`matmulQ8: inner dims don't match: ${a.shape[1]} vs ${K}`)
  }

  const out = T.create([M, N], a.dtype)
  const params = new Uint32Array([M, N, K, groups])

  run('matmul_q8', [
    { buffer: a.buffer, index: 0 },
    { buffer: bBuffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], { x: N, y: M }, { x: Math.min(N, 16), y: Math.min(M, 16) },
  { data: params, index: 3 })

  return out
}

export { quantizeQ4, matmulQ4, matmulQ8, GROUP_SIZE, GROUP_BYTES, Q8_BLOCK_SIZE, Q8_BLOCK_BYTES }
