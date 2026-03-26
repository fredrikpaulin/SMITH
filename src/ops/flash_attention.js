// smith/src/ops/flash_attention.js
// GPU flash attention dispatch.
// Forward: fused Q*K^T → softmax → V in tiles, O(n) memory.
// Backward: recomputes attention from saved log-sum-exp stats.
// Supports GQA (grouped query attention) and sliding window attention.

import * as T from '../tensor.js'
import { run, k } from '../dispatch.js'

// Tile sizes must match the shader constants
const Br = 32
const Bc = 32

// Build params struct matching FlashAttnParams in the shader:
// { uint N, uint d, uint numQHeads, float scale, uint causal, uint numKVHeads, uint windowSize }
function flashAttnParams(N, d, numQHeads, scale, causal, numKVHeads, windowSize) {
  const buf = new ArrayBuffer(28) // 5 uints + 1 float + 1 uint = 28 bytes
  const u = new Uint32Array(buf)
  const f = new Float32Array(buf)
  u[0] = N
  u[1] = d
  u[2] = numQHeads
  f[3] = scale         // float at byte offset 12
  u[4] = causal ? 1 : 0
  u[5] = numKVHeads
  u[6] = windowSize
  return new Uint8Array(buf)
}

// Forward: compute attention output + save stats for backward
// Q: [numQHeads, N, d], K: [numKVHeads, N, d], V: [numKVHeads, N, d]
// opts: { causal, numKVHeads, windowSize } or boolean for backward compat
// Returns { O, L, M } where L = row sums, M = row maxes
function flashAttentionForward(Q_, K_, V_, opts = {}) {
  if (typeof opts === 'boolean') opts = { causal: opts }
  const { causal = true, windowSize = 0 } = opts

  const Q = T.contiguous(Q_)
  const K = T.contiguous(K_)
  const V = T.contiguous(V_)

  const numQHeads = Q.shape[0]
  const numKVHeads = opts.numKVHeads || K.shape[0]
  const N = Q.shape[1]
  const d = Q.shape[2]
  const scale = 1.0 / Math.sqrt(d)

  const O = T.create([numQHeads, N, d], Q.dtype)
  const L = T.create([numQHeads, N], 'f32')     // row sums (always f32 for precision)
  const M = T.create([numQHeads, N], 'f32')     // row maxes (always f32 for precision)

  const params = flashAttnParams(N, d, numQHeads, scale, causal, numKVHeads, windowSize)

  const numRowBlocks = Math.ceil(N / Br)
  const tpg = Math.min(Bc, 32) // threads per group — kept small for register pressure

  run(k('flash_attention_forward', Q.dtype), [
    { buffer: Q.buffer, index: 0 },
    { buffer: K.buffer, index: 1 },
    { buffer: V.buffer, index: 2 },
    { buffer: O.buffer, index: 3 },
    { buffer: L.buffer, index: 4 },
    { buffer: M.buffer, index: 5 },
  ],
  { x: numQHeads * tpg, y: numRowBlocks },
  { x: tpg, y: 1 },
  { data: params, index: 6 })

  return { O, L, M }
}

// Backward: compute dQ, dK, dV from dO and saved stats
function flashAttentionBackward(Q_, K_, V_, O_, dO_, L_, M_, opts = {}) {
  if (typeof opts === 'boolean') opts = { causal: opts }
  const { causal = true, windowSize = 0 } = opts

  const Q = T.contiguous(Q_)
  const K = T.contiguous(K_)
  const V = T.contiguous(V_)
  const O = T.contiguous(O_)
  const dO = T.contiguous(dO_)
  const L = T.contiguous(L_)
  const M = T.contiguous(M_)

  const numQHeads = Q.shape[0]
  const numKVHeads = opts.numKVHeads || K.shape[0]
  const N = Q.shape[1]
  const d = Q.shape[2]
  const scale = 1.0 / Math.sqrt(d)

  // dQ indexed by Q heads, dK/dV indexed by KV heads
  const dQ = T.zeros([numQHeads, N, d], Q.dtype)
  const dK = T.zeros([numKVHeads, N, d], Q.dtype)
  const dV = T.zeros([numKVHeads, N, d], Q.dtype)

  const params = flashAttnParams(N, d, numQHeads, scale, causal, numKVHeads, windowSize)

  const numRowBlocks = Math.ceil(N / Br)
  const tpg = Math.min(Bc, 32)

  run(k('flash_attention_backward', Q.dtype), [
    { buffer: Q.buffer, index: 0 },
    { buffer: K.buffer, index: 1 },
    { buffer: V.buffer, index: 2 },
    { buffer: O.buffer, index: 3 },
    { buffer: dO.buffer, index: 4 },
    { buffer: L.buffer, index: 5 },
    { buffer: M.buffer, index: 6 },
    { buffer: dQ.buffer, index: 7 },
    { buffer: dK.buffer, index: 8 },
    { buffer: dV.buffer, index: 9 },
  ],
  { x: numQHeads * tpg, y: numRowBlocks },
  { x: tpg, y: 1 },
  { data: params, index: 10 })

  return { dQ, dK, dV }
}

export { flashAttentionForward, flashAttentionBackward }
