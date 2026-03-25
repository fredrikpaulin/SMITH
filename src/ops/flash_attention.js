// smith/src/ops/flash_attention.js
// GPU flash attention dispatch.
// Forward: fused Q*K^T → softmax → V in tiles, O(n) memory.
// Backward: recomputes attention from saved log-sum-exp stats.

import * as T from '../tensor.js'
import { run, k } from '../dispatch.js'

// Tile sizes must match the shader constants
const Br = 32
const Bc = 32

// Build params struct matching FlashAttnParams in the shader:
// { uint N, uint d, uint numHeads, float scale, uint causal }
function flashAttnParams(N, d, numHeads, scale, causal) {
  const buf = new ArrayBuffer(20) // 4 uints + 1 float = 20 bytes
  const u = new Uint32Array(buf)
  const f = new Float32Array(buf)
  u[0] = N
  u[1] = d
  u[2] = numHeads
  f[3] = scale         // float at byte offset 12
  u[4] = causal ? 1 : 0
  return new Uint8Array(buf)
}

// Forward: compute attention output + save stats for backward
// Q, K, V: contiguous tensors [numHeads, N, d]
// Returns { O, L, M } where L = row sums, M = row maxes
function flashAttentionForward(Q_, K_, V_, causal = true) {
  const Q = T.contiguous(Q_)
  const K = T.contiguous(K_)
  const V = T.contiguous(V_)

  const numHeads = Q.shape[0]
  const N = Q.shape[1]
  const d = Q.shape[2]
  const scale = 1.0 / Math.sqrt(d)

  const O = T.create([numHeads, N, d], Q.dtype)
  const L = T.create([numHeads, N], 'f32')     // row sums (always f32 for precision)
  const M = T.create([numHeads, N], 'f32')     // row maxes (always f32 for precision)

  const params = flashAttnParams(N, d, numHeads, scale, causal)

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
  { x: numHeads * tpg, y: numRowBlocks },
  { x: tpg, y: 1 },
  { data: params, index: 6 })

  return { O, L, M }
}

// Backward: compute dQ, dK, dV from dO and saved stats
function flashAttentionBackward(Q_, K_, V_, O_, dO_, L_, M_, causal = true) {
  const Q = T.contiguous(Q_)
  const K = T.contiguous(K_)
  const V = T.contiguous(V_)
  const O = T.contiguous(O_)
  const dO = T.contiguous(dO_)
  const L = T.contiguous(L_)
  const M = T.contiguous(M_)

  const numHeads = Q.shape[0]
  const N = Q.shape[1]
  const d = Q.shape[2]
  const scale = 1.0 / Math.sqrt(d)

  // Initialize dQ, dK, dV to zero
  const dQ = T.zeros([numHeads, N, d], Q.dtype)
  const dK = T.zeros([numHeads, N, d], Q.dtype)
  const dV = T.zeros([numHeads, N, d], Q.dtype)

  const params = flashAttnParams(N, d, numHeads, scale, causal)

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
  { x: numHeads * tpg, y: numRowBlocks },
  { x: tpg, y: 1 },
  { data: params, index: 10 })

  return { dQ, dK, dV }
}

export { flashAttentionForward, flashAttentionBackward }
