// smith/src/ops/rope.js
// GPU RoPE (Rotary Position Embeddings) dispatch.

import * as T from '../tensor.js'
import { run, k } from '../dispatch.js'

// Precompute cos/sin frequency tables for RoPE
// Returns { cos, sin } tensors of shape [maxSeqLen, halfDim]
// cos/sin tables are always f32 for precision
function precomputeRoPE(dim, maxSeqLen, freqBase = 10000) {
  const halfDim = dim / 2
  const freqs = new Float32Array(halfDim)
  for (let i = 0; i < halfDim; i++) {
    freqs[i] = 1.0 / Math.pow(freqBase, (2 * i) / dim)
  }

  const cosData = new Float32Array(maxSeqLen * halfDim)
  const sinData = new Float32Array(maxSeqLen * halfDim)
  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const angle = pos * freqs[i]
      cosData[pos * halfDim + i] = Math.cos(angle)
      sinData[pos * halfDim + i] = Math.sin(angle)
    }
  }

  return {
    cos: T.tensor(Array.from(cosData), [maxSeqLen, halfDim]),
    sin: T.tensor(Array.from(sinData), [maxSeqLen, halfDim]),
  }
}

function ropeParams(seqLen, dim, startPos) {
  return new Uint32Array([seqLen, dim, dim / 2, startPos])
}

// Apply RoPE: input [seqLen, dim] → output [seqLen, dim]
// ropeTable: { cos, sin } from precomputeRoPE (always f32)
function ropeForward(input, ropeTable, startPos = 0) {
  const seqLen = input.shape[0]
  const dim = input.shape[1]
  const halfDim = dim / 2
  const out = T.create(input.shape, input.dtype)
  const params = ropeParams(seqLen, dim, startPos)

  const grpX = Math.min(halfDim, 256)
  run(k('rope_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: ropeTable.cos.buffer, index: 1 },
    { buffer: ropeTable.sin.buffer, index: 2 },
    { buffer: out.buffer, index: 3 },
  ], { x: halfDim, y: seqLen }, { x: grpX, y: 1 },
  { data: params, index: 4 })

  return out
}

// Backward: RoPE backward is rotation with negated sin
function ropeBackward(gradOut, ropeTable, startPos = 0) {
  const seqLen = gradOut.shape[0]
  const dim = gradOut.shape[1]
  const halfDim = dim / 2
  const gradIn = T.create(gradOut.shape, gradOut.dtype)
  const params = ropeParams(seqLen, dim, startPos)

  const grpX = Math.min(halfDim, 256)
  run(k('rope_backward', gradOut.dtype), [
    { buffer: gradOut.buffer, index: 0 },
    { buffer: ropeTable.cos.buffer, index: 1 },
    { buffer: ropeTable.sin.buffer, index: 2 },
    { buffer: gradIn.buffer, index: 3 },
  ], { x: halfDim, y: seqLen }, { x: grpX, y: 1 },
  { data: params, index: 4 })

  return gradIn
}

export { precomputeRoPE, ropeForward, ropeBackward }
