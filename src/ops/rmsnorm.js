// smith/src/ops/rmsnorm.js
// GPU RMSNorm (Llama-style) dispatch.

import * as T from '../tensor.js'
import { run, k } from '../dispatch.js'

function rmsnormParams(rows, cols, eps) {
  const buf = new ArrayBuffer(12)
  const u = new Uint32Array(buf, 0, 2)
  const f = new Float32Array(buf, 8, 1)
  u[0] = rows
  u[1] = cols
  f[0] = eps
  return new Uint8Array(buf)
}

// Forward: out = gamma * x / sqrt(mean(x²) + eps)
// input: [rows..., cols], gamma: [cols]
function rmsnormForward(input, gamma, eps = 1e-5) {
  const ndim = input.shape.length
  const cols = input.shape[ndim - 1]
  let rows = 1
  for (let i = 0; i < ndim - 1; i++) rows *= input.shape[i]

  const out = T.create(input.shape, input.dtype)
  const params = rmsnormParams(rows, cols, eps)
  const tpg = 1 << Math.ceil(Math.log2(Math.max(Math.min(cols, 256), 2)))

  run(k('rmsnorm_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: gamma.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], { x: rows * tpg }, { x: tpg },
  { data: params, index: 3 })

  return out
}

// Backward: computes gradInput on GPU, gradGamma accumulated on CPU
function rmsnormBackward(gradOut, input, gamma, eps = 1e-5) {
  const ndim = input.shape.length
  const cols = input.shape[ndim - 1]
  let rows = 1
  for (let i = 0; i < ndim - 1; i++) rows *= input.shape[i]

  const gradInput = T.create(input.shape, input.dtype)
  const params = rmsnormParams(rows, cols, eps)
  const tpg = 1 << Math.ceil(Math.log2(Math.max(Math.min(cols, 256), 2)))

  run(k('rmsnorm_backward', input.dtype), [
    { buffer: gradOut.buffer, index: 0 },
    { buffer: input.buffer, index: 1 },
    { buffer: gamma.buffer, index: 2 },
    { buffer: gradInput.buffer, index: 3 },
  ], { x: rows * tpg }, { x: tpg },
  { data: params, index: 4 })

  // gradGamma: CPU accumulation (sum over rows of gradOut * x * rms_inv)
  const gradGamma = T.create(gamma.shape, gamma.dtype)
  const ggAcc = new Float32Array(cols)
  for (let r = 0; r < rows; r++) {
    // Recompute rms for this row
    let sumSq = 0
    for (let c = 0; c < cols; c++) {
      const v = T.getValue(input, r * cols + c)
      sumSq += v * v
    }
    const rmsInv = 1 / Math.sqrt(sumSq / cols + eps)
    for (let c = 0; c < cols; c++) {
      ggAcc[c] += T.getValue(gradOut, r * cols + c) * T.getValue(input, r * cols + c) * rmsInv
    }
  }
  for (let c = 0; c < cols; c++) {
    T.setValue(gradGamma, c, ggAcc[c])
  }

  return { gradInput, gradGamma }
}

export { rmsnormForward, rmsnormBackward }
