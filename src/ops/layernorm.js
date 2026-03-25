// smith/src/ops/layernorm.js
// GPU layer normalization along the last dimension.

import * as T from '../tensor.js'
import { run, k } from '../dispatch.js'

function layernormForward(input, gamma, beta, eps = 1e-5) {
  const ndim = input.shape.length
  const cols = input.shape[ndim - 1]
  let rows = 1
  for (let i = 0; i < ndim - 1; i++) rows *= input.shape[i]

  const out = T.create(input.shape, input.dtype)
  const xhat = T.create(input.shape, input.dtype) // saved for backward

  const buf = new ArrayBuffer(12)
  const u = new Uint32Array(buf, 0, 2)
  const f = new Float32Array(buf, 8, 1)
  u[0] = rows
  u[1] = cols
  f[0] = eps
  const params = new Uint8Array(buf)

  const tpg = 1 << Math.ceil(Math.log2(Math.max(Math.min(cols, 256), 2)))

  run(k('layernorm_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: gamma.buffer, index: 1 },
    { buffer: beta.buffer, index: 2 },
    { buffer: out.buffer, index: 3 },
    { buffer: xhat.buffer, index: 4 },
  ], { x: rows * tpg }, { x: tpg },
  { data: params, index: 5 })

  return { out, xhat }
}

function layernormBackward(gradOut, xhat, gamma, input, eps = 1e-5) {
  const ndim = input.shape.length
  const cols = input.shape[ndim - 1]
  let rows = 1
  for (let i = 0; i < ndim - 1; i++) rows *= input.shape[i]

  const gradInput = T.create(input.shape, input.dtype)
  // grad_gamma and grad_beta: accumulated over rows on GPU per-element
  // We'll compute them on CPU from grad_out and xhat (unified memory, simple)
  const gradGamma = T.create(gamma.shape, gamma.dtype)
  const gradBeta = T.create(gamma.shape, gamma.dtype)

  // CPU accumulation for grad_gamma and grad_beta
  // Use f32 accumulators then write back (handles both f32 and f16 dtypes)
  const ggAcc = new Float32Array(cols)
  const gbAcc = new Float32Array(cols)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c
      const go = T.getValue(gradOut, idx)
      const xh = T.getValue(xhat, idx)
      ggAcc[c] += go * xh
      gbAcc[c] += go
    }
  }
  for (let c = 0; c < cols; c++) {
    T.setValue(gradGamma, c, ggAcc[c])
    T.setValue(gradBeta, c, gbAcc[c])
  }

  // GPU backward for grad_input
  const buf = new ArrayBuffer(12)
  const u = new Uint32Array(buf, 0, 2)
  const f = new Float32Array(buf, 8, 1)
  u[0] = rows
  u[1] = cols
  f[0] = eps
  const params = new Uint8Array(buf)

  const tpg = 1 << Math.ceil(Math.log2(Math.max(Math.min(cols, 256), 2)))

  run(k('layernorm_backward', input.dtype), [
    { buffer: gradOut.buffer, index: 0 },
    { buffer: xhat.buffer, index: 1 },
    { buffer: gamma.buffer, index: 2 },
    { buffer: input.buffer, index: 3 },
    { buffer: gradInput.buffer, index: 4 },
    { buffer: gradGamma.buffer, index: 5 },
    { buffer: gradBeta.buffer, index: 6 },
  ], { x: rows * tpg }, { x: tpg },
  { data: params, index: 7 })

  return { gradInput, gradGamma, gradBeta }
}

export { layernormForward, layernormBackward }
