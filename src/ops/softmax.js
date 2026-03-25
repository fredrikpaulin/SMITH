// smith/src/ops/softmax.js
// GPU softmax along the last dimension.

import * as T from '../tensor.js'
import { run, k } from '../dispatch.js'

function softmax(input, axis = -1) {
  const ndim = input.shape.length
  const ax = axis < 0 ? axis + ndim : axis
  if (ax !== ndim - 1) throw new Error('softmax only supports last-axis for now')

  const cols = input.shape[ax]
  let rows = 1
  for (let i = 0; i < ax; i++) rows *= input.shape[i]

  const out = T.create(input.shape, input.dtype)
  const params = new Uint32Array([rows, cols])
  const tpg = Math.min(cols, 256)
  // Round tpg to next power of 2 for reduction to work
  const tpgPow2 = 1 << Math.ceil(Math.log2(Math.max(tpg, 2)))

  run(k('softmax_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], { x: rows * tpgPow2 }, { x: tpgPow2 },
  { data: params, index: 2 })

  return out
}

export { softmax }
