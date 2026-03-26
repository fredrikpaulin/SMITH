// smith/src/ops/tanh.js
// Tanh activation: tanh(x)
// Backward: grad * (1 - tanh(x)²) — uses saved output

import * as T from '../tensor.js'
import { runElementwise, k } from '../dispatch.js'

function tanh(input) {
  const out = T.create(input.shape, input.dtype)
  runElementwise(k('tanh_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], input.size)
  return out
}

function tanhBackward(output, gradOutput) {
  const gradInput = T.create(output.shape, output.dtype)
  runElementwise(k('tanh_backward', output.dtype), [
    { buffer: output.buffer, index: 0 },
    { buffer: gradOutput.buffer, index: 1 },
    { buffer: gradInput.buffer, index: 2 },
  ], output.size)
  return gradInput
}

export { tanh, tanhBackward }
