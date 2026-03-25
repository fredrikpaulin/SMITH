// smith/src/ops/gelu.js
// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Backward formula ported from TinyFormer.

import * as T from '../tensor.js'
import { runElementwise, k } from '../dispatch.js'

function gelu(input) {
  const out = T.create(input.shape, input.dtype)
  runElementwise(k('gelu_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], input.size)
  return out
}

function geluBackward(input, gradOutput) {
  const gradInput = T.create(input.shape, input.dtype)
  runElementwise(k('gelu_backward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: gradOutput.buffer, index: 1 },
    { buffer: gradInput.buffer, index: 2 },
  ], input.size)
  return gradInput
}

export { gelu, geluBackward }
