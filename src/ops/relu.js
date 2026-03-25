// smith/src/ops/relu.js
// ReLU activation: max(0, x)
// Backward: grad * (input > 0)

import * as T from '../tensor.js'
import { runElementwise, k } from '../dispatch.js'

function relu(input) {
  const out = T.create(input.shape, input.dtype)
  runElementwise(k('relu_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], input.size)
  return out
}

function reluBackward(input, gradOutput) {
  const gradInput = T.create(input.shape, input.dtype)
  runElementwise(k('relu_backward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: gradOutput.buffer, index: 1 },
    { buffer: gradInput.buffer, index: 2 },
  ], input.size)
  return gradInput
}

export { relu, reluBackward }
