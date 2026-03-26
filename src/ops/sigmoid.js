// smith/src/ops/sigmoid.js
// Sigmoid activation: 1 / (1 + exp(-x))
// Backward: grad * sigmoid(x) * (1 - sigmoid(x)) — uses saved output

import * as T from '../tensor.js'
import { runElementwise, k } from '../dispatch.js'

function sigmoid(input) {
  const out = T.create(input.shape, input.dtype)
  runElementwise(k('sigmoid_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], input.size)
  return out
}

function sigmoidBackward(output, gradOutput) {
  const gradInput = T.create(output.shape, output.dtype)
  runElementwise(k('sigmoid_backward', output.dtype), [
    { buffer: output.buffer, index: 0 },
    { buffer: gradOutput.buffer, index: 1 },
    { buffer: gradInput.buffer, index: 2 },
  ], output.size)
  return gradInput
}

export { sigmoid, sigmoidBackward }
