// smith/src/ops/relusquared.js
// Fused relu(x)^2 activation (used in autoresearch MLP)
// Forward: max(0, x)^2
// Backward: grad * 2 * max(0, x)

import * as T from '../tensor.js'
import { runElementwise, k } from '../dispatch.js'

function reluSquared(input) {
  const out = T.create(input.shape, input.dtype)
  runElementwise(k('relusquared_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], input.size)
  return out
}

function reluSquaredBackward(input, gradOutput) {
  const gradInput = T.create(input.shape, input.dtype)
  runElementwise(k('relusquared_backward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: gradOutput.buffer, index: 1 },
    { buffer: gradInput.buffer, index: 2 },
  ], input.size)
  return gradInput
}

export { reluSquared, reluSquaredBackward }
