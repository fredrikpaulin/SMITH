// smith/src/ops/swiglu.js
// GPU SwiGLU activation: out = silu(gate) * up
// Fuses SiLU activation and element-wise multiply.

import * as T from '../tensor.js'
import { runElementwise, k } from '../dispatch.js'

// Forward: out = silu(gate) * up
// gate, up: same shape tensors
function swigluForward(gate, up) {
  const out = T.create(gate.shape, gate.dtype)
  runElementwise(k('swiglu_forward', gate.dtype), [
    { buffer: gate.buffer, index: 0 },
    { buffer: up.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], gate.size)
  return out
}

// Backward: computes gradGate and gradUp
// gradGate = gradOut * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
// gradUp = gradOut * silu(gate)
function swigluBackward(gradOut, gate, up) {
  const gradGate = T.create(gate.shape, gate.dtype)
  const gradUp = T.create(up.shape, up.dtype)
  runElementwise(k('swiglu_backward', gate.dtype), [
    { buffer: gradOut.buffer, index: 0 },
    { buffer: gate.buffer, index: 1 },
    { buffer: up.buffer, index: 2 },
    { buffer: gradGate.buffer, index: 3 },
    { buffer: gradUp.buffer, index: 4 },
  ], gate.size)
  return { gradGate, gradUp }
}

export { swigluForward, swigluBackward }
