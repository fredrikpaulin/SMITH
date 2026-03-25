// smith/src/ops/cast.js
// Dtype conversion between f32 and f16 via Metal compute shaders.

import * as T from '../tensor.js'
import { runElementwise } from '../dispatch.js'

// Cast a tensor to a different dtype
function cast(input, targetDtype) {
  if (input.dtype === targetDtype) return input

  const kernelName = input.dtype === 'f32' && targetDtype === 'f16'
    ? 'cast_f32_to_f16'
    : input.dtype === 'f16' && targetDtype === 'f32'
      ? 'cast_f16_to_f32'
      : null

  if (!kernelName) throw new Error(`cast: unsupported ${input.dtype} → ${targetDtype}`)

  const out = T.create(input.shape, targetDtype)
  runElementwise(kernelName, [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], input.size)
  return out
}

export { cast }
