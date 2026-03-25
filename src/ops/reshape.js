// smith/src/ops/reshape.js
// Virtual reshape: no GPU work, just changes shape/strides.
// Handles -1 inference (one unknown dimension), ported from TinyFormer.

import * as T from '../tensor.js'

function reshape(input, newShape) {
  // Handle -1 (infer one dimension)
  const neg = newShape.indexOf(-1)
  if (neg !== -1) {
    let known = 1
    for (let i = 0; i < newShape.length; i++) if (i !== neg) known *= newShape[i]
    newShape = newShape.slice()
    newShape[neg] = input.size / known
  }

  if (T.shapeSize(newShape) !== input.size) {
    throw new Error(`Cannot reshape [${input.shape}] to [${newShape}]`)
  }

  return {
    buffer: input.buffer,
    data: input.data,
    shape: newShape.slice(),
    strides: T.computeStrides(newShape),
    dtype: input.dtype,
    size: input.size,
    offset: input.offset || 0,
  }
}

export { reshape }
