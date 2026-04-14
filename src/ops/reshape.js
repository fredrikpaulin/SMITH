// smith/src/ops/reshape.js
// Virtual reshape: no GPU work, just changes shape/strides.
// Handles -1 inference (one unknown dimension), ported from TinyFormer.

import * as T from '../tensor.js'

function reshape(input, newShape) {
  // Handle -1 (infer one dimension)
  const neg = newShape.indexOf(-1)
  if (neg !== -1) {
    if (newShape.lastIndexOf(-1) !== neg) {
      throw new Error(`reshape: only one dimension can be -1, got [${newShape}]`)
    }
    let known = 1
    for (let i = 0; i < newShape.length; i++) if (i !== neg) known *= newShape[i]
    newShape = newShape.slice()
    newShape[neg] = input.size / known
  }

  if (T.shapeSize(newShape) !== input.size) {
    throw new Error(`Cannot reshape [${input.shape}] to [${newShape}]`)
  }

  // Reshape assumes contiguous data layout. If the input is a non-contiguous
  // view (e.g., from transpose), we must copy to contiguous memory first.
  // Without this, the new strides won't match the actual data layout.
  const src = T.isContiguous(input) ? input : T.contiguous(input)

  return {
    buffer: src.buffer,
    data: src.data,
    shape: newShape.slice(),
    strides: T.computeStrides(newShape),
    dtype: src.dtype,
    size: src.size,
    offset: src.offset || 0,
  }
}

export { reshape }
