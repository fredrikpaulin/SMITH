// smith/src/ops/transpose.js
// Virtual transpose: no GPU work, just manipulates shape/strides.
// Ported from TinyFormer's tensor.js transpose logic.

import * as T from '../tensor.js'

function transpose(input, axes) {
  if (!axes) {
    // Default: reverse all dimensions
    axes = []
    for (let i = input.shape.length - 1; i >= 0; i--) axes.push(i)
  }
  return {
    buffer: input.buffer,
    data: input.data,
    shape: axes.map(a => input.shape[a]),
    strides: axes.map(a => input.strides[a]),
    dtype: input.dtype,
    size: input.size,
    offset: input.offset || 0,
  }
}

// Compute the inverse permutation of an axes array
function inverseAxes(axes) {
  const inv = new Array(axes.length)
  for (let i = 0; i < axes.length; i++) inv[axes[i]] = i
  return inv
}

export { transpose, inverseAxes }
