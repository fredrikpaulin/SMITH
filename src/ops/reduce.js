// smith/src/ops/reduce.js
// Reduction operations: sum, max along axes.
// Uses parallel reduction shaders.

import * as T from '../tensor.js'
import { run, axisReduceParams, scaleParams, GROUP_1D } from '../dispatch.js'

// Sum along a specific axis
function sumAxis(input, axis) {
  if (axis < 0) axis += input.shape.length
  const ndim = input.shape.length

  // Compute outer/inner sizes around the axis
  let outer = 1, inner = 1
  for (let i = 0; i < axis; i++) outer *= input.shape[i]
  for (let i = axis + 1; i < ndim; i++) inner *= input.shape[i]

  const outShape = input.shape.filter((_, i) => i !== axis)
  const outSize = outer * inner
  const out = T.create(outShape.length ? outShape : [], input.dtype)
  const params = axisReduceParams(outer, input.shape[axis], inner)

  run('reduce_sum_axis', [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], { x: outSize }, { x: Math.min(outSize, GROUP_1D) },
  { data: params, index: 2 })

  return out
}

// Max along a specific axis
function maxAxis(input, axis) {
  if (axis < 0) axis += input.shape.length
  const ndim = input.shape.length

  let outer = 1, inner = 1
  for (let i = 0; i < axis; i++) outer *= input.shape[i]
  for (let i = axis + 1; i < ndim; i++) inner *= input.shape[i]

  const outShape = input.shape.filter((_, i) => i !== axis)
  const outSize = outer * inner
  const out = T.create(outShape.length ? outShape : [], input.dtype)
  const params = axisReduceParams(outer, input.shape[axis], inner)

  run('reduce_max_axis', [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], { x: outSize }, { x: Math.min(outSize, GROUP_1D) },
  { data: params, index: 2 })

  return out
}

// Full sum (reduce to scalar)
function sumAll(input) {
  // Use axis reduction cascading through all dims
  let current = input
  for (let i = current.shape.length - 1; i >= 0; i--) {
    current = sumAxis(current, i)
  }
  return current
}

// Public: sum with optional axis
function sum(input, axis) {
  if (axis === undefined || axis === null) return sumAll(input)
  return sumAxis(input, axis)
}

function max(input, axis) {
  if (axis === undefined || axis === null) {
    let current = input
    for (let i = current.shape.length - 1; i >= 0; i--) {
      current = maxAxis(current, i)
    }
    return current
  }
  return maxAxis(input, axis)
}

function mean(input, axis) {
  const s = sum(input, axis)
  const n = axis === undefined || axis === null
    ? T.shapeSize(input.shape)
    : input.shape[axis < 0 ? axis + input.shape.length : axis]
  // scale by 1/n — inline dispatch to avoid circular import
  const out = T.create(s.shape, s.dtype)
  run('elementwise_scale', [
    { buffer: s.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], { x: s.size }, null, { data: scaleParams(1 / n), index: 2 })
  return out
}

export { sum, sumAxis, max, maxAxis, mean }
