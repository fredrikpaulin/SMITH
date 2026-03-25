// smith/src/ops/add.js
// Element-wise addition with broadcasting support.
// Backward: addGrad(a, grad), addGrad(b, grad)

import * as T from '../tensor.js'
import { run, runElementwise, broadcastParams, k } from '../dispatch.js'

// GPU add for same-shape tensors (no broadcast)
function gpuAdd(a, b) {
  const out = T.create(a.shape, a.dtype)
  runElementwise(k('elementwise_add', a.dtype), [
    { buffer: a.buffer, index: 0 },
    { buffer: b.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], a.size)
  return out
}

// GPU add with broadcasting
function gpuBroadcastAdd(a, b) {
  const outShape = T.broadcastShapes(a.shape, b.shape)
  const outSize = T.shapeSize(outShape)
  const out = T.create(outShape, a.dtype)

  // Compute broadcast strides for a and b
  const aStrides = computeBroadcastStrides(a.shape, outShape)
  const bStrides = computeBroadcastStrides(b.shape, outShape)
  const params = broadcastParams(outShape, aStrides, bStrides, outSize)

  run(k('broadcast_add', a.dtype), [
    { buffer: a.buffer, index: 0 },
    { buffer: b.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], { x: outSize }, null, { data: params, index: 3 })
  return out
}

// Compute the strides a source tensor would have if broadcast to targetShape.
// Dims of size 1 get stride 0 (read same element repeatedly).
function computeBroadcastStrides(srcShape, targetShape) {
  const ndim = targetShape.length
  const padded = ndim - srcShape.length
  const strides = new Array(ndim)
  // Compute source strides
  const srcStrides = T.computeStrides(srcShape)
  for (let i = 0; i < ndim; i++) {
    const si = i - padded
    if (si < 0 || srcShape[si] === 1) {
      strides[i] = 0 // broadcast dimension
    } else {
      strides[i] = srcStrides[si]
    }
  }
  return strides
}

// Public: dispatches broadcast or simple path
function add(a, b) {
  if (T.shapesEqual(a.shape, b.shape)) return gpuAdd(a, b)
  return gpuBroadcastAdd(a, b)
}

export { add, gpuAdd, gpuBroadcastAdd, computeBroadcastStrides }
