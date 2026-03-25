// smith/src/ops/mul.js
// Element-wise multiplication with broadcasting support.
// Backward: addGrad(a, grad * b), addGrad(b, grad * a)

import * as T from '../tensor.js'
import { run, runElementwise, broadcastParams, scaleParams, k } from '../dispatch.js'
import { computeBroadcastStrides } from './add.js'

function gpuMul(a, b) {
  const out = T.create(a.shape, a.dtype)
  runElementwise(k('elementwise_mul', a.dtype), [
    { buffer: a.buffer, index: 0 },
    { buffer: b.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], a.size)
  return out
}

function gpuBroadcastMul(a, b) {
  const outShape = T.broadcastShapes(a.shape, b.shape)
  const outSize = T.shapeSize(outShape)
  const out = T.create(outShape, a.dtype)

  const aStrides = computeBroadcastStrides(a.shape, outShape)
  const bStrides = computeBroadcastStrides(b.shape, outShape)
  const params = broadcastParams(outShape, aStrides, bStrides, outSize)

  run(k('broadcast_mul', a.dtype), [
    { buffer: a.buffer, index: 0 },
    { buffer: b.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], { x: outSize }, null, { data: params, index: 3 })
  return out
}

function mul(a, b) {
  if (T.shapesEqual(a.shape, b.shape)) return gpuMul(a, b)
  return gpuBroadcastMul(a, b)
}

// Scalar multiply
function scale(a, s) {
  const out = T.create(a.shape, a.dtype)
  run(k('elementwise_scale', a.dtype), [
    { buffer: a.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], { x: a.size }, null, { data: scaleParams(s), index: 2 })
  return out
}

function neg(a) {
  const out = T.create(a.shape, a.dtype)
  runElementwise(k('elementwise_neg', a.dtype), [
    { buffer: a.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], a.size)
  return out
}

export { mul, gpuMul, gpuBroadcastMul, scale, neg }
