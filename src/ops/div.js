// smith/src/ops/div.js
// Element-wise division with broadcasting support.

import * as T from '../tensor.js'
import { run, runElementwise, broadcastParams } from '../dispatch.js'
import { computeBroadcastStrides } from './add.js'

function gpuDiv(a, b) {
  const out = T.create(a.shape, a.dtype)
  runElementwise('elementwise_div', [
    { buffer: a.buffer, index: 0 },
    { buffer: b.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], a.size)
  return out
}

function gpuBroadcastDiv(a, b) {
  const outShape = T.broadcastShapes(a.shape, b.shape)
  const outSize = T.shapeSize(outShape)
  const out = T.create(outShape, a.dtype)
  const aStrides = computeBroadcastStrides(a.shape, outShape)
  const bStrides = computeBroadcastStrides(b.shape, outShape)
  const params = broadcastParams(outShape, aStrides, bStrides, outSize)
  run('broadcast_div', [
    { buffer: a.buffer, index: 0 },
    { buffer: b.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], { x: outSize }, null, { data: params, index: 3 })
  return out
}

function div(a, b) {
  if (T.shapesEqual(a.shape, b.shape)) return gpuDiv(a, b)
  return gpuBroadcastDiv(a, b)
}

export { div, gpuDiv, gpuBroadcastDiv }
