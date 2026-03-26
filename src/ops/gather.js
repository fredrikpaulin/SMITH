// smith/src/ops/gather.js
// GPU dispatch for gather (indexed read) and scatter (indexed write) operations.
// Used for embedding lookup, advanced indexing, and their backward passes.
//
// Layout: input is treated as [outer, dimSize, inner] where:
//   outer = product of dims before the gather axis
//   dimSize = size of the gather axis
//   inner = product of dims after the gather axis

import * as T from '../tensor.js'
import * as device from '../device.js'
import { run, GROUP_1D } from '../dispatch.js'

// Compute outer/inner products for a given axis
function axisLayout(shape, axis) {
  let outer = 1, inner = 1
  for (let i = 0; i < axis; i++) outer *= shape[i]
  for (let i = axis + 1; i < shape.length; i++) inner *= shape[i]
  return { outer, dimSize: shape[axis], inner }
}

// Build GatherParams struct: { outer, dimSize, inner, indexLen, totalOut }
function gatherParams(outer, dimSize, inner, indexLen) {
  return new Uint32Array([outer, dimSize, inner, indexLen, outer * indexLen * inner])
}

// gather: output[o][i][k] = input[o][indices[i]][k]
// input shape: [..., dimSize, ...], indices: flat uint32 array of length indexLen
// output shape: same as input but axis dimension replaced with indexLen
function gather(input, axis, indices) {
  const ax = axis < 0 ? input.shape.length + axis : axis
  const { outer, dimSize, inner } = axisLayout(input.shape, ax)
  const indexLen = indices.length

  // Build output shape
  const outShape = input.shape.slice()
  outShape[ax] = indexLen
  const out = T.create(outShape, input.dtype)
  const totalOut = outer * indexLen * inner

  // Upload indices to GPU buffer
  const idxBuf = T.create([indexLen])
  const u32View = new Uint32Array(idxBuf.data.buffer, idxBuf.data.byteOffset, indexLen)
  for (let i = 0; i < indexLen; i++) u32View[i] = indices[i]

  const params = gatherParams(outer, dimSize, inner, indexLen)
  run('gather_forward', [
    { buffer: input.buffer, index: 0 },
    { buffer: idxBuf.buffer, index: 1 },
    { buffer: out.buffer, index: 2 },
  ], { x: totalOut }, { x: Math.min(totalOut, GROUP_1D) },
  { data: params, index: 3 })

  return out
}

// scatterAdd: dst[o][indices[i]][k] += src[o][i][k]
// dst must be pre-allocated and pre-filled (usually zeros).
// Uses atomic add for duplicate indices.
function scatterAdd(dst, axis, indices, src) {
  const ax = axis < 0 ? dst.shape.length + axis : axis
  const { outer, dimSize, inner } = axisLayout(dst.shape, ax)
  const indexLen = indices.length
  const totalSrc = outer * indexLen * inner

  // Upload indices to GPU buffer
  const idxBuf = T.create([indexLen])
  const u32View = new Uint32Array(idxBuf.data.buffer, idxBuf.data.byteOffset, indexLen)
  for (let i = 0; i < indexLen; i++) u32View[i] = indices[i]

  const params = gatherParams(outer, dimSize, inner, indexLen)
  run('scatter_add', [
    { buffer: src.buffer, index: 0 },
    { buffer: idxBuf.buffer, index: 1 },
    { buffer: dst.buffer, index: 2 },
  ], { x: totalSrc }, { x: Math.min(totalSrc, GROUP_1D) },
  { data: params, index: 3 })

  return dst
}

// scatter: output = input.clone(), then output[o][indices[i]][k] = src[o][i][k]
// Non-atomic write — last write wins for duplicate indices.
function scatter(input, axis, indices, src) {
  const ax = axis < 0 ? input.shape.length + axis : axis
  const { outer, dimSize, inner } = axisLayout(input.shape, ax)
  const indexLen = indices.length
  const totalSrc = outer * indexLen * inner

  // Clone input to output
  const out = T.create(input.shape, input.dtype)
  out.data.set(input.data)

  // Upload indices to GPU buffer
  const idxBuf = T.create([indexLen])
  const u32View = new Uint32Array(idxBuf.data.buffer, idxBuf.data.byteOffset, indexLen)
  for (let i = 0; i < indexLen; i++) u32View[i] = indices[i]

  const params = gatherParams(outer, dimSize, inner, indexLen)
  run('scatter_forward', [
    { buffer: input.buffer, index: 0 },
    { buffer: src.buffer, index: 1 },
    { buffer: idxBuf.buffer, index: 2 },
    { buffer: out.buffer, index: 3 },
  ], { x: totalSrc }, { x: Math.min(totalSrc, GROUP_1D) },
  { data: params, index: 4 })

  return out
}

export { gather, scatterAdd, scatter, axisLayout, gatherParams }
