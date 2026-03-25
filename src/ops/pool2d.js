// smith/src/ops/pool2d.js
// 2D pooling: max pool and average pool forward/backward.
// Input layout: NCHW [batch, channels, height, width]

import * as T from '../tensor.js'
import { run } from '../dispatch.js'

function pool2dParams(batch, channels, inH, inW, outH, outW, kH, kW, strideH, strideW, padH, padW) {
  const params = new Uint32Array(12)
  params[0]  = batch
  params[1]  = channels
  params[2]  = inH
  params[3]  = inW
  params[4]  = outH
  params[5]  = outW
  params[6]  = kH
  params[7]  = kW
  params[8]  = strideH
  params[9]  = strideW
  params[10] = padH
  params[11] = padW
  return params
}

function poolOutputSize(inSize, kSize, stride, pad) {
  return Math.floor((inSize + 2 * pad - kSize) / stride) + 1
}

// Max pool forward — returns output and argmax indices (for backward)
function maxPool2dForward(input, opts = {}) {
  const { kernelSize = 2, stride, padding = 0 } = opts
  const [kH, kW] = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize]
  const [strideH, strideW] = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : [kH, kW]
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]

  const [batch, channels, inH, inW] = input.shape
  const outH = poolOutputSize(inH, kH, strideH, padH)
  const outW = poolOutputSize(inW, kW, strideW, padW)

  const out = T.create([batch, channels, outH, outW], input.dtype)
  const indices = T.create([batch, channels, outH, outW], 'u32')
  const params = pool2dParams(batch, channels, inH, inW, outH, outW, kH, kW, strideH, strideW, padH, padW)

  run('maxpool2d_forward', [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
    { buffer: indices.buffer, index: 2 },
  ], { x: outW, y: outH, z: batch * channels },
  { x: Math.min(outW, 16), y: Math.min(outH, 16), z: 1 },
  { data: params, index: 3 })

  return { out, indices }
}

// Max pool backward — scatter gradient to max positions
function maxPool2dBackward(gradOutput, indices, inputShape) {
  const gradInput = T.zeros(inputShape, gradOutput.dtype)
  const totalOut = T.shapeSize(gradOutput.shape)

  const [batch, channels, inH, inW] = inputShape
  const [, , outH, outW] = gradOutput.shape
  const params = pool2dParams(batch, channels, inH, inW, outH, outW, 0, 0, 0, 0, 0, 0)

  run('maxpool2d_backward', [
    { buffer: gradOutput.buffer, index: 0 },
    { buffer: indices.buffer, index: 1 },
    { buffer: gradInput.buffer, index: 2 },
  ], { x: totalOut }, { x: Math.min(totalOut, 256) },
  { data: params, index: 3 })

  return gradInput
}

// Average pool forward
function avgPool2dForward(input, opts = {}) {
  const { kernelSize = 2, stride, padding = 0 } = opts
  const [kH, kW] = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize]
  const [strideH, strideW] = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : [kH, kW]
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]

  const [batch, channels, inH, inW] = input.shape
  const outH = poolOutputSize(inH, kH, strideH, padH)
  const outW = poolOutputSize(inW, kW, strideW, padW)

  const out = T.create([batch, channels, outH, outW], input.dtype)
  const params = pool2dParams(batch, channels, inH, inW, outH, outW, kH, kW, strideH, strideW, padH, padW)

  run('avgpool2d_forward', [
    { buffer: input.buffer, index: 0 },
    { buffer: out.buffer, index: 1 },
  ], { x: outW, y: outH, z: batch * channels },
  { x: Math.min(outW, 16), y: Math.min(outH, 16), z: 1 },
  { data: params, index: 2 })

  return out
}

// Average pool backward
function avgPool2dBackward(gradOutput, inputShape, opts = {}) {
  const { kernelSize = 2, stride, padding = 0 } = opts
  const [kH, kW] = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize]
  const [strideH, strideW] = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : [kH, kW]
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]

  const [batch, channels, inH, inW] = inputShape
  const [, , outH, outW] = gradOutput.shape

  const gradInput = T.zeros(inputShape, gradOutput.dtype)
  const params = pool2dParams(batch, channels, inH, inW, outH, outW, kH, kW, strideH, strideW, padH, padW)

  run('avgpool2d_backward', [
    { buffer: gradOutput.buffer, index: 0 },
    { buffer: gradInput.buffer, index: 1 },
  ], { x: inW, y: inH, z: batch * channels },
  { x: Math.min(inW, 16), y: Math.min(inH, 16), z: 1 },
  { data: params, index: 2 })

  return gradInput
}

export {
  maxPool2dForward, maxPool2dBackward,
  avgPool2dForward, avgPool2dBackward,
  poolOutputSize, pool2dParams,
}
