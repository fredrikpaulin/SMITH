// smith/src/ops/conv2d.js
// 2D convolution forward and backward dispatch.
// Input layout: NCHW [batch, channels, height, width]
// Weight layout: [outChannels, inChannels/groups, kH, kW]

import * as T from '../tensor.js'
import { run } from '../dispatch.js'

// Build the Conv2dParams struct matching the shader
function conv2dParams(batch, inC, inH, inW, outC, outH, outW, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, groups) {
  const params = new Uint32Array(16)
  params[0]  = batch
  params[1]  = inC
  params[2]  = inH
  params[3]  = inW
  params[4]  = outC
  params[5]  = outH
  params[6]  = outW
  params[7]  = kH
  params[8]  = kW
  params[9]  = strideH
  params[10] = strideW
  params[11] = padH
  params[12] = padW
  params[13] = dilationH
  params[14] = dilationW
  params[15] = groups
  return params
}

// Compute output spatial dimensions
function convOutputSize(inSize, kSize, stride, pad, dilation) {
  return Math.floor((inSize + 2 * pad - dilation * (kSize - 1) - 1) / stride) + 1
}

// Forward pass
function conv2dForward(input, weight, bias, opts = {}) {
  const { stride = 1, padding = 0, dilation = 1, groups = 1 } = opts
  const [strideH, strideW] = Array.isArray(stride) ? stride : [stride, stride]
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]
  const [dilationH, dilationW] = Array.isArray(dilation) ? dilation : [dilation, dilation]

  const [batch, inC, inH, inW] = input.shape
  const [outC, , kH, kW] = weight.shape

  const outH = convOutputSize(inH, kH, strideH, padH, dilationH)
  const outW = convOutputSize(inW, kW, strideW, padW, dilationW)
  const out = T.create([batch, outC, outH, outW], input.dtype)

  const params = conv2dParams(batch, inC, inH, inW, outC, outH, outW, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, groups)

  if (bias) {
    run('conv2d_forward', [
      { buffer: input.buffer, index: 0 },
      { buffer: weight.buffer, index: 1 },
      { buffer: bias.buffer, index: 2 },
      { buffer: out.buffer, index: 3 },
    ], { x: outW, y: outH, z: batch * outC },
    { x: Math.min(outW, 16), y: Math.min(outH, 16), z: 1 },
    { data: params, index: 4 })
  } else {
    run('conv2d_forward_no_bias', [
      { buffer: input.buffer, index: 0 },
      { buffer: weight.buffer, index: 1 },
      { buffer: out.buffer, index: 2 },
    ], { x: outW, y: outH, z: batch * outC },
    { x: Math.min(outW, 16), y: Math.min(outH, 16), z: 1 },
    { data: params, index: 3 })
  }

  return out
}

// Backward: gradient w.r.t. input
function conv2dBackwardInput(gradOutput, weight, inputShape, opts = {}) {
  const { stride = 1, padding = 0, dilation = 1, groups = 1 } = opts
  const [strideH, strideW] = Array.isArray(stride) ? stride : [stride, stride]
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]
  const [dilationH, dilationW] = Array.isArray(dilation) ? dilation : [dilation, dilation]

  const [batch, inC, inH, inW] = inputShape
  const [outC, , kH, kW] = weight.shape
  const outH = gradOutput.shape[2]
  const outW = gradOutput.shape[3]

  const gradInput = T.zeros(inputShape, gradOutput.dtype)
  const params = conv2dParams(batch, inC, inH, inW, outC, outH, outW, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, groups)

  run('conv2d_backward_input', [
    { buffer: gradOutput.buffer, index: 0 },
    { buffer: weight.buffer, index: 1 },
    { buffer: gradInput.buffer, index: 2 },
  ], { x: inW, y: inH, z: batch * inC },
  { x: Math.min(inW, 16), y: Math.min(inH, 16), z: 1 },
  { data: params, index: 3 })

  return gradInput
}

// Backward: gradient w.r.t. weight
function conv2dBackwardWeight(input, gradOutput, weightShape, opts = {}) {
  const { stride = 1, padding = 0, dilation = 1, groups = 1 } = opts
  const [strideH, strideW] = Array.isArray(stride) ? stride : [stride, stride]
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]
  const [dilationH, dilationW] = Array.isArray(dilation) ? dilation : [dilation, dilation]

  const [batch, inC, inH, inW] = input.shape
  const [outC, groupSize, kH, kW] = weightShape
  const outH = gradOutput.shape[2]
  const outW = gradOutput.shape[3]

  const gradWeight = T.zeros(weightShape, gradOutput.dtype)
  const params = conv2dParams(batch, inC, inH, inW, outC, outH, outW, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, groups)

  run('conv2d_backward_weight', [
    { buffer: input.buffer, index: 0 },
    { buffer: gradOutput.buffer, index: 1 },
    { buffer: gradWeight.buffer, index: 2 },
  ], { x: kW, y: kH, z: outC * groupSize },
  { x: Math.min(kW, 8), y: Math.min(kH, 8), z: 1 },
  { data: params, index: 3 })

  return gradWeight
}

// Backward: gradient w.r.t. bias
function conv2dBackwardBias(gradOutput) {
  const [batch, outC, outH, outW] = gradOutput.shape
  const gradBias = T.zeros([outC], gradOutput.dtype)

  // Reuse conv2d params — only need batch, outC, outH, outW, spatial = outH * outW
  const params = new Uint32Array(16)
  params[0] = batch
  params[4] = outC
  params[5] = outH
  params[6] = outW

  run('conv2d_backward_bias', [
    { buffer: gradOutput.buffer, index: 0 },
    { buffer: gradBias.buffer, index: 1 },
  ], { x: outC }, { x: Math.min(outC, 256) },
  { data: params, index: 2 })

  return gradBias
}

export {
  conv2dForward, conv2dBackwardInput, conv2dBackwardWeight, conv2dBackwardBias,
  convOutputSize, conv2dParams,
}
