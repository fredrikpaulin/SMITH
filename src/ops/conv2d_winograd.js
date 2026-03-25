// smith/src/ops/conv2d_winograd.js
// Winograd F(2x2, 3x3) convolution dispatch.
// Pre-transforms 3x3 filters with G * g * G^T on CPU (once).
// Falls back to direct conv for non-3x3 / non-stride-1 kernels.

import * as T from '../tensor.js'
import { run } from '../dispatch.js'
import { conv2dForward, conv2dBackwardInput, conv2dBackwardWeight, conv2dBackwardBias, convOutputSize } from './conv2d.js'

// G matrix for filter transform: G * g * G^T
// G (4x3): [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
const G = [
  1,    0,    0,
  0.5,  0.5,  0.5,
  0.5, -0.5,  0.5,
  0,    0,    1,
]

// Pre-transform a 3x3 filter: G * g * G^T → 4x4
// g: 9 floats (3x3), output: 16 floats (4x4)
function transformFilter3x3(g) {
  // tmp = G * g  (4x3 * 3x3 = 4x3)
  const tmp = new Float32Array(12)
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 3; c++) {
      let s = 0
      for (let k = 0; k < 3; k++) s += G[r * 3 + k] * g[k * 3 + c]
      tmp[r * 3 + c] = s
    }
  }
  // out = tmp * G^T  (4x3 * 3x4 = 4x4)
  const out = new Float32Array(16)
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0
      for (let k = 0; k < 3; k++) s += tmp[r * 3 + k] * G[c * 3 + k]  // G^T[k,c] = G[c,k]
      out[r * 4 + c] = s
    }
  }
  return out
}

// Pre-transform all filters: [outC, inC/groups, 3, 3] → [outC, inC/groups, 4, 4]
// Computed once on CPU when the weight is first used.
function transformWeights(weight) {
  const [outC, groupSize, kH, kW] = weight.shape
  if (kH !== 3 || kW !== 3) throw new Error('Winograd requires 3x3 kernels')

  const total = outC * groupSize
  const out = T.create([outC, groupSize, 4, 4], weight.dtype)

  for (let i = 0; i < total; i++) {
    const srcOff = i * 9
    const dstOff = i * 16
    const g = weight.data.subarray(srcOff, srcOff + 9)
    const t = transformFilter3x3(g)
    out.data.set(t, dstOff)
  }
  return out
}

// Check if Winograd path is applicable
function canUseWinograd(weight, opts) {
  const [, , kH, kW] = weight.shape
  const stride = opts.stride ?? 1
  const dilation = opts.dilation ?? 1
  const [sH, sW] = Array.isArray(stride) ? stride : [stride, stride]
  const [dH, dW] = Array.isArray(dilation) ? dilation : [dilation, dilation]
  const groups = opts.groups ?? 1
  return kH === 3 && kW === 3 && sH === 1 && sW === 1 && dH === 1 && dW === 1 && groups === 1
}

// Build WinogradParams struct
function winogradParams(batch, inC, inH, inW, outC, outH, outW, tileH, tileW, padH, padW, groups) {
  const params = new Uint32Array(12)
  params[0]  = batch
  params[1]  = inC
  params[2]  = inH
  params[3]  = inW
  params[4]  = outC
  params[5]  = outH
  params[6]  = outW
  params[7]  = tileH
  params[8]  = tileW
  params[9]  = padH
  params[10] = padW
  params[11] = groups
  return params
}

// Forward: Winograd convolution
function winogradForward(input, weight, bias, opts = {}) {
  const { padding = 0, groups = 1 } = opts
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]
  const [batch, inC, inH, inW] = input.shape
  const [outC] = weight.shape

  const outH = convOutputSize(inH, 3, 1, padH, 1)
  const outW = convOutputSize(inW, 3, 1, padW, 1)
  const tileH = Math.ceil(outH / 2)
  const tileW = Math.ceil(outW / 2)

  const out = T.create([batch, outC, outH, outW], input.dtype)

  // Pre-transform weights
  const transW = transformWeights(weight)

  const params = winogradParams(batch, inC, inH, inW, outC, outH, outW, tileH, tileW, padH, padW, groups)

  if (bias) {
    run('conv2d_winograd_forward', [
      { buffer: input.buffer, index: 0 },
      { buffer: transW.buffer, index: 1 },
      { buffer: bias.buffer, index: 2 },
      { buffer: out.buffer, index: 3 },
    ], { x: tileW, y: tileH, z: batch * outC },
    { x: Math.min(tileW, 16), y: Math.min(tileH, 16), z: 1 },
    { data: params, index: 4 })
  } else {
    run('conv2d_winograd_forward_no_bias', [
      { buffer: input.buffer, index: 0 },
      { buffer: transW.buffer, index: 1 },
      { buffer: out.buffer, index: 2 },
    ], { x: tileW, y: tileH, z: batch * outC },
    { x: Math.min(tileW, 16), y: Math.min(tileH, 16), z: 1 },
    { data: params, index: 3 })
  }

  return out
}

// Backward: Winograd gradient w.r.t. input
function winogradBackwardInput(gradOutput, weight, inputShape, opts = {}) {
  const { padding = 0, groups = 1 } = opts
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]
  const [batch, inC, inH, inW] = inputShape
  const [outC] = weight.shape
  const outH = gradOutput.shape[2]
  const outW = gradOutput.shape[3]
  const tileH = Math.ceil(outH / 2)
  const tileW = Math.ceil(outW / 2)

  const gradInput = T.zeros(inputShape, gradOutput.dtype)
  const transW = transformWeights(weight)

  const params = winogradParams(batch, inC, inH, inW, outC, outH, outW, tileH, tileW, padH, padW, groups)

  run('conv2d_winograd_backward_input', [
    { buffer: gradOutput.buffer, index: 0 },
    { buffer: transW.buffer, index: 1 },
    { buffer: gradInput.buffer, index: 2 },
  ], { x: tileW, y: tileH, z: batch * inC },
  { x: Math.min(tileW, 16), y: Math.min(tileH, 16), z: 1 },
  { data: params, index: 3 })

  return gradInput
}

export {
  transformWeights, transformFilter3x3, canUseWinograd,
  winogradForward, winogradBackwardInput,
}
