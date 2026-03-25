// smith/src/ops/conv2d_im2col.js
// im2col + GEMM convolution path.
// Rearranges input patches into a column matrix, then uses the existing tiled
// matmul shader. Trades memory for compute efficiency on larger kernels.

import * as T from '../tensor.js'
import { run, k } from '../dispatch.js'
import { matmul2d } from './matmul.js'
import { convOutputSize } from './conv2d.js'

// Build Im2colParams struct matching the shader
function im2colParams(batch, inC, inH, inW, outH, outW, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW) {
  const params = new Uint32Array(14)
  params[0]  = batch
  params[1]  = inC
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
  params[12] = dilationH
  params[13] = dilationW
  return params
}

// im2col: input [batch, inC, inH, inW] → cols [batch, inC*kH*kW, outH*outW]
function im2col(input, opts) {
  const { stride = 1, padding = 0, dilation = 1 } = opts
  const [strideH, strideW] = Array.isArray(stride) ? stride : [stride, stride]
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]
  const [dilationH, dilationW] = Array.isArray(dilation) ? dilation : [dilation, dilation]
  const [batch, inC, inH, inW] = input.shape
  const kH = opts.kH, kW = opts.kW

  const outH = convOutputSize(inH, kH, strideH, padH, dilationH)
  const outW = convOutputSize(inW, kW, strideW, padW, dilationW)
  const colRows = inC * kH * kW
  const colCols = outH * outW

  const cols = T.create([batch, colRows, colCols], input.dtype)
  const params = im2colParams(batch, inC, inH, inW, outH, outW, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW)

  run('im2col_forward', [
    { buffer: input.buffer, index: 0 },
    { buffer: cols.buffer, index: 1 },
  ], { x: colCols, y: colRows, z: batch },
  { x: Math.min(colCols, 16), y: Math.min(colRows, 16), z: 1 },
  { data: params, index: 2 })

  return { cols, outH, outW, colRows, colCols }
}

// col2im: cols [batch, inC*kH*kW, outH*outW] → gradInput [batch, inC, inH, inW]
function col2im(cols, inputShape, opts) {
  const { stride = 1, padding = 0, dilation = 1 } = opts
  const [strideH, strideW] = Array.isArray(stride) ? stride : [stride, stride]
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]
  const [dilationH, dilationW] = Array.isArray(dilation) ? dilation : [dilation, dilation]
  const [batch, inC, inH, inW] = inputShape
  const kH = opts.kH, kW = opts.kW

  const outH = convOutputSize(inH, kH, strideH, padH, dilationH)
  const outW = convOutputSize(inW, kW, strideW, padW, dilationW)

  const gradInput = T.zeros(inputShape, cols.dtype)
  const params = im2colParams(batch, inC, inH, inW, outH, outW, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW)

  run('col2im_backward', [
    { buffer: cols.buffer, index: 0 },
    { buffer: gradInput.buffer, index: 1 },
  ], { x: inW, y: inH, z: batch * inC },
  { x: Math.min(inW, 16), y: Math.min(inH, 16), z: 1 },
  { data: params, index: 2 })

  return gradInput
}

// Forward: im2col + GEMM
// input [batch, inC, inH, inW], weight [outC, inC/groups, kH, kW], bias [outC]
function im2colForward(input, weight, bias, opts = {}) {
  const { groups = 1 } = opts
  const [batch, inC, inH, inW] = input.shape
  const [outC, groupInC, kH, kW] = weight.shape

  const im2colOpts = { ...opts, kH, kW }
  const { cols, outH, outW, colRows, colCols } = im2col(input, im2colOpts)

  // Reshape weight: [outC, groupInC*kH*kW] for matmul
  // weight is [outC, groupInC, kH, kW] → flatten last 3 dims
  const weightFlat = T.create([outC, groupInC * kH * kW], weight.dtype)
  weightFlat.data.set(weight.data)

  // For each batch: output = weight_flat @ cols[n]
  // weight_flat: [outC, colRows], cols[n]: [colRows, colCols]
  // result[n]: [outC, colCols] = [outC, outH*outW]
  const out = T.create([batch, outC, outH, outW], input.dtype)

  for (let n = 0; n < batch; n++) {
    // Extract cols for this batch sample
    const colsN = T.create([colRows, colCols], input.dtype)
    colsN.data.set(cols.data.subarray(n * colRows * colCols, (n + 1) * colRows * colCols))

    // GEMM: [outC, colRows] × [colRows, colCols] = [outC, colCols]
    const gemm = matmul2d(weightFlat, colsN)

    // Copy to output and add bias
    const outN = out.data.subarray(n * outC * outH * outW, (n + 1) * outC * outH * outW)
    outN.set(gemm.data)

    if (bias) {
      for (let oc = 0; oc < outC; oc++) {
        const b = bias.data[oc]
        const off = oc * outH * outW
        for (let s = 0; s < outH * outW; s++) {
          outN[off + s] += b
        }
      }
    }
  }

  return out
}

// Backward: input gradient via col2im
// gradOutput [batch, outC, outH, outW] → gradInput [batch, inC, inH, inW]
function im2colBackwardInput(gradOutput, weight, inputShape, opts = {}) {
  const [batch, outC, outH, outW] = gradOutput.shape
  const [, groupInC, kH, kW] = weight.shape
  const colRows = groupInC * kH * kW  // = inC * kH * kW for groups=1
  const colCols = outH * outW

  // weight^T @ gradOutput → cols for each batch
  // weight_flat: [outC, colRows], need [colRows, outC]
  const weightFlat = T.create([outC, colRows], weight.dtype)
  weightFlat.data.set(weight.data)

  // Transpose weight: [colRows, outC]
  const weightT = T.create([colRows, outC], weight.dtype)
  for (let r = 0; r < outC; r++) {
    for (let c = 0; c < colRows; c++) {
      weightT.data[c * outC + r] = weightFlat.data[r * colRows + c]
    }
  }

  // For each batch: cols[n] = weight^T @ gradOutput[n]
  const cols = T.create([batch, colRows, colCols], gradOutput.dtype)

  for (let n = 0; n < batch; n++) {
    const goN = T.create([outC, colCols], gradOutput.dtype)
    goN.data.set(gradOutput.data.subarray(n * outC * colCols, (n + 1) * outC * colCols))

    // [colRows, outC] × [outC, colCols] = [colRows, colCols]
    const gemm = matmul2d(weightT, goN)
    cols.data.set(gemm.data, n * colRows * colCols)
  }

  // col2im: scatter-add cols back to input layout
  const im2colOpts = { ...opts, kH, kW }
  return col2im(cols, inputShape, im2colOpts)
}

// Backward: weight gradient
// input [batch, inC, inH, inW], gradOutput [batch, outC, outH, outW]
// → gradWeight [outC, inC/groups, kH, kW]
function im2colBackwardWeight(input, gradOutput, weightShape, opts = {}) {
  const [batch, outC, outH, outW] = gradOutput.shape
  const [, groupInC, kH, kW] = weightShape
  const colRows = groupInC * kH * kW
  const colCols = outH * outW

  const im2colOpts = { ...opts, kH, kW }
  const { cols } = im2col(input, im2colOpts)

  // For each batch: gradW += gradOutput[n] @ cols[n]^T
  // gradOutput[n]: [outC, colCols], cols[n]^T: [colCols, colRows]
  // result: [outC, colRows]
  const gradWeight = T.zeros(weightShape, gradOutput.dtype)

  for (let n = 0; n < batch; n++) {
    const goN = T.create([outC, colCols], gradOutput.dtype)
    goN.data.set(gradOutput.data.subarray(n * outC * colCols, (n + 1) * outC * colCols))

    // Transpose cols[n]: [colRows, colCols] → [colCols, colRows]
    const colsN = T.create([colCols, colRows], input.dtype)
    const srcOff = n * colRows * colCols
    for (let r = 0; r < colRows; r++) {
      for (let c = 0; c < colCols; c++) {
        colsN.data[c * colRows + r] = cols.data[srcOff + r * colCols + c]
      }
    }

    // [outC, colCols] × [colCols, colRows] = [outC, colRows]
    const gemm = matmul2d(goN, colsN)

    // Accumulate
    for (let i = 0; i < gemm.size; i++) {
      gradWeight.data[i] += gemm.data[i]
    }
  }

  return gradWeight
}

// Check if im2col path should be used
function shouldUseIm2col(weight, opts) {
  const [, , kH, kW] = weight.shape
  const stride = opts.stride ?? 1
  const [sH, sW] = Array.isArray(stride) ? stride : [stride, stride]
  const dilation = opts.dilation ?? 1
  const [dH, dW] = Array.isArray(dilation) ? dilation : [dilation, dilation]
  // Use im2col for non-3x3 kernels (3x3 stride-1 uses Winograd)
  // or for strided 3x3 (Winograd doesn't handle stride)
  if (kH === 3 && kW === 3 && sH === 1 && sW === 1 && dH === 1 && dW === 1) return false
  return kH * kW > 1  // anything except 1x1 pointwise
}

export {
  im2col, col2im,
  im2colForward, im2colBackwardInput, im2colBackwardWeight,
  shouldUseIm2col,
}
