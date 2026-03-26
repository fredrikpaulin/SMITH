// smith/src/ops/conv1d.js
// GPU im2col + GEMM for 1D convolution.
// Input: [C_in, length], weight: [C_out, C_in, kernelSize], bias: [C_out] or null.
// Mirrors the 2D im2col path (conv2d_im2col.js) but for a single spatial dimension.

import * as T from '../tensor.js'
import { run } from '../dispatch.js'
import { matmul2d } from './matmul.js'

function conv1dOutputSize(length, kernelSize, stride = 1, padding = 0) {
  return Math.floor((length + 2 * padding - kernelSize) / stride) + 1
}

// Build Conv1dParams struct matching the shader
function conv1dParams(cIn, length, outLen, kernelSize, stride, padding) {
  const params = new Uint32Array(6)
  params[0] = cIn
  params[1] = length
  params[2] = outLen
  params[3] = kernelSize
  params[4] = stride
  params[5] = padding
  return params
}

// im2col: input [C_in, length] → cols [C_in * kernelSize, outLen]
function im2col1d(input, kernelSize, stride, padding) {
  const [cIn, length] = input.shape
  const outLen = conv1dOutputSize(length, kernelSize, stride, padding)
  const colRows = cIn * kernelSize
  const cols = T.create([colRows, outLen], input.dtype)
  const params = conv1dParams(cIn, length, outLen, kernelSize, stride, padding)

  run('im2col_1d_forward', [
    { buffer: input.buffer, index: 0 },
    { buffer: cols.buffer, index: 1 },
  ], { x: outLen, y: colRows },
  { x: Math.min(outLen, 16), y: Math.min(colRows, 16) },
  { data: params, index: 2 })

  return { cols, outLen, colRows }
}

// col2im: cols [C_in * kernelSize, outLen] → gradInput [C_in, length]
function col2im1d(cols, cIn, length, kernelSize, stride, padding) {
  const outLen = conv1dOutputSize(length, kernelSize, stride, padding)
  const gradInput = T.zeros([cIn, length], cols.dtype)
  const params = conv1dParams(cIn, length, outLen, kernelSize, stride, padding)

  run('col2im_1d_backward', [
    { buffer: cols.buffer, index: 0 },
    { buffer: gradInput.buffer, index: 1 },
  ], { x: length, y: cIn },
  { x: Math.min(length, 16), y: Math.min(cIn, 16) },
  { data: params, index: 2 })

  return gradInput
}

// Forward: im2col + GEMM
// input [C_in, length], weight [C_out, C_in, kernelSize], bias [C_out] or null
// Returns { output: [C_out, outLen], patches: tensor } — patches retained for backward
function conv1dForward(input, weight, bias, stride, padding) {
  const [cOut, cIn, kernelSize] = weight.shape

  const { cols, outLen } = im2col1d(input, kernelSize, stride, padding)

  // weight: [C_out, C_in, K] → [C_out, C_in * K]
  const wFlat = T.create([cOut, cIn * kernelSize], weight.dtype)
  wFlat.data.set(weight.data)

  // GEMM: [C_out, C_in*K] × [C_in*K, outLen] = [C_out, outLen]
  const out = matmul2d(wFlat, cols)

  // Add bias: [C_out] broadcast over outLen
  if (bias) {
    for (let oc = 0; oc < cOut; oc++) {
      const b = bias.data[oc]
      const off = oc * outLen
      for (let t = 0; t < outLen; t++) {
        out.data[off + t] += b
      }
    }
  }

  return { output: out, patches: cols }
}

// Backward input gradient: W^T @ grad → col2im
// grad [C_out, outLen], weight [C_out, C_in, kernelSize] → gradInput [C_in, length]
function conv1dBackwardInput(grad, weight, inputLength, stride, padding) {
  const [cOut, cIn, kernelSize] = weight.shape

  // weight flat: [C_out, C_in*K] → transpose to [C_in*K, C_out]
  const colRows = cIn * kernelSize
  const wFlat = T.create([cOut, colRows], weight.dtype)
  wFlat.data.set(weight.data)
  const wT = T.create([colRows, cOut], weight.dtype)
  for (let r = 0; r < cOut; r++) {
    for (let c = 0; c < colRows; c++) {
      wT.data[c * cOut + r] = wFlat.data[r * colRows + c]
    }
  }

  // [C_in*K, C_out] × [C_out, outLen] = [C_in*K, outLen]
  const dCols = matmul2d(wT, grad)

  return col2im1d(dCols, cIn, inputLength, kernelSize, stride, padding)
}

// Backward weight gradient: grad @ patches^T
// grad [C_out, outLen], patches [C_in*K, outLen] → gradWeight [C_out, C_in, kernelSize]
function conv1dBackwardWeight(grad, patches, weightShape) {
  const [cOut, cIn, kernelSize] = weightShape
  const colRows = cIn * kernelSize
  const outLen = patches.shape[1]

  // Transpose patches: [C_in*K, outLen] → [outLen, C_in*K]
  const patchesT = T.create([outLen, colRows], patches.dtype)
  for (let r = 0; r < colRows; r++) {
    for (let c = 0; c < outLen; c++) {
      patchesT.data[c * colRows + r] = patches.data[r * outLen + c]
    }
  }

  // [C_out, outLen] × [outLen, C_in*K] = [C_out, C_in*K]
  const dWFlat = matmul2d(grad, patchesT)

  // Reshape to [C_out, C_in, kernelSize]
  const dW = T.create(weightShape, grad.dtype)
  dW.data.set(dWFlat.data)
  return dW
}

export {
  conv1dOutputSize,
  im2col1d, col2im1d,
  conv1dForward, conv1dBackwardInput, conv1dBackwardWeight,
}
