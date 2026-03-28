// smith/src/ops/conv1d_transpose.js
// GPU transposed 1D convolution (deconvolution / upsampling).
// Input: [C_in, length], weight: [C_in, C_out, kernelSize], bias: [C_out] or null.
// Output: [C_out, outLen] where outLen = (length - 1) * stride - 2 * padding + kernelSize

import * as T from '../tensor.js'
import { run, k } from '../dispatch.js'

function convTranspose1dOutputSize(length, kernelSize, stride = 1, padding = 0) {
  return (length - 1) * stride - 2 * padding + kernelSize
}

function convTranspose1dForward(input, weight, bias, stride = 1, padding = 0) {
  const [cIn, length] = input.shape
  const [wCIn, cOut, kernelSize] = weight.shape
  const outLen = convTranspose1dOutputSize(length, kernelSize, stride, padding)
  const output = T.create([cOut, outLen], input.dtype)

  const params = new Uint32Array([cIn, length, cOut, outLen, kernelSize, stride, padding])

  run(k('conv_transpose_1d_forward', input.dtype), [
    { buffer: input.buffer, index: 0 },
    { buffer: weight.buffer, index: 1 },
    { buffer: output.buffer, index: 2 },
  ], { x: outLen, y: cOut },
  { x: Math.min(outLen, 16), y: Math.min(cOut, 16) },
  { data: params, index: 3 })

  // Add bias: [C_out] broadcast over outLen
  if (bias) {
    for (let oc = 0; oc < cOut; oc++) {
      const b = bias.data[oc]
      const off = oc * outLen
      for (let t = 0; t < outLen; t++) {
        output.data[off + t] += b
      }
    }
  }

  return output
}

export { convTranspose1dOutputSize, convTranspose1dForward }
