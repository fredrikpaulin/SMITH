// smith/tests/conv.test.js
// Tests for Phase 10: Conv2d, Pool2d, BatchNorm

import { test, expect } from 'bun:test'
import * as T from '../src/tensor.js'
import {
  conv2dForward, conv2dBackwardInput, conv2dBackwardWeight, conv2dBackwardBias,
  convOutputSize, conv2dParams,
} from '../src/ops/conv2d.js'
import {
  maxPool2dForward, maxPool2dBackward,
  avgPool2dForward, avgPool2dBackward,
  poolOutputSize,
} from '../src/ops/pool2d.js'
import {
  createBatchNorm, batchnormForward, batchnormInference, batchnormBackward,
} from '../src/ops/batchnorm.js'
import * as A from '../src/autograd.js'

// --- Helpers ---

// Read tensor data as flat Float32Array (avoids nested toArray)
function flat(t) {
  const arr = new Float32Array(t.size)
  for (let i = 0; i < t.size; i++) arr[i] = t.data[i]
  return arr
}

function cpuConv2d(input, weight, bias, opts = {}) {
  const { stride = 1, padding = 0, dilation = 1, groups = 1 } = opts
  const [sH, sW] = Array.isArray(stride) ? stride : [stride, stride]
  const [pH, pW] = Array.isArray(padding) ? padding : [padding, padding]
  const [dH, dW] = Array.isArray(dilation) ? dilation : [dilation, dilation]

  const [batch, inC, inH, inW] = input.shape
  const [outC, groupInC, kH, kW] = weight.shape
  const outH = Math.floor((inH + 2 * pH - dH * (kH - 1) - 1) / sH) + 1
  const outW = Math.floor((inW + 2 * pW - dW * (kW - 1) - 1) / sW) + 1
  const groupOutC = outC / groups

  const out = new Float32Array(batch * outC * outH * outW)
  for (let n = 0; n < batch; n++) {
    for (let g = 0; g < groups; g++) {
      for (let oc = 0; oc < groupOutC; oc++) {
        const outCIdx = g * groupOutC + oc
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let sum = bias ? bias.data[outCIdx] : 0
            for (let ic = 0; ic < groupInC; ic++) {
              const inCIdx = g * groupInC + ic
              for (let kh = 0; kh < kH; kh++) {
                for (let kw = 0; kw < kW; kw++) {
                  const ih = oh * sH + kh * dH - pH
                  const iw = ow * sW + kw * dW - pW
                  if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    const inIdx = n * inC * inH * inW + inCIdx * inH * inW + ih * inW + iw
                    const wIdx = outCIdx * groupInC * kH * kW + ic * kH * kW + kh * kW + kw
                    sum += input.data[inIdx] * weight.data[wIdx]
                  }
                }
              }
            }
            const outIdx = n * outC * outH * outW + outCIdx * outH * outW + oh * outW + ow
            out[outIdx] = sum
          }
        }
      }
    }
  }
  return out
}

function cpuMaxPool2d(input, opts = {}) {
  const { kernelSize = 2, stride, padding = 0 } = opts
  const [kH, kW] = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize]
  const [sH, sW] = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : [kH, kW]
  const [pH, pW] = Array.isArray(padding) ? padding : [padding, padding]

  const [batch, channels, inH, inW] = input.shape
  const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1
  const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1

  const out = new Float32Array(batch * channels * outH * outW)
  for (let n = 0; n < batch; n++) {
    for (let c = 0; c < channels; c++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let maxVal = -Infinity
          for (let kh = 0; kh < kH; kh++) {
            for (let kw = 0; kw < kW; kw++) {
              const ih = oh * sH + kh - pH
              const iw = ow * sW + kw - pW
              if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                const idx = n * channels * inH * inW + c * inH * inW + ih * inW + iw
                if (input.data[idx] > maxVal) maxVal = input.data[idx]
              }
            }
          }
          const outIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow
          out[outIdx] = maxVal
        }
      }
    }
  }
  return out
}

function cpuAvgPool2d(input, opts = {}) {
  const { kernelSize = 2, stride, padding = 0 } = opts
  const [kH, kW] = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize]
  const [sH, sW] = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : [kH, kW]
  const [pH, pW] = Array.isArray(padding) ? padding : [padding, padding]

  const [batch, channels, inH, inW] = input.shape
  const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1
  const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1

  const out = new Float32Array(batch * channels * outH * outW)
  for (let n = 0; n < batch; n++) {
    for (let c = 0; c < channels; c++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let sum = 0, count = 0
          for (let kh = 0; kh < kH; kh++) {
            for (let kw = 0; kw < kW; kw++) {
              const ih = oh * sH + kh - pH
              const iw = ow * sW + kw - pW
              if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                const idx = n * channels * inH * inW + c * inH * inW + ih * inW + iw
                sum += input.data[idx]
                count++
              }
            }
          }
          const outIdx = n * channels * outH * outW + c * outH * outW + oh * outW + ow
          out[outIdx] = count > 0 ? sum / count : 0
        }
      }
    }
  }
  return out
}

// --- Conv2d tests ---

test('convOutputSize basic', () => {
  expect(convOutputSize(8, 3, 1, 0, 1)).toBe(6)
  expect(convOutputSize(8, 3, 1, 1, 1)).toBe(8) // same padding
  expect(convOutputSize(8, 3, 2, 0, 1)).toBe(3)
  expect(convOutputSize(8, 3, 1, 0, 2)).toBe(4) // dilation
})

test('poolOutputSize basic', () => {
  expect(poolOutputSize(8, 2, 2, 0)).toBe(4)
  expect(poolOutputSize(8, 3, 1, 1)).toBe(8)
  expect(poolOutputSize(7, 2, 2, 0)).toBe(3)
})

test('conv2d forward 1x1x4x4 → 1 filter 3x3 no padding', () => {
  const input = T.tensor(
    Array.from({ length: 16 }, (_, i) => i + 1),
    [1, 1, 4, 4]
  )
  const weight = T.ones([1, 1, 3, 3])
  const out = conv2dForward(input, weight, null)

  const cpuOut = cpuConv2d(input, weight, null)
  expect(out.shape).toEqual([1, 1, 2, 2])
  const arr = flat(out)
  for (let i = 0; i < cpuOut.length; i++) {
    expect(Math.abs(arr[i] - cpuOut[i])).toBeLessThan(1e-4)
  }
})

test('conv2d forward with bias', () => {
  const input = T.ones([1, 1, 4, 4])
  const weight = T.ones([2, 1, 3, 3])
  const bias = T.tensor([10, 20], [2])
  const out = conv2dForward(input, weight, bias)

  expect(out.shape).toEqual([1, 2, 2, 2])
  const arr = flat(out)
  // Each output = 9 (sum of 3x3 ones) + bias
  for (let i = 0; i < 4; i++) expect(Math.abs(arr[i] - 19)).toBeLessThan(1e-4) // channel 0: 9+10
  for (let i = 4; i < 8; i++) expect(Math.abs(arr[i] - 29)).toBeLessThan(1e-4) // channel 1: 9+20
})

test('conv2d forward with padding preserves spatial dims', () => {
  const input = T.rand([1, 1, 4, 4])
  const weight = T.rand([1, 1, 3, 3])
  const out = conv2dForward(input, weight, null, { padding: 1 })
  expect(out.shape).toEqual([1, 1, 4, 4])
})

test('conv2d forward with stride 2', () => {
  const input = T.rand([1, 1, 8, 8])
  const weight = T.rand([1, 1, 3, 3])
  const out = conv2dForward(input, weight, null, { stride: 2, padding: 1 })
  expect(out.shape).toEqual([1, 1, 4, 4])
})

test('conv2d forward matches CPU reference (random)', () => {
  const input = T.rand([2, 3, 6, 6])
  const weight = T.rand([4, 3, 3, 3])
  const bias = T.rand([4])
  const opts = { padding: 1 }
  const out = conv2dForward(input, weight, bias, opts)
  const cpuOut = cpuConv2d(input, weight, bias, opts)

  expect(out.shape).toEqual([2, 4, 6, 6])
  const arr = flat(out)
  for (let i = 0; i < cpuOut.length; i++) {
    expect(Math.abs(arr[i] - cpuOut[i])).toBeLessThan(1e-3)
  }
})

test('conv2d backward produces correct shapes', () => {
  const inputShape = [1, 1, 4, 4]
  const weightShape = [2, 1, 3, 3]
  const input = T.rand(inputShape)
  const weight = T.rand(weightShape)
  const out = conv2dForward(input, weight, null)
  const gradOut = T.ones(out.shape)

  const gradInput = conv2dBackwardInput(gradOut, weight, inputShape)
  const gradWeight = conv2dBackwardWeight(input, gradOut, weightShape)
  const gradBias = conv2dBackwardBias(gradOut)

  expect(gradInput.shape).toEqual(inputShape)
  expect(gradWeight.shape).toEqual(weightShape)
  expect(gradBias.shape).toEqual([2])
})

test('conv2d backward bias sums correctly', () => {
  // gradBias[c] = sum over batch, H, W of gradOutput[:, c, :, :]
  const gradOut = T.ones([2, 3, 4, 4])
  const gradBias = conv2dBackwardBias(gradOut)
  const arr = T.toArray(gradBias)
  // Each channel: 2 * 4 * 4 = 32
  for (let c = 0; c < 3; c++) {
    expect(Math.abs(arr[c] - 32)).toBeLessThan(1e-4)
  }
})

// --- MaxPool2d tests ---

test('maxpool2d forward 2x2 basic', () => {
  const input = T.tensor(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    [1, 1, 4, 4]
  )
  const { out } = maxPool2dForward(input, { kernelSize: 2 })
  expect(out.shape).toEqual([1, 1, 2, 2])
  const arr = flat(out)
  expect(arr[0]).toBe(6)   // max(1,2,5,6)
  expect(arr[1]).toBe(8)   // max(3,4,7,8)
  expect(arr[2]).toBe(14)  // max(9,10,13,14)
  expect(arr[3]).toBe(16)  // max(11,12,15,16)
})

test('maxpool2d forward matches CPU reference', () => {
  const input = T.rand([2, 3, 8, 8])
  const opts = { kernelSize: 2, stride: 2 }
  const { out } = maxPool2dForward(input, opts)
  const cpuOut = cpuMaxPool2d(input, opts)

  expect(out.shape).toEqual([2, 3, 4, 4])
  const arr = flat(out)
  for (let i = 0; i < cpuOut.length; i++) {
    expect(Math.abs(arr[i] - cpuOut[i])).toBeLessThan(1e-5)
  }
})

test('maxpool2d backward scatters to max positions', () => {
  const input = T.tensor(
    [1, 4, 2, 3, 8, 5, 7, 6, 9, 10, 11, 12, 16, 13, 14, 15],
    [1, 1, 4, 4]
  )
  const { out, indices } = maxPool2dForward(input, { kernelSize: 2 })
  const gradOut = T.tensor([1, 2, 3, 4], [1, 1, 2, 2])
  const gradInput = maxPool2dBackward(gradOut, indices, [1, 1, 4, 4])

  expect(gradInput.shape).toEqual([1, 1, 4, 4])
  const arr = flat(gradInput)
  // Gradient should be at max positions only
  // Max positions: (1,0)=8→grad 1, (0,3)=3... check GPU max positions
  // input: [[1,4,2,3],[8,5,7,6],[9,10,11,12],[16,13,14,15]]
  // Pool 0,0: max(1,4,8,5)=8 at pos(1,0) → idx 4
  // Pool 0,1: max(2,3,7,6)=7 at pos(1,2) → idx 6
  // Pool 1,0: max(9,10,16,13)=16 at pos(3,0) → idx 12
  // Pool 1,1: max(11,12,14,15)=15 at pos(3,3) → idx 15
  expect(arr[4]).toBe(1)    // position of max in pool(0,0)
  expect(arr[6]).toBe(2)    // position of max in pool(0,1)
  expect(arr[12]).toBe(3)   // position of max in pool(1,0)
  expect(arr[15]).toBe(4)   // position of max in pool(1,1)
  // Non-max positions should be 0
  expect(arr[0]).toBe(0)
  expect(arr[1]).toBe(0)
})

test('maxpool2d with stride 1 and padding', () => {
  const input = T.rand([1, 1, 4, 4])
  const { out } = maxPool2dForward(input, { kernelSize: 3, stride: 1, padding: 1 })
  expect(out.shape).toEqual([1, 1, 4, 4])
})

// --- AvgPool2d tests ---

test('avgpool2d forward 2x2 basic', () => {
  const input = T.tensor(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    [1, 1, 4, 4]
  )
  const out = avgPool2dForward(input, { kernelSize: 2 })
  expect(out.shape).toEqual([1, 1, 2, 2])
  const arr = flat(out)
  expect(Math.abs(arr[0] - 3.5)).toBeLessThan(1e-5)  // avg(1,2,5,6) = 3.5
  expect(Math.abs(arr[1] - 5.5)).toBeLessThan(1e-5)  // avg(3,4,7,8) = 5.5
  expect(Math.abs(arr[2] - 11.5)).toBeLessThan(1e-5) // avg(9,10,13,14) = 11.5
  expect(Math.abs(arr[3] - 13.5)).toBeLessThan(1e-5) // avg(11,12,15,16) = 13.5
})

test('avgpool2d forward matches CPU reference', () => {
  const input = T.rand([2, 3, 8, 8])
  const opts = { kernelSize: 2, stride: 2 }
  const out = avgPool2dForward(input, opts)
  const cpuOut = cpuAvgPool2d(input, opts)

  expect(out.shape).toEqual([2, 3, 4, 4])
  const arr = flat(out)
  for (let i = 0; i < cpuOut.length; i++) {
    expect(Math.abs(arr[i] - cpuOut[i])).toBeLessThan(1e-4)
  }
})

test('avgpool2d backward shape', () => {
  const input = T.rand([1, 1, 4, 4])
  const out = avgPool2dForward(input, { kernelSize: 2 })
  const gradOut = T.ones(out.shape)
  const gradInput = avgPool2dBackward(gradOut, [1, 1, 4, 4], { kernelSize: 2 })
  expect(gradInput.shape).toEqual([1, 1, 4, 4])
})

test('avgpool2d backward distributes uniformly', () => {
  // With 2x2 non-overlapping pool, each input gets 1/4 of the gradient
  const gradOut = T.tensor([4], [1, 1, 1, 1])
  const gradInput = avgPool2dBackward(gradOut, [1, 1, 2, 2], { kernelSize: 2 })
  const arr = flat(gradInput)
  for (let i = 0; i < 4; i++) {
    expect(Math.abs(arr[i] - 1)).toBeLessThan(1e-5)  // 4 / 4 = 1
  }
})

// --- BatchNorm tests ---

test('createBatchNorm initializes correctly', () => {
  const bn = createBatchNorm(64)
  expect(bn.channels).toBe(64)
  expect(bn.gamma.shape).toEqual([64])
  expect(bn.beta.shape).toEqual([64])
  expect(bn.runningMean.shape).toEqual([64])
  expect(bn.runningVar.shape).toEqual([64])
  expect(bn.eps).toBe(1e-5)
  expect(bn.momentum).toBe(0.1)
  // gamma should be ones
  const g = T.toArray(bn.gamma)
  expect(Math.abs(g[0] - 1)).toBeLessThan(1e-6)
  // beta should be zeros
  const b = T.toArray(bn.beta)
  expect(Math.abs(b[0])).toBeLessThan(1e-6)
})

test('batchnorm forward normalizes to ~zero mean, ~unit var', () => {
  const bn = createBatchNorm(2)
  // Input: [batch=2, channels=2, H=2, W=2]
  const input = T.tensor([
    // batch 0, channel 0
    1, 2, 3, 4,
    // batch 0, channel 1
    10, 20, 30, 40,
    // batch 1, channel 0
    5, 6, 7, 8,
    // batch 1, channel 1
    50, 60, 70, 80,
  ], [2, 2, 2, 2])

  const { out, savedMean, savedInvStd } = batchnormForward(input, bn)
  expect(out.shape).toEqual([2, 2, 2, 2])
  expect(savedMean.shape).toEqual([2])
  expect(savedInvStd.shape).toEqual([2])

  // Check normalized output: mean ~ 0, var ~ 1 per channel
  const arr = flat(out)
  for (let c = 0; c < 2; c++) {
    let sum = 0, sumSq = 0
    const count = 2 * 4  // batch * spatial
    for (let n = 0; n < 2; n++) {
      for (let s = 0; s < 4; s++) {
        const idx = n * 2 * 4 + c * 4 + s
        sum += arr[idx]
        sumSq += arr[idx] * arr[idx]
      }
    }
    const mean = sum / count
    const variance = sumSq / count - mean * mean
    expect(Math.abs(mean)).toBeLessThan(1e-4)
    expect(Math.abs(variance - 1)).toBeLessThan(0.1) // approximate unit variance
  }
})

test('batchnorm forward updates running stats', () => {
  const bn = createBatchNorm(1)
  const input = T.tensor([1, 2, 3, 4], [1, 1, 2, 2])
  batchnormForward(input, bn)

  const rm = T.toArray(bn.runningMean)
  const rv = T.toArray(bn.runningVar)
  // running_mean = 0.9 * 0 + 0.1 * 2.5 = 0.25
  expect(Math.abs(rm[0] - 0.25)).toBeLessThan(1e-4)
  // running_var = 0.9 * 1.0 + 0.1 * var(1,2,3,4) = 0.9 + 0.1*1.25 = 1.025
  expect(Math.abs(rv[0] - 1.025)).toBeLessThan(1e-3)
})

test('batchnorm inference uses running stats', () => {
  const bn = createBatchNorm(1)
  // Set running stats manually
  bn.runningMean.data[0] = 5.0
  bn.runningVar.data[0] = 4.0
  bn.gamma.data[0] = 2.0
  bn.beta.data[0] = 1.0

  const input = T.tensor([5, 7, 3, 9], [1, 1, 2, 2])
  const out = batchnormInference(input, bn)
  const arr = flat(out)

  // y = gamma * (x - mean) / sqrt(var + eps) + beta
  // = 2 * (x - 5) / sqrt(4 + 1e-5) + 1
  const invStd = 1 / Math.sqrt(4 + 1e-5)
  const expected = [5, 7, 3, 9].map(x => 2 * (x - 5) * invStd + 1)
  for (let i = 0; i < 4; i++) {
    expect(Math.abs(arr[i] - expected[i])).toBeLessThan(1e-3)
  }
})

test('batchnorm backward produces correct shapes', () => {
  const bn = createBatchNorm(2)
  const input = T.rand([2, 2, 3, 3])
  const { out, savedMean, savedInvStd } = batchnormForward(input, bn)
  const gradOut = T.ones(out.shape)

  const { gradInput, gradGamma, gradBeta } = batchnormBackward(gradOut, input, savedMean, savedInvStd, bn)
  expect(gradInput.shape).toEqual([2, 2, 3, 3])
  expect(gradGamma.shape).toEqual([2])
  expect(gradBeta.shape).toEqual([2])
})

test('batchnorm backward gradBeta = sum of gradOutput per channel', () => {
  const bn = createBatchNorm(2)
  const input = T.rand([2, 2, 2, 2])
  const { savedMean, savedInvStd } = batchnormForward(input, bn)
  const gradOut = T.ones([2, 2, 2, 2])

  const { gradBeta } = batchnormBackward(gradOut, input, savedMean, savedInvStd, bn)
  const arr = T.toArray(gradBeta)
  // gradBeta[c] = sum of all gradOutput for channel c = batch * spatial = 2 * 4 = 8
  for (let c = 0; c < 2; c++) {
    expect(Math.abs(arr[c] - 8)).toBeLessThan(1e-3)
  }
})

// --- Autograd integration tests ---

test('autograd conv2d forward produces correct shape', () => {
  const input = A.variable(T.rand([1, 1, 4, 4]))
  const weight = A.variable(T.rand([2, 1, 3, 3]), { requiresGrad: true })
  const bias = A.variable(T.rand([2]), { requiresGrad: true })
  const out = A.conv2d(input, weight, bias)
  expect(out.data.shape).toEqual([1, 2, 2, 2])
})

test('autograd conv2d backward computes gradients', () => {
  const input = A.variable(T.rand([1, 1, 4, 4]), { requiresGrad: true })
  const weight = A.variable(T.rand([2, 1, 3, 3]), { requiresGrad: true })
  const bias = A.variable(T.rand([2]), { requiresGrad: true })
  const out = A.conv2d(input, weight, bias)

  // Sum output to get scalar loss
  const loss = A.sum(out)
  A.backward(loss)

  expect(input.grad).not.toBeNull()
  expect(weight.grad).not.toBeNull()
  expect(bias.grad).not.toBeNull()
  expect(input.grad.shape).toEqual([1, 1, 4, 4])
  expect(weight.grad.shape).toEqual([2, 1, 3, 3])
  expect(bias.grad.shape).toEqual([2])
})

test('autograd maxPool2d forward and backward', () => {
  const input = A.variable(T.rand([1, 1, 4, 4]), { requiresGrad: true })
  const out = A.maxPool2d(input, { kernelSize: 2 })
  expect(out.data.shape).toEqual([1, 1, 2, 2])

  const loss = A.sum(out)
  A.backward(loss)
  expect(input.grad).not.toBeNull()
  expect(input.grad.shape).toEqual([1, 1, 4, 4])
})

test('autograd avgPool2d forward and backward', () => {
  const input = A.variable(T.rand([1, 1, 4, 4]), { requiresGrad: true })
  const out = A.avgPool2d(input, { kernelSize: 2 })
  expect(out.data.shape).toEqual([1, 1, 2, 2])

  const loss = A.sum(out)
  A.backward(loss)
  expect(input.grad).not.toBeNull()
  expect(input.grad.shape).toEqual([1, 1, 4, 4])
})

test('autograd batchnorm forward and backward', () => {
  const bn = createBatchNorm(2)
  const input = A.variable(T.rand([2, 2, 3, 3]), { requiresGrad: true })
  const out = A.batchnorm(input, bn, true)
  expect(out.data.shape).toEqual([2, 2, 3, 3])

  const loss = A.sum(out)
  A.backward(loss)
  expect(input.grad).not.toBeNull()
  expect(input.grad.shape).toEqual([2, 2, 3, 3])
})

test('autograd conv2d + relu + maxpool pipeline', () => {
  const input = A.variable(T.rand([1, 1, 8, 8]), { requiresGrad: true })
  const weight = A.variable(T.rand([4, 1, 3, 3]), { requiresGrad: true })

  const convOut = A.conv2d(input, weight, null, { padding: 1 })
  const reluOut = A.relu(convOut)
  const poolOut = A.maxPool2d(reluOut, { kernelSize: 2 })

  expect(poolOut.data.shape).toEqual([1, 4, 4, 4])

  const loss = A.sum(poolOut)
  A.backward(loss)

  expect(input.grad).not.toBeNull()
  expect(weight.grad).not.toBeNull()
  expect(input.grad.shape).toEqual([1, 1, 8, 8])
  expect(weight.grad.shape).toEqual([4, 1, 3, 3])
})

test('autograd batchnorm inference (no grad)', () => {
  const bn = createBatchNorm(2)
  // Run forward to set running stats
  const trainInput = T.rand([2, 2, 3, 3])
  batchnormForward(trainInput, bn)

  const input = A.variable(T.rand([2, 2, 3, 3]))
  const out = A.batchnorm(input, bn, false)
  expect(out.data.shape).toEqual([2, 2, 3, 3])
  expect(out._backward).toBeNull()
})

// --- Numerical gradient check for conv2d ---

test('conv2d numerical gradient check (weight)', () => {
  const eps = 1e-3
  const inputData = T.rand([1, 1, 4, 4])
  const weightData = T.rand([1, 1, 3, 3])

  // Analytic gradient
  const input = A.variable(inputData, { requiresGrad: false })
  const weight = A.variable(weightData, { requiresGrad: true })
  const out = A.conv2d(input, weight, null)
  const loss = A.sum(out)
  A.backward(loss)

  // Numerical gradient for a few weight elements
  const wArr = flat(weightData)
  const analyticGrad = flat(weight.grad)

  for (let idx = 0; idx < Math.min(4, wArr.length); idx++) {
    // f(w + eps)
    const wPlus = T.tensor([...wArr], weightData.shape)
    wPlus.data[idx] += eps
    const outPlus = conv2dForward(inputData, wPlus, null)
    let sumPlus = 0
    const fPlus = flat(outPlus)
    for (let i = 0; i < fPlus.length; i++) sumPlus += fPlus[i]

    // f(w - eps)
    const wMinus = T.tensor([...wArr], weightData.shape)
    wMinus.data[idx] -= eps
    const outMinus = conv2dForward(inputData, wMinus, null)
    let sumMinus = 0
    const fMinus = flat(outMinus)
    for (let i = 0; i < fMinus.length; i++) sumMinus += fMinus[i]

    const numGrad = (sumPlus - sumMinus) / (2 * eps)
    expect(Math.abs(analyticGrad[idx] - numGrad)).toBeLessThan(0.05)
  }
})
