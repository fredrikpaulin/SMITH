// tests/im2col.test.js
// Phase 14: im2col Convolution Path
// Tests im2col transform, forward equivalence with direct conv for 5x5/7x7,
// col2im backward, auto-dispatch, numerical gradient checks.

import { test, expect, describe } from 'bun:test'
import * as T from '../src/tensor.js'
import * as A from '../src/autograd.js'
import { conv2dForward, convOutputSize } from '../src/ops/conv2d.js'
import { im2col, col2im, im2colForward, im2colBackwardInput, im2colBackwardWeight, shouldUseIm2col } from '../src/ops/conv2d_im2col.js'

const flat = t => new Float32Array(t.data.buffer, t.data.byteOffset, t.data.length)

// CPU reference: direct 2D convolution (NCHW)
function conv2dCPU(input, weight, bias, opts = {}) {
  const { padding = 0, stride = 1, dilation = 1 } = opts
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]
  const [sH, sW] = Array.isArray(stride) ? stride : [stride, stride]
  const [dH, dW] = Array.isArray(dilation) ? dilation : [dilation, dilation]
  const [batch, inC, inH, inW] = input.shape
  const [outC, groupSize, kH, kW] = weight.shape
  const outH = convOutputSize(inH, kH, sH, padH, dH)
  const outW = convOutputSize(inW, kW, sW, padW, dW)
  const out = new Float32Array(batch * outC * outH * outW)

  for (let n = 0; n < batch; n++) {
    for (let oc = 0; oc < outC; oc++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let acc = bias ? bias.data[oc] : 0
          for (let ic = 0; ic < groupSize; ic++) {
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * sH + kh * dH - padH
                const iw = ow * sW + kw * dW - padW
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                  acc += input.data[n * inC * inH * inW + ic * inH * inW + ih * inW + iw]
                       * weight.data[oc * groupSize * kH * kW + ic * kH * kW + kh * kW + kw]
                }
              }
            }
          }
          out[n * outC * outH * outW + oc * outH * outW + oh * outW + ow] = acc
        }
      }
    }
  }
  return out
}


// === shouldUseIm2col ===

describe('shouldUseIm2col', () => {
  test('false for 3x3 stride-1 (Winograd handles it)', () => {
    const w = T.zeros([4, 3, 3, 3])
    expect(shouldUseIm2col(w, {})).toBe(false)
  })

  test('true for 5x5', () => {
    const w = T.zeros([4, 3, 5, 5])
    expect(shouldUseIm2col(w, {})).toBe(true)
  })

  test('true for 7x7', () => {
    const w = T.zeros([4, 3, 7, 7])
    expect(shouldUseIm2col(w, {})).toBe(true)
  })

  test('true for 3x3 with stride > 1', () => {
    const w = T.zeros([4, 3, 3, 3])
    expect(shouldUseIm2col(w, { stride: 2 })).toBe(true)
  })

  test('true for 3x3 with dilation > 1', () => {
    const w = T.zeros([4, 3, 3, 3])
    expect(shouldUseIm2col(w, { dilation: 2 })).toBe(true)
  })

  test('false for 1x1 pointwise', () => {
    const w = T.zeros([4, 3, 1, 1])
    expect(shouldUseIm2col(w, {})).toBe(false)
  })
})


// === im2col transform ===

describe('im2col', () => {
  test('output shape is [batch, inC*kH*kW, outH*outW]', () => {
    const input = T.tensor(Float32Array.from({ length: 1 * 2 * 6 * 6 }, (_, i) => i), [1, 2, 6, 6])
    const { cols, outH, outW, colRows, colCols } = im2col(input, { kH: 5, kW: 5, padding: 0 })

    expect(outH).toBe(2)
    expect(outW).toBe(2)
    expect(colRows).toBe(2 * 5 * 5)  // inC * kH * kW = 50
    expect(colCols).toBe(4)           // outH * outW = 4
    expect(cols.shape).toEqual([1, 50, 4])
  })

  test('im2col buffer size matches expected', () => {
    const batch = 2, inC = 3, H = 8, W = 8, kH = 5, kW = 5, pad = 2
    const input = T.zeros([batch, inC, H, W])
    const outH = convOutputSize(H, kH, 1, pad, 1)
    const outW = convOutputSize(W, kW, 1, pad, 1)
    const { cols } = im2col(input, { kH, kW, padding: pad })

    const expectedSize = batch * (inC * kH * kW) * (outH * outW)
    expect(cols.size).toBe(expectedSize)
  })

  test('im2col with stride', () => {
    const input = T.zeros([1, 1, 7, 7])
    const { outH, outW } = im2col(input, { kH: 3, kW: 3, stride: 2, padding: 0 })
    expect(outH).toBe(3)
    expect(outW).toBe(3)
  })
})


// === im2col forward vs direct conv (numerical equivalence) ===

describe('im2colForward vs direct conv', () => {
  test('matches direct conv for 5x5 no padding', () => {
    const batch = 1, inC = 1, outC = 1, H = 8, W = 8
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 25 }, () => Math.random() - 0.5), [outC, inC, 5, 5])
    const bias = T.tensor(Float32Array.from({ length: outC }, () => Math.random()), [outC])

    const directOut = conv2dForward(input, weight, bias, { padding: 0 })
    const im2colOut = im2colForward(input, weight, bias, { padding: 0 })

    expect(im2colOut.shape).toEqual(directOut.shape)
    const d = flat(directOut)
    const ic = flat(im2colOut)
    for (let i = 0; i < d.length; i++) {
      expect(ic[i]).toBeCloseTo(d[i], 2)
    }
  })

  test('matches direct conv for 5x5 with padding', () => {
    const batch = 1, inC = 2, outC = 3, H = 8, W = 8
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 25 }, () => Math.random() - 0.5), [outC, inC, 5, 5])
    const bias = T.tensor(Float32Array.from({ length: outC }, () => Math.random()), [outC])

    const directOut = conv2dForward(input, weight, bias, { padding: 2 })
    const im2colOut = im2colForward(input, weight, bias, { padding: 2 })

    expect(im2colOut.shape).toEqual(directOut.shape)
    const d = flat(directOut)
    const ic = flat(im2colOut)
    for (let i = 0; i < d.length; i++) {
      expect(ic[i]).toBeCloseTo(d[i], 2)
    }
  })

  test('matches direct conv for 7x7 stride 2 (ResNet first layer)', () => {
    const batch = 1, inC = 3, outC = 4, H = 16, W = 16
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 49 }, () => Math.random() - 0.5), [outC, inC, 7, 7])
    const bias = T.tensor(Float32Array.from({ length: outC }, () => Math.random()), [outC])

    const directOut = conv2dForward(input, weight, bias, { padding: 3, stride: 2 })
    const im2colOut = im2colForward(input, weight, bias, { padding: 3, stride: 2 })

    expect(im2colOut.shape).toEqual(directOut.shape)
    const d = flat(directOut)
    const ic = flat(im2colOut)
    for (let i = 0; i < d.length; i++) {
      expect(ic[i]).toBeCloseTo(d[i], 1)
    }
  })

  test('matches direct conv for batch > 1', () => {
    const batch = 2, inC = 2, outC = 2, H = 8, W = 8
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 25 }, () => Math.random() - 0.5), [outC, inC, 5, 5])
    const bias = T.tensor(Float32Array.from({ length: outC }, () => Math.random()), [outC])

    const directOut = conv2dForward(input, weight, bias, { padding: 2 })
    const im2colOut = im2colForward(input, weight, bias, { padding: 2 })

    expect(im2colOut.shape).toEqual(directOut.shape)
    const d = flat(directOut)
    const ic = flat(im2colOut)
    for (let i = 0; i < d.length; i++) {
      expect(ic[i]).toBeCloseTo(d[i], 1)
    }
  })

  test('matches direct conv without bias', () => {
    const batch = 1, inC = 1, outC = 2, H = 6, W = 6
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 25 }, () => Math.random() - 0.5), [outC, inC, 5, 5])

    const directOut = conv2dForward(input, weight, null, { padding: 2 })
    const im2colOut = im2colForward(input, weight, null, { padding: 2 })

    expect(im2colOut.shape).toEqual(directOut.shape)
    const d = flat(directOut)
    const ic = flat(im2colOut)
    for (let i = 0; i < d.length; i++) {
      expect(ic[i]).toBeCloseTo(d[i], 2)
    }
  })

  test('matches CPU reference for 5x5', () => {
    const batch = 1, inC = 1, outC = 1, H = 8, W = 8
    const input = T.tensor(Float32Array.from({ length: H * W }, (_, i) => i * 0.01), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: 25 }, () => Math.random() - 0.5), [outC, inC, 5, 5])
    const bias = T.tensor(new Float32Array([0.1]), [outC])

    const cpuRef = conv2dCPU(input, weight, bias, { padding: 2 })
    const gpuOut = im2colForward(input, weight, bias, { padding: 2 })

    const g = flat(gpuOut)
    for (let i = 0; i < cpuRef.length; i++) {
      expect(g[i]).toBeCloseTo(cpuRef[i], 2)
    }
  })
})


// === Auto-dispatch ===

describe('autograd conv2d auto-dispatch', () => {
  test('5x5 conv produces correct result (auto-selects im2col)', () => {
    const batch = 1, inC = 1, outC = 1, H = 8, W = 8
    const input = A.variable(T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W]))
    const weight = A.variable(T.tensor(Float32Array.from({ length: outC * inC * 25 }, () => Math.random() - 0.5), [outC, inC, 5, 5]))

    const result = A.conv2d(input, weight, null, { padding: 2 })
    const cpuRef = conv2dCPU(input.data, weight.data, null, { padding: 2 })
    const gpu = flat(result.data)

    for (let i = 0; i < cpuRef.length; i++) {
      expect(gpu[i]).toBeCloseTo(cpuRef[i], 1)
    }
  })

  test('3x3 stride-2 uses im2col (not Winograd)', () => {
    const batch = 1, inC = 1, outC = 1, H = 8, W = 8
    const input = A.variable(T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W]))
    const weight = A.variable(T.tensor(Float32Array.from({ length: outC * inC * 9 }, () => Math.random() - 0.5), [outC, inC, 3, 3]))

    // stride 2 should use im2col, not Winograd
    const result = A.conv2d(input, weight, null, { padding: 1, stride: 2 })
    expect(result.data.shape[2]).toBe(4)  // 8/2 = 4
    expect(result.data.shape[3]).toBe(4)
  })
})


// === Backward (gradient) tests ===

describe('im2col backward', () => {
  test('conv2d backward produces gradients for 5x5', () => {
    const batch = 1, inC = 1, outC = 1, H = 6, W = 6
    const input = A.variable(T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() * 0.1), [batch, inC, H, W]), { requiresGrad: true })
    const weight = A.variable(T.tensor(Float32Array.from({ length: outC * inC * 25 }, () => Math.random() * 0.1), [outC, inC, 5, 5]), { requiresGrad: true })
    const bias = A.variable(T.tensor(Float32Array.from({ length: outC }, () => 0.1), [outC]), { requiresGrad: true })

    const out = A.conv2d(input, weight, bias, { padding: 2 })
    const loss = A.sum(out)
    A.backward(loss)

    const gi = flat(input.grad)
    const gw = flat(weight.grad)
    const gb = flat(bias.grad)

    expect(gi.length).toBe(batch * inC * H * W)
    expect(gw.length).toBe(outC * inC * 25)
    expect(gb.length).toBe(outC)

    for (let i = 0; i < gi.length; i++) expect(Number.isFinite(gi[i])).toBe(true)
    for (let i = 0; i < gw.length; i++) expect(Number.isFinite(gw[i])).toBe(true)
    for (let i = 0; i < gb.length; i++) expect(Number.isFinite(gb[i])).toBe(true)
  })

  test('numerical gradient check for input (5x5 im2col)', () => {
    const batch = 1, inC = 1, outC = 1, H = 6, W = 6
    const eps = 1e-3
    const inputData = Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5)
    const weightData = Float32Array.from({ length: outC * inC * 25 }, () => Math.random() - 0.5)

    const input = A.variable(T.tensor(new Float32Array(inputData), [batch, inC, H, W]), { requiresGrad: true })
    const weight = A.variable(T.tensor(new Float32Array(weightData), [outC, inC, 5, 5]))

    const out = A.conv2d(input, weight, null, { padding: 2 })
    const loss = A.sum(out)
    A.backward(loss)
    const analyticGrad = flat(input.grad)

    for (let idx = 0; idx < Math.min(8, inputData.length); idx++) {
      const dataPlus = new Float32Array(inputData)
      dataPlus[idx] += eps
      const dataMin = new Float32Array(inputData)
      dataMin[idx] -= eps

      let lPlus, lMinus
      A.noGrad(() => {
        const ip = A.variable(T.tensor(dataPlus, [batch, inC, H, W]))
        const op = A.conv2d(ip, weight, null, { padding: 2 })
        lPlus = flat(op.data).reduce((a, b) => a + b, 0)

        const im = A.variable(T.tensor(dataMin, [batch, inC, H, W]))
        const om = A.conv2d(im, weight, null, { padding: 2 })
        lMinus = flat(om.data).reduce((a, b) => a + b, 0)
      })

      const numGrad = (lPlus - lMinus) / (2 * eps)
      expect(analyticGrad[idx]).toBeCloseTo(numGrad, 1)
    }
  })

  test('numerical gradient check for weight (5x5 im2col)', () => {
    const batch = 1, inC = 1, outC = 1, H = 6, W = 6
    const eps = 1e-3
    const inputData = Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5)
    const weightData = Float32Array.from({ length: outC * inC * 25 }, () => Math.random() - 0.5)

    const input = A.variable(T.tensor(new Float32Array(inputData), [batch, inC, H, W]))
    const weight = A.variable(T.tensor(new Float32Array(weightData), [outC, inC, 5, 5]), { requiresGrad: true })

    const out = A.conv2d(input, weight, null, { padding: 2 })
    const loss = A.sum(out)
    A.backward(loss)
    const analyticGrad = flat(weight.grad)

    for (let idx = 0; idx < Math.min(10, weightData.length); idx++) {
      const wPlus = new Float32Array(weightData)
      wPlus[idx] += eps
      const wMin = new Float32Array(weightData)
      wMin[idx] -= eps

      let lPlus, lMinus
      A.noGrad(() => {
        const wp = A.variable(T.tensor(wPlus, [outC, inC, 5, 5]))
        const op = A.conv2d(input, wp, null, { padding: 2 })
        lPlus = flat(op.data).reduce((a, b) => a + b, 0)

        const wm = A.variable(T.tensor(wMin, [outC, inC, 5, 5]))
        const om = A.conv2d(input, wm, null, { padding: 2 })
        lMinus = flat(om.data).reduce((a, b) => a + b, 0)
      })

      const numGrad = (lPlus - lMinus) / (2 * eps)
      expect(analyticGrad[idx]).toBeCloseTo(numGrad, 1)
    }
  })
})


// === Autograd pipeline ===

describe('im2col autograd pipeline', () => {
  test('conv2d(5x5) → relu → sum with backward', () => {
    const batch = 1, C1 = 2, C2 = 1, H = 8, W = 8

    const input = A.variable(T.tensor(Float32Array.from({ length: batch * C1 * H * W }, () => Math.random() - 0.5), [batch, C1, H, W]), { requiresGrad: true })
    const w1 = A.variable(T.tensor(Float32Array.from({ length: C2 * C1 * 25 }, () => (Math.random() - 0.5) * 0.1), [C2, C1, 5, 5]), { requiresGrad: true })
    const b1 = A.variable(T.tensor(Float32Array.from({ length: C2 }, () => 0), [C2]), { requiresGrad: true })

    const h1 = A.relu(A.conv2d(input, w1, b1, { padding: 2 }))
    const loss = A.sum(h1)
    A.backward(loss)

    for (const p of [input, w1, b1]) {
      expect(p.grad).not.toBeNull()
      const g = flat(p.grad)
      for (let i = 0; i < g.length; i++) {
        expect(Number.isFinite(g[i])).toBe(true)
      }
    }
  })
})
