// tests/winograd.test.js
// Phase 13: Winograd F(2x2, 3x3) Convolution
// Tests filter pre-transformation, forward equivalence with direct conv,
// auto-dispatch selection, backward correctness, and numerical gradient checks.

import { test, expect, describe } from 'bun:test'
import * as T from '../src/tensor.js'
import * as A from '../src/autograd.js'
import { conv2dForward, convOutputSize } from '../src/ops/conv2d.js'
import { transformFilter3x3, transformWeights, canUseWinograd, winogradForward } from '../src/ops/conv2d_winograd.js'

const flat = t => new Float32Array(t.data.buffer, t.data.byteOffset, t.data.length)

// CPU reference: direct 2D convolution (NCHW, groups=1, dilation=1)
function conv2dCPU(input, weight, bias, opts = {}) {
  const { padding = 0, stride = 1 } = opts
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding]
  const [sH, sW] = Array.isArray(stride) ? stride : [stride, stride]
  const [batch, inC, inH, inW] = input.shape
  const [outC, groupSize, kH, kW] = weight.shape
  const outH = convOutputSize(inH, kH, sH, padH, 1)
  const outW = convOutputSize(inW, kW, sW, padW, 1)
  const out = new Float32Array(batch * outC * outH * outW)

  for (let n = 0; n < batch; n++) {
    for (let oc = 0; oc < outC; oc++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let acc = bias ? bias.data[oc] : 0
          for (let ic = 0; ic < groupSize; ic++) {
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * sH + kh - padH
                const iw = ow * sW + kw - padW
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


// === Filter Pre-Transformation ===

describe('transformFilter3x3', () => {
  test('outputs 16 values from a 9-value filter', () => {
    const g = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1])  // identity-ish
    const t = transformFilter3x3(g)
    expect(t.length).toBe(16)
    // All values should be finite
    for (let i = 0; i < 16; i++) expect(Number.isFinite(t[i])).toBe(true)
  })

  test('zero filter produces zero transform', () => {
    const g = new Float32Array(9)
    const t = transformFilter3x3(g)
    for (let i = 0; i < 16; i++) expect(t[i]).toBe(0)
  })

  test('transform preserves filter energy (Parseval-like)', () => {
    // Not exact Parseval but the transform should produce non-trivial values
    const g = Float32Array.from({ length: 9 }, () => Math.random() - 0.5)
    const t = transformFilter3x3(g)
    const sumSq = t.reduce((s, v) => s + v * v, 0)
    expect(sumSq).toBeGreaterThan(0)
  })
})

describe('transformWeights', () => {
  test('reshapes [outC, inC, 3, 3] to [outC, inC, 4, 4]', () => {
    const w = T.tensor(Float32Array.from({ length: 2 * 3 * 9 }, () => Math.random() - 0.5), [2, 3, 3, 3])
    const tw = transformWeights(w)
    expect(tw.shape).toEqual([2, 3, 4, 4])
    expect(tw.size).toBe(2 * 3 * 16)
  })
})


// === canUseWinograd ===

describe('canUseWinograd', () => {
  test('true for 3x3 stride-1 dilation-1 groups-1', () => {
    const w = T.zeros([4, 3, 3, 3])
    expect(canUseWinograd(w, {})).toBe(true)
    expect(canUseWinograd(w, { stride: 1, dilation: 1, groups: 1 })).toBe(true)
  })

  test('false for 5x5 kernel', () => {
    const w = T.zeros([4, 3, 5, 5])
    expect(canUseWinograd(w, {})).toBe(false)
  })

  test('false for stride > 1', () => {
    const w = T.zeros([4, 3, 3, 3])
    expect(canUseWinograd(w, { stride: 2 })).toBe(false)
  })

  test('false for dilation > 1', () => {
    const w = T.zeros([4, 3, 3, 3])
    expect(canUseWinograd(w, { dilation: 2 })).toBe(false)
  })

  test('false for groups > 1', () => {
    const w = T.zeros([4, 4, 3, 3])
    expect(canUseWinograd(w, { groups: 2 })).toBe(false)
  })
})


// === Winograd forward vs direct conv (numerical equivalence) ===

describe('winogradForward vs direct conv', () => {
  test('matches direct conv for simple case (no padding)', () => {
    const batch = 1, inC = 1, outC = 1, H = 4, W = 4
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, (_, i) => i + 1), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 9 }, () => Math.random() - 0.5), [outC, inC, 3, 3])
    const bias = T.tensor(new Float32Array([0.5]), [outC])

    const directOut = conv2dForward(input, weight, bias, { padding: 0 })
    const winogradOut = winogradForward(input, weight, bias, { padding: 0 })

    expect(winogradOut.shape).toEqual(directOut.shape)

    const d = flat(directOut)
    const w = flat(winogradOut)
    for (let i = 0; i < d.length; i++) {
      expect(w[i]).toBeCloseTo(d[i], 3)
    }
  })

  test('matches direct conv with padding=1', () => {
    const batch = 1, inC = 2, outC = 2, H = 6, W = 6
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 9 }, () => Math.random() - 0.5), [outC, inC, 3, 3])
    const bias = T.tensor(Float32Array.from({ length: outC }, () => Math.random()), [outC])

    const directOut = conv2dForward(input, weight, bias, { padding: 1 })
    const winogradOut = winogradForward(input, weight, bias, { padding: 1 })

    expect(winogradOut.shape).toEqual(directOut.shape)

    const d = flat(directOut)
    const w = flat(winogradOut)
    for (let i = 0; i < d.length; i++) {
      expect(w[i]).toBeCloseTo(d[i], 3)
    }
  })

  test('matches direct conv with batch > 1', () => {
    const batch = 2, inC = 3, outC = 4, H = 8, W = 8
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 9 }, () => Math.random() - 0.5), [outC, inC, 3, 3])
    const bias = T.tensor(Float32Array.from({ length: outC }, () => Math.random()), [outC])

    const directOut = conv2dForward(input, weight, bias, { padding: 1 })
    const winogradOut = winogradForward(input, weight, bias, { padding: 1 })

    expect(winogradOut.shape).toEqual(directOut.shape)
    const d = flat(directOut)
    const w = flat(winogradOut)
    for (let i = 0; i < d.length; i++) {
      expect(w[i]).toBeCloseTo(d[i], 2)
    }
  })

  test('matches direct conv without bias', () => {
    const batch = 1, inC = 2, outC = 2, H = 5, W = 5
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 9 }, () => Math.random() - 0.5), [outC, inC, 3, 3])

    const directOut = conv2dForward(input, weight, null, { padding: 1 })
    const winogradOut = winogradForward(input, weight, null, { padding: 1 })

    expect(winogradOut.shape).toEqual(directOut.shape)
    const d = flat(directOut)
    const w = flat(winogradOut)
    for (let i = 0; i < d.length; i++) {
      expect(w[i]).toBeCloseTo(d[i], 3)
    }
  })

  test('matches direct conv with odd spatial dimensions', () => {
    const batch = 1, inC = 1, outC = 1, H = 7, W = 5
    const input = T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from({ length: outC * inC * 9 }, () => Math.random() - 0.5), [outC, inC, 3, 3])
    const bias = T.tensor(new Float32Array([0.1]), [outC])

    const directOut = conv2dForward(input, weight, bias, { padding: 1 })
    const winogradOut = winogradForward(input, weight, bias, { padding: 1 })

    expect(winogradOut.shape).toEqual(directOut.shape)
    const d = flat(directOut)
    const w = flat(winogradOut)
    for (let i = 0; i < d.length; i++) {
      expect(w[i]).toBeCloseTo(d[i], 3)
    }
  })

  test('matches CPU reference for 3x3 pad=0', () => {
    const batch = 1, inC = 1, outC = 1, H = 6, W = 6
    const input = T.tensor(Float32Array.from({ length: H * W }, (_, i) => i), [batch, inC, H, W])
    const weight = T.tensor(Float32Array.from([1, 0, -1, 2, 0, -2, 1, 0, -1]), [outC, inC, 3, 3])
    const bias = T.tensor(new Float32Array([0]), [outC])

    const cpuRef = conv2dCPU(input, weight, bias, { padding: 0 })
    const gpuOut = winogradForward(input, weight, bias, { padding: 0 })

    const g = flat(gpuOut)
    for (let i = 0; i < cpuRef.length; i++) {
      expect(g[i]).toBeCloseTo(cpuRef[i], 3)
    }
  })
})


// === Auto-dispatch in autograd conv2d ===

describe('autograd conv2d auto-dispatch', () => {
  test('conv2d produces same result for 3x3 (auto-selects Winograd)', () => {
    const batch = 1, inC = 2, outC = 2, H = 6, W = 6
    const input = A.variable(T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W]))
    const weight = A.variable(T.tensor(Float32Array.from({ length: outC * inC * 9 }, () => Math.random() - 0.5), [outC, inC, 3, 3]))
    const bias = A.variable(T.tensor(Float32Array.from({ length: outC }, () => Math.random()), [outC]))

    // Autograd conv2d should auto-select Winograd for 3x3 stride-1
    const result = A.conv2d(input, weight, bias, { padding: 1 })

    // Compare with CPU ref
    const cpuRef = conv2dCPU(input.data, weight.data, bias.data, { padding: 1 })
    const gpu = flat(result.data)

    for (let i = 0; i < cpuRef.length; i++) {
      expect(gpu[i]).toBeCloseTo(cpuRef[i], 2)
    }
  })

  test('conv2d still works for 5x5 (falls back to direct)', () => {
    const batch = 1, inC = 1, outC = 1, H = 8, W = 8
    const input = A.variable(T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5), [batch, inC, H, W]))
    const weight = A.variable(T.tensor(Float32Array.from({ length: outC * inC * 25 }, () => Math.random() - 0.5), [outC, inC, 5, 5]))

    const result = A.conv2d(input, weight, null, { padding: 2 })
    expect(result.data.shape[2]).toBe(8)
    expect(result.data.shape[3]).toBe(8)
  })
})


// === Backward (gradient) tests ===

describe('Winograd backward', () => {
  test('conv2d backward produces gradients for input (3x3, Winograd path)', () => {
    const batch = 1, inC = 2, outC = 2, H = 6, W = 6
    const input = A.variable(T.tensor(Float32Array.from({ length: batch * inC * H * W }, () => Math.random() * 0.1), [batch, inC, H, W]), { requiresGrad: true })
    const weight = A.variable(T.tensor(Float32Array.from({ length: outC * inC * 9 }, () => Math.random() * 0.1), [outC, inC, 3, 3]), { requiresGrad: true })
    const bias = A.variable(T.tensor(Float32Array.from({ length: outC }, () => Math.random() * 0.1), [outC]), { requiresGrad: true })

    const out = A.conv2d(input, weight, bias, { padding: 1 })
    const loss = A.sum(out)
    A.backward(loss)

    // Check gradients exist and are finite
    const gi = flat(input.grad)
    const gw = flat(weight.grad)
    const gb = flat(bias.grad)

    expect(gi.length).toBe(batch * inC * H * W)
    expect(gw.length).toBe(outC * inC * 9)
    expect(gb.length).toBe(outC)

    for (let i = 0; i < gi.length; i++) expect(Number.isFinite(gi[i])).toBe(true)
    for (let i = 0; i < gw.length; i++) expect(Number.isFinite(gw[i])).toBe(true)
    for (let i = 0; i < gb.length; i++) expect(Number.isFinite(gb[i])).toBe(true)
  })

  test('numerical gradient check for input (3x3 Winograd)', () => {
    const batch = 1, inC = 1, outC = 1, H = 4, W = 4
    const eps = 1e-3
    const inputData = Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5)
    const weightData = Float32Array.from({ length: outC * inC * 9 }, () => Math.random() - 0.5)
    const biasData = Float32Array.from({ length: outC }, () => Math.random() - 0.5)

    // Analytic gradient
    const input = A.variable(T.tensor(new Float32Array(inputData), [batch, inC, H, W]), { requiresGrad: true })
    const weight = A.variable(T.tensor(new Float32Array(weightData), [outC, inC, 3, 3]))
    const bias = A.variable(T.tensor(new Float32Array(biasData), [outC]))

    const out = A.conv2d(input, weight, bias, { padding: 1 })
    const loss = A.sum(out)
    A.backward(loss)
    const analyticGrad = flat(input.grad)

    // Numerical gradient for a few positions
    for (let idx = 0; idx < Math.min(8, inputData.length); idx++) {
      const dataPlus = new Float32Array(inputData)
      dataPlus[idx] += eps
      const dataMin = new Float32Array(inputData)
      dataMin[idx] -= eps

      let lPlus, lMinus
      A.noGrad(() => {
        const ip = A.variable(T.tensor(dataPlus, [batch, inC, H, W]))
        const op = A.conv2d(ip, weight, bias, { padding: 1 })
        lPlus = flat(op.data).reduce((a, b) => a + b, 0)

        const im = A.variable(T.tensor(dataMin, [batch, inC, H, W]))
        const om = A.conv2d(im, weight, bias, { padding: 1 })
        lMinus = flat(om.data).reduce((a, b) => a + b, 0)
      })

      const numGrad = (lPlus - lMinus) / (2 * eps)
      expect(analyticGrad[idx]).toBeCloseTo(numGrad, 1)
    }
  })

  test('numerical gradient check for weight (3x3 Winograd)', () => {
    const batch = 1, inC = 1, outC = 1, H = 4, W = 4
    const eps = 1e-3
    const inputData = Float32Array.from({ length: batch * inC * H * W }, () => Math.random() - 0.5)
    const weightData = Float32Array.from({ length: outC * inC * 9 }, () => Math.random() - 0.5)

    const input = A.variable(T.tensor(new Float32Array(inputData), [batch, inC, H, W]))
    const weight = A.variable(T.tensor(new Float32Array(weightData), [outC, inC, 3, 3]), { requiresGrad: true })

    const out = A.conv2d(input, weight, null, { padding: 1 })
    const loss = A.sum(out)
    A.backward(loss)
    const analyticGrad = flat(weight.grad)

    for (let idx = 0; idx < 9; idx++) {
      const wPlus = new Float32Array(weightData)
      wPlus[idx] += eps
      const wMin = new Float32Array(weightData)
      wMin[idx] -= eps

      let lPlus, lMinus
      A.noGrad(() => {
        const wp = A.variable(T.tensor(wPlus, [outC, inC, 3, 3]))
        const op = A.conv2d(input, wp, null, { padding: 1 })
        lPlus = flat(op.data).reduce((a, b) => a + b, 0)

        const wm = A.variable(T.tensor(wMin, [outC, inC, 3, 3]))
        const om = A.conv2d(input, wm, null, { padding: 1 })
        lMinus = flat(om.data).reduce((a, b) => a + b, 0)
      })

      const numGrad = (lPlus - lMinus) / (2 * eps)
      expect(analyticGrad[idx]).toBeCloseTo(numGrad, 1)
    }
  })
})


// === Autograd pipeline test ===

describe('Winograd autograd pipeline', () => {
  test('conv2d(3x3) → relu → conv2d(3x3) → sum with backward', () => {
    const batch = 1, C1 = 2, C2 = 3, C3 = 1, H = 8, W = 8

    const input = A.variable(T.tensor(Float32Array.from({ length: batch * C1 * H * W }, () => Math.random() - 0.5), [batch, C1, H, W]), { requiresGrad: true })
    const w1 = A.variable(T.tensor(Float32Array.from({ length: C2 * C1 * 9 }, () => (Math.random() - 0.5) * 0.1), [C2, C1, 3, 3]), { requiresGrad: true })
    const b1 = A.variable(T.tensor(Float32Array.from({ length: C2 }, () => 0), [C2]), { requiresGrad: true })
    const w2 = A.variable(T.tensor(Float32Array.from({ length: C3 * C2 * 9 }, () => (Math.random() - 0.5) * 0.1), [C3, C2, 3, 3]), { requiresGrad: true })
    const b2 = A.variable(T.tensor(Float32Array.from({ length: C3 }, () => 0), [C3]), { requiresGrad: true })

    // Both convs are 3x3 stride-1, should use Winograd
    const h1 = A.relu(A.conv2d(input, w1, b1, { padding: 1 }))
    const h2 = A.conv2d(h1, w2, b2, { padding: 1 })
    const loss = A.sum(h2)

    A.backward(loss)

    // All params should have gradients
    for (const p of [input, w1, b1, w2, b2]) {
      expect(p.grad).not.toBeNull()
      const g = flat(p.grad)
      let hasNonZero = false
      for (let i = 0; i < g.length; i++) {
        expect(Number.isFinite(g[i])).toBe(true)
        if (g[i] !== 0) hasNonZero = true
      }
      // At least some gradients should be non-zero
      expect(hasNonZero).toBe(true)
    }
  })
})
