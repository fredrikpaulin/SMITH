// smith/tests/gpu_ops.test.js
// Tests for Phase 11: GPU RoPE, RMSNorm, SwiGLU

import { test, expect } from 'bun:test'
import * as T from '../src/tensor.js'
import * as A from '../src/autograd.js'
import { precomputeRoPE, ropeForward, ropeBackward } from '../src/ops/rope.js'
import { rmsnormForward, rmsnormBackward } from '../src/ops/rmsnorm.js'
import { swigluForward, swigluBackward } from '../src/ops/swiglu.js'

// Flat read helper
function flat(t) {
  const arr = new Float32Array(t.size)
  for (let i = 0; i < t.size; i++) arr[i] = t.data[i]
  return arr
}

// --- CPU reference implementations ---

function cpuRoPE(input, cosTab, sinTab, seqLen, dim, startPos) {
  const halfDim = dim / 2
  const out = new Float32Array(seqLen * dim)
  for (let s = 0; s < seqLen; s++) {
    const pos = startPos + s
    for (let i = 0; i < halfDim; i++) {
      const x0 = input[s * dim + i]
      const x1 = input[s * dim + halfDim + i]
      const c = cosTab[pos * halfDim + i]
      const sn = sinTab[pos * halfDim + i]
      out[s * dim + i] = x0 * c - x1 * sn
      out[s * dim + halfDim + i] = x0 * sn + x1 * c
    }
  }
  return out
}

function cpuRMSNorm(input, gamma, rows, cols, eps) {
  const out = new Float32Array(rows * cols)
  for (let r = 0; r < rows; r++) {
    let sumSq = 0
    for (let c = 0; c < cols; c++) {
      const v = input[r * cols + c]
      sumSq += v * v
    }
    const rms = Math.sqrt(sumSq / cols + eps)
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = input[r * cols + c] * gamma[c] / rms
    }
  }
  return out
}

function cpuSwiGLU(gate, up, size) {
  const out = new Float32Array(size)
  for (let i = 0; i < size; i++) {
    const g = gate[i]
    const silu_g = g / (1 + Math.exp(-g))
    out[i] = silu_g * up[i]
  }
  return out
}

// === RoPE Tests ===

test('precomputeRoPE shape', () => {
  const table = precomputeRoPE(64, 128)
  expect(table.cos.shape).toEqual([128, 32])
  expect(table.sin.shape).toEqual([128, 32])
})

test('precomputeRoPE cos[0] = 1, sin[0] = 0', () => {
  const table = precomputeRoPE(64, 128)
  const cos = flat(table.cos)
  const sin = flat(table.sin)
  // At position 0, angle = 0 for all frequencies
  for (let i = 0; i < 32; i++) {
    expect(Math.abs(cos[i] - 1)).toBeLessThan(1e-6)
    expect(Math.abs(sin[i])).toBeLessThan(1e-6)
  }
})

test('precomputeRoPE custom frequency base', () => {
  const t1 = precomputeRoPE(64, 128, 10000)
  const t2 = precomputeRoPE(64, 128, 500000)
  // Different bases should produce different cos values at position 1
  const cos1 = flat(t1.cos)
  const cos2 = flat(t2.cos)
  // Position 1, freq index 1 (index 0 has freq=1.0 regardless of base)
  expect(cos1[32 + 1]).not.toBe(cos2[32 + 1])
})

test('ropeForward matches CPU reference', () => {
  const seqLen = 4, dim = 8
  const table = precomputeRoPE(dim, 16)
  const inputData = T.rand([seqLen, dim])
  const out = ropeForward(inputData, table, 0)

  const cpuOut = cpuRoPE(flat(inputData), flat(table.cos), flat(table.sin), seqLen, dim, 0)
  const gpuOut = flat(out)
  for (let i = 0; i < cpuOut.length; i++) {
    expect(Math.abs(gpuOut[i] - cpuOut[i])).toBeLessThan(1e-5)
  }
})

test('ropeForward with startPos offset', () => {
  const seqLen = 2, dim = 8
  const table = precomputeRoPE(dim, 16)
  const inputData = T.rand([seqLen, dim])

  const out0 = ropeForward(inputData, table, 0)
  const out5 = ropeForward(inputData, table, 5)
  // Different start positions should give different output
  expect(flat(out0)[0]).not.toBe(flat(out5)[0])

  // Verify against CPU ref with offset
  const cpuOut = cpuRoPE(flat(inputData), flat(table.cos), flat(table.sin), seqLen, dim, 5)
  const gpuOut = flat(out5)
  for (let i = 0; i < cpuOut.length; i++) {
    expect(Math.abs(gpuOut[i] - cpuOut[i])).toBeLessThan(1e-5)
  }
})

test('ropeForward at position 0 is near identity for small angles', () => {
  // At position 0, cos=1, sin=0, so output should equal input
  const dim = 8
  const table = precomputeRoPE(dim, 16)
  const inputData = T.rand([1, dim])
  const out = ropeForward(inputData, table, 0)

  const inArr = flat(inputData)
  const outArr = flat(out)
  for (let i = 0; i < dim; i++) {
    expect(Math.abs(outArr[i] - inArr[i])).toBeLessThan(1e-5)
  }
})

test('ropeBackward inverts ropeForward', () => {
  // RoPE backward with same cos/sin should undo the rotation
  const seqLen = 3, dim = 8
  const table = precomputeRoPE(dim, 16)
  const inputData = T.rand([seqLen, dim])

  const rotated = ropeForward(inputData, table, 2)
  const recovered = ropeBackward(rotated, table, 2)

  const inArr = flat(inputData)
  const recArr = flat(recovered)
  for (let i = 0; i < inArr.length; i++) {
    expect(Math.abs(recArr[i] - inArr[i])).toBeLessThan(1e-5)
  }
})

test('autograd rope forward and backward', () => {
  const dim = 8
  const table = precomputeRoPE(dim, 16)
  const input = A.variable(T.rand([2, dim]), { requiresGrad: true })
  const out = A.rope(input, table, 0)
  expect(out.data.shape).toEqual([2, dim])

  const loss = A.sum(out)
  A.backward(loss)
  expect(input.grad).not.toBeNull()
  expect(input.grad.shape).toEqual([2, dim])
})

// === RMSNorm Tests ===

test('rmsnormForward matches CPU reference', () => {
  const rows = 3, cols = 8
  const input = T.rand([rows, cols])
  const gamma = T.rand([cols])
  const eps = 1e-5

  const out = rmsnormForward(input, gamma, eps)
  const cpuOut = cpuRMSNorm(flat(input), flat(gamma), rows, cols, eps)

  const gpuOut = flat(out)
  for (let i = 0; i < cpuOut.length; i++) {
    expect(Math.abs(gpuOut[i] - cpuOut[i])).toBeLessThan(1e-4)
  }
})

test('rmsnormForward with ones gamma is just normalization', () => {
  const input = T.tensor([3, 4], [1, 2])
  const gamma = T.ones([2])
  const out = rmsnormForward(input, gamma, 1e-5)
  const arr = flat(out)
  // rms = sqrt((9+16)/2 + 1e-5) = sqrt(12.5)
  const rms = Math.sqrt(12.5 + 1e-5)
  expect(Math.abs(arr[0] - 3 / rms)).toBeLessThan(1e-4)
  expect(Math.abs(arr[1] - 4 / rms)).toBeLessThan(1e-4)
})

test('rmsnormForward 1D input', () => {
  const input = T.rand([16])
  const gamma = T.rand([16])
  const out = rmsnormForward(input, gamma)
  expect(out.shape).toEqual([16])
})

test('rmsnormBackward produces correct shapes', () => {
  const input = T.rand([4, 8])
  const gamma = T.rand([8])
  const out = rmsnormForward(input, gamma)
  const gradOut = T.ones(out.shape)

  const { gradInput, gradGamma } = rmsnormBackward(gradOut, input, gamma)
  expect(gradInput.shape).toEqual([4, 8])
  expect(gradGamma.shape).toEqual([8])
})

test('rmsnorm numerical gradient check', () => {
  const eps_num = 1e-3
  const input = T.rand([2, 4])
  const gamma = T.ones([4])
  const norm_eps = 1e-5

  // Forward + backward
  const out = rmsnormForward(input, gamma, norm_eps)
  const gradOut = T.ones(out.shape)
  const { gradInput } = rmsnormBackward(gradOut, input, gamma, norm_eps)
  const analyticGrad = flat(gradInput)

  // Numerical gradient for a few elements
  const inArr = flat(input)
  for (let idx = 0; idx < Math.min(4, inArr.length); idx++) {
    const inPlus = T.tensor([...inArr], input.shape)
    inPlus.data[idx] += eps_num
    const outPlus = rmsnormForward(inPlus, gamma, norm_eps)
    let sumPlus = 0
    const fPlus = flat(outPlus)
    for (let i = 0; i < fPlus.length; i++) sumPlus += fPlus[i]

    const inMinus = T.tensor([...inArr], input.shape)
    inMinus.data[idx] -= eps_num
    const outMinus = rmsnormForward(inMinus, gamma, norm_eps)
    let sumMinus = 0
    const fMinus = flat(outMinus)
    for (let i = 0; i < fMinus.length; i++) sumMinus += fMinus[i]

    const numGrad = (sumPlus - sumMinus) / (2 * eps_num)
    expect(Math.abs(analyticGrad[idx] - numGrad)).toBeLessThan(0.05)
  }
})

test('autograd rmsNorm forward and backward', () => {
  const input = A.variable(T.rand([3, 8]), { requiresGrad: true })
  const gamma = A.variable(T.ones([8]), { requiresGrad: true })
  const out = A.rmsNorm(input, gamma)
  expect(out.data.shape).toEqual([3, 8])

  const loss = A.sum(out)
  A.backward(loss)
  expect(input.grad).not.toBeNull()
  expect(gamma.grad).not.toBeNull()
  expect(input.grad.shape).toEqual([3, 8])
  expect(gamma.grad.shape).toEqual([8])
})

// === SwiGLU Tests ===

test('swigluForward matches CPU reference', () => {
  const size = 16
  const gate = T.rand([4, 4])
  const up = T.rand([4, 4])
  const out = swigluForward(gate, up)

  const cpuOut = cpuSwiGLU(flat(gate), flat(up), size)
  const gpuOut = flat(out)
  for (let i = 0; i < cpuOut.length; i++) {
    expect(Math.abs(gpuOut[i] - cpuOut[i])).toBeLessThan(1e-5)
  }
})

test('swigluForward with zero gate produces zero', () => {
  const gate = T.zeros([2, 4])
  const up = T.rand([2, 4])
  const out = swigluForward(gate, up)
  const arr = flat(out)
  // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0, so out = 0 * up = 0
  for (let i = 0; i < arr.length; i++) {
    expect(Math.abs(arr[i])).toBeLessThan(1e-6)
  }
})

test('swigluForward large positive gate ≈ gate * up', () => {
  // For large x, silu(x) ≈ x (sigmoid → 1)
  const gate = T.tensor([10, 10, 10, 10], [1, 4])
  const up = T.tensor([1, 2, 3, 4], [1, 4])
  const out = swigluForward(gate, up)
  const arr = flat(out)
  // silu(10) ≈ 10 * 0.99995 ≈ 9.9995
  for (let i = 0; i < 4; i++) {
    const expected = 10 * flat(up)[i]
    expect(Math.abs(arr[i] - expected) / expected).toBeLessThan(0.001)
  }
})

test('swigluBackward produces correct shapes', () => {
  const gate = T.rand([3, 8])
  const up = T.rand([3, 8])
  const gradOut = T.ones([3, 8])
  const { gradGate, gradUp } = swigluBackward(gradOut, gate, up)
  expect(gradGate.shape).toEqual([3, 8])
  expect(gradUp.shape).toEqual([3, 8])
})

test('swiglu numerical gradient check (gate)', () => {
  const eps = 1e-3
  const gateData = T.rand([2, 4])
  const upData = T.rand([2, 4])

  // Analytic backward
  const gradOut = T.ones([2, 4])
  const { gradGate } = swigluBackward(gradOut, gateData, upData)
  const analyticGrad = flat(gradGate)

  const gArr = flat(gateData)
  for (let idx = 0; idx < Math.min(4, gArr.length); idx++) {
    const gPlus = T.tensor([...gArr], gateData.shape)
    gPlus.data[idx] += eps
    const outPlus = swigluForward(gPlus, upData)
    let sumPlus = 0
    const fPlus = flat(outPlus)
    for (let i = 0; i < fPlus.length; i++) sumPlus += fPlus[i]

    const gMinus = T.tensor([...gArr], gateData.shape)
    gMinus.data[idx] -= eps
    const outMinus = swigluForward(gMinus, upData)
    let sumMinus = 0
    const fMinus = flat(outMinus)
    for (let i = 0; i < fMinus.length; i++) sumMinus += fMinus[i]

    const numGrad = (sumPlus - sumMinus) / (2 * eps)
    expect(Math.abs(analyticGrad[idx] - numGrad)).toBeLessThan(0.05)
  }
})

test('swiglu numerical gradient check (up)', () => {
  const eps = 1e-3
  const gateData = T.rand([2, 4])
  const upData = T.rand([2, 4])

  const gradOut = T.ones([2, 4])
  const { gradUp } = swigluBackward(gradOut, gateData, upData)
  const analyticGrad = flat(gradUp)

  const uArr = flat(upData)
  for (let idx = 0; idx < Math.min(4, uArr.length); idx++) {
    const uPlus = T.tensor([...uArr], upData.shape)
    uPlus.data[idx] += eps
    const outPlus = swigluForward(gateData, uPlus)
    let sumPlus = 0
    const fPlus = flat(outPlus)
    for (let i = 0; i < fPlus.length; i++) sumPlus += fPlus[i]

    const uMinus = T.tensor([...uArr], upData.shape)
    uMinus.data[idx] -= eps
    const outMinus = swigluForward(gateData, uMinus)
    let sumMinus = 0
    const fMinus = flat(outMinus)
    for (let i = 0; i < fMinus.length; i++) sumMinus += fMinus[i]

    const numGrad = (sumPlus - sumMinus) / (2 * eps)
    expect(Math.abs(analyticGrad[idx] - numGrad)).toBeLessThan(0.05)
  }
})

test('autograd swiglu forward and backward', () => {
  const gate = A.variable(T.rand([3, 8]), { requiresGrad: true })
  const up = A.variable(T.rand([3, 8]), { requiresGrad: true })
  const out = A.swiglu(gate, up)
  expect(out.data.shape).toEqual([3, 8])

  const loss = A.sum(out)
  A.backward(loss)
  expect(gate.grad).not.toBeNull()
  expect(up.grad).not.toBeNull()
  expect(gate.grad.shape).toEqual([3, 8])
  expect(up.grad.shape).toEqual([3, 8])
})

// === Integration: pipeline test ===

test('rmsNorm → linear → swiglu pipeline', () => {
  const seqLen = 4, dim = 16, ffnDim = 32
  const input = A.variable(T.rand([seqLen, dim]), { requiresGrad: true })
  const gamma = A.variable(T.ones([dim]), { requiresGrad: true })

  // RMSNorm
  const normed = A.rmsNorm(input, gamma)

  // Gate and up projections (simplified — just matmul with random weights)
  const gateW = A.variable(T.rand([dim, ffnDim]), { requiresGrad: true })
  const upW = A.variable(T.rand([dim, ffnDim]), { requiresGrad: true })
  const gateOut = A.matmul(normed, gateW)
  const upOut = A.matmul(normed, upW)

  // Fused SwiGLU
  const activated = A.swiglu(gateOut, upOut)
  expect(activated.data.shape).toEqual([seqLen, ffnDim])

  // Down projection
  const downW = A.variable(T.rand([ffnDim, dim]), { requiresGrad: true })
  const out = A.matmul(activated, downW)
  expect(out.data.shape).toEqual([seqLen, dim])

  // Backward through the whole pipeline
  const loss = A.sum(out)
  A.backward(loss)

  expect(input.grad).not.toBeNull()
  expect(gamma.grad).not.toBeNull()
  expect(gateW.grad).not.toBeNull()
  expect(upW.grad).not.toBeNull()
  expect(downW.grad).not.toBeNull()
})
