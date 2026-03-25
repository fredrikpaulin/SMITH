// smith/tests/flash_attention.test.js
// Tests for Phase 7: Flash Attention
// Core property: flash attention must produce numerically equivalent results
// to standard attention, while using O(n) memory instead of O(n²).

import { test, expect } from 'bun:test'
import smith from '../src/index.js'

const T = { tensor: smith.tensor, zeros: smith.zeros, ones: smith.ones, rand: smith.rand, toArray: smith.toArray }
const A = {
  variable: smith.variable, param: smith.param, backward: smith.backward,
  zeroGrad: smith.zeroGrad, noGrad: smith.noGrad,
  matmul: smith.matmul, transpose: smith.transpose, scale: smith.scale,
  add: smith.add, softmax: smith.softmax, reshape: smith.reshape,
  flashAttention: smith.flashAttention,
}

// Helper: standard attention (decomposed) for reference
function standardAttention(Q, K, V, causal) {
  const dk = Q.data.shape[Q.data.shape.length - 1]
  const scaleFactor = 1 / Math.sqrt(dk)
  const N = Q.data.shape[1]

  const ndim = K.data.shape.length
  const kAxes = []
  for (let i = 0; i < ndim - 2; i++) kAxes.push(i)
  kAxes.push(ndim - 1, ndim - 2)

  const scores = A.scale(A.matmul(Q, A.transpose(K, kAxes)), scaleFactor)

  let masked = scores
  if (causal) {
    // Build causal mask [N, N] and broadcast
    const maskData = new Float32Array(N * N)
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        maskData[i * N + j] = j > i ? -Infinity : 0
    const mask = A.variable(T.tensor(Array.from(maskData), [N, N]), { requiresGrad: false })
    masked = A.add(scores, mask)
  }

  const weights = A.softmax(masked, -1)
  return A.matmul(weights, V)
}

// --- Shape tests ---

test('flash attention output has correct shape', () => {
  const numHeads = 2, seqLen = 16, headDim = 8
  const Q = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: true })
  const K = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: true })
  const V = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: true })

  let out
  A.noGrad(() => { out = A.flashAttention(Q, K, V, true) })

  expect(out.data.shape).toEqual([numHeads, seqLen, headDim])
})

test('flash attention works with seqLen=1', () => {
  const numHeads = 2, seqLen = 1, headDim = 8
  const Q = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: false })
  const K = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: false })
  const V = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: false })

  let out
  A.noGrad(() => { out = A.flashAttention(Q, K, V, true) })
  expect(out.data.shape).toEqual([numHeads, seqLen, headDim])
})

test('flash attention works with seqLen not a multiple of tile size', () => {
  // Tile size Br=Bc=32, so seqLen=17 tests partial tiles
  const numHeads = 1, seqLen = 17, headDim = 8
  const Q = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: false })
  const K = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: false })
  const V = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: false })

  let out
  A.noGrad(() => { out = A.flashAttention(Q, K, V, true) })
  expect(out.data.shape).toEqual([numHeads, seqLen, headDim])
})

// --- Numerical equivalence ---

test('flash attention matches standard attention (causal, small)', () => {
  const numHeads = 2, seqLen = 8, headDim = 4
  const qData = T.rand([numHeads, seqLen, headDim])
  const kData = T.rand([numHeads, seqLen, headDim])
  const vData = T.rand([numHeads, seqLen, headDim])

  let flashOut, stdOut
  A.noGrad(() => {
    const Q1 = A.variable(qData, { requiresGrad: false })
    const K1 = A.variable(kData, { requiresGrad: false })
    const V1 = A.variable(vData, { requiresGrad: false })
    flashOut = A.flashAttention(Q1, K1, V1, true)

    const Q2 = A.variable(qData, { requiresGrad: false })
    const K2 = A.variable(kData, { requiresGrad: false })
    const V2 = A.variable(vData, { requiresGrad: false })
    stdOut = standardAttention(Q2, K2, V2, true)
  })

  const fData = flashOut.data.data
  const sData = stdOut.data.data
  expect(fData.length).toBe(sData.length)

  for (let i = 0; i < fData.length; i++) {
    expect(fData[i]).toBeCloseTo(sData[i], 3) // f32 tolerance
  }
})

test('flash attention matches standard attention (no causal mask)', () => {
  const numHeads = 2, seqLen = 8, headDim = 4
  const qData = T.rand([numHeads, seqLen, headDim])
  const kData = T.rand([numHeads, seqLen, headDim])
  const vData = T.rand([numHeads, seqLen, headDim])

  let flashOut, stdOut
  A.noGrad(() => {
    const Q1 = A.variable(qData, { requiresGrad: false })
    const K1 = A.variable(kData, { requiresGrad: false })
    const V1 = A.variable(vData, { requiresGrad: false })
    flashOut = A.flashAttention(Q1, K1, V1, false)

    const Q2 = A.variable(qData, { requiresGrad: false })
    const K2 = A.variable(kData, { requiresGrad: false })
    const V2 = A.variable(vData, { requiresGrad: false })
    stdOut = standardAttention(Q2, K2, V2, false)
  })

  const fData = flashOut.data.data
  const sData = stdOut.data.data
  for (let i = 0; i < fData.length; i++) {
    expect(fData[i]).toBeCloseTo(sData[i], 3)
  }
})

test('flash attention matches standard attention (larger seqLen)', () => {
  const numHeads = 4, seqLen = 64, headDim = 16
  const qData = T.rand([numHeads, seqLen, headDim])
  const kData = T.rand([numHeads, seqLen, headDim])
  const vData = T.rand([numHeads, seqLen, headDim])

  let flashOut, stdOut
  A.noGrad(() => {
    const Q1 = A.variable(qData, { requiresGrad: false })
    const K1 = A.variable(kData, { requiresGrad: false })
    const V1 = A.variable(vData, { requiresGrad: false })
    flashOut = A.flashAttention(Q1, K1, V1, true)

    const Q2 = A.variable(qData, { requiresGrad: false })
    const K2 = A.variable(kData, { requiresGrad: false })
    const V2 = A.variable(vData, { requiresGrad: false })
    stdOut = standardAttention(Q2, K2, V2, true)
  })

  const fData = flashOut.data.data
  const sData = stdOut.data.data
  for (let i = 0; i < fData.length; i++) {
    expect(fData[i]).toBeCloseTo(sData[i], 2)
  }
})

// --- Gradient tests ---

test('flash attention backward produces gradients for Q, K, V', () => {
  const numHeads = 2, seqLen = 8, headDim = 4
  const Q = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: true })
  const K = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: true })
  const V = A.variable(T.rand([numHeads, seqLen, headDim]), { requiresGrad: true })

  const out = A.flashAttention(Q, K, V, true)
  const loss = smith.sum(out)
  smith.backward(loss)

  expect(Q.grad).not.toBeNull()
  expect(K.grad).not.toBeNull()
  expect(V.grad).not.toBeNull()
  expect(Q.grad.shape).toEqual([numHeads, seqLen, headDim])
  expect(K.grad.shape).toEqual([numHeads, seqLen, headDim])
  expect(V.grad.shape).toEqual([numHeads, seqLen, headDim])
})

test('flash attention gradients are close to standard attention gradients', () => {
  const numHeads = 2, seqLen = 8, headDim = 4
  const qData = T.rand([numHeads, seqLen, headDim])
  const kData = T.rand([numHeads, seqLen, headDim])
  const vData = T.rand([numHeads, seqLen, headDim])

  // Flash path
  const Q1 = A.variable(qData, { requiresGrad: true })
  const K1 = A.variable(kData, { requiresGrad: true })
  const V1 = A.variable(vData, { requiresGrad: true })
  const flashOut = A.flashAttention(Q1, K1, V1, true)
  const flashLoss = smith.sum(flashOut)
  smith.backward(flashLoss)

  // Standard path
  const Q2 = A.variable(qData, { requiresGrad: true })
  const K2 = A.variable(kData, { requiresGrad: true })
  const V2 = A.variable(vData, { requiresGrad: true })
  const stdOut = standardAttention(Q2, K2, V2, true)
  const stdLoss = smith.sum(stdOut)
  smith.backward(stdLoss)

  // Compare gradients
  const tol = 0.05 // relaxed tolerance for flash backward numerical differences
  for (let i = 0; i < Q1.grad.data.length; i++) {
    expect(Q1.grad.data[i]).toBeCloseTo(Q2.grad.data[i], 1)
  }
  for (let i = 0; i < K1.grad.data.length; i++) {
    expect(K1.grad.data[i]).toBeCloseTo(K2.grad.data[i], 1)
  }
  for (let i = 0; i < V1.grad.data.length; i++) {
    expect(V1.grad.data[i]).toBeCloseTo(V2.grad.data[i], 1)
  }
})

// --- CPU reference backward (validates GPU kernel independently) ---

test('flash backward dK matches CPU reference backward', () => {
  const numHeads = 2, N = 8, d = 4
  const scale = 1 / Math.sqrt(d)

  const qData = T.rand([numHeads, N, d])
  const kData = T.rand([numHeads, N, d])
  const vData = T.rand([numHeads, N, d])

  // GPU flash forward + backward
  const Q = A.variable(qData, { requiresGrad: true })
  const K = A.variable(kData, { requiresGrad: true })
  const V = A.variable(vData, { requiresGrad: true })
  const out = A.flashAttention(Q, K, V, true)
  smith.backward(smith.sum(out))

  // CPU reference backward (same math, no GPU)
  const q = new Float32Array(qData.data)
  const k = new Float32Array(kData.data)
  const v = new Float32Array(vData.data)
  const o = new Float32Array(out.data.data)
  const dK_cpu = new Float32Array(numHeads * N * d)

  for (let h = 0; h < numHeads; h++) {
    const off = h * N * d
    const P = new Float32Array(N * N)

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let dot = 0
        for (let kk = 0; kk < d; kk++) dot += q[off + i * d + kk] * k[off + j * d + kk]
        P[i * N + j] = j > i ? -Infinity : dot * scale
      }
      let maxS = -Infinity
      for (let j = 0; j < N; j++) maxS = Math.max(maxS, P[i * N + j])
      let sumExp = 0
      for (let j = 0; j < N; j++) { P[i * N + j] = Math.exp(P[i * N + j] - maxS); sumExp += P[i * N + j] }
      for (let j = 0; j < N; j++) P[i * N + j] /= sumExp
    }

    const D = new Float32Array(N)
    for (let i = 0; i < N; i++) for (let kk = 0; kk < d; kk++) D[i] += o[off + i * d + kk]

    const dS = new Float32Array(N * N)
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let dp = 0
      for (let kk = 0; kk < d; kk++) dp += v[off + j * d + kk]
      dS[i * N + j] = P[i * N + j] * (dp - D[i])
    }

    for (let j = 0; j < N; j++) for (let kk = 0; kk < d; kk++) {
      let sum = 0
      for (let i = 0; i < N; i++) sum += dS[i * N + j] * q[off + i * d + kk]
      dK_cpu[off + j * d + kk] = scale * sum
    }
  }

  for (let i = 0; i < dK_cpu.length; i++) {
    expect(K.grad.data[i]).toBeCloseTo(dK_cpu[i], 2)
  }
})

// --- Model-level integration ---

test('forwardFlash produces same logits shape as forward', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 32 })
  const tokens = [1, 5, 10, 3]

  let stdLogits, flashLogits
  A.noGrad(() => {
    stdLogits = smith.forward(model, tokens).logits
    flashLogits = smith.forwardFlash(model, tokens).logits
  })

  expect(flashLogits.data.shape).toEqual(stdLogits.data.shape)
})

test('forwardFlash and forward produce equivalent logits', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 32 })
  const tokens = [1, 5, 10]

  let stdLogits, flashLogits
  A.noGrad(() => {
    stdLogits = smith.forward(model, tokens).logits
    flashLogits = smith.forwardFlash(model, tokens).logits
  })

  const sData = stdLogits.data.data
  const fData = flashLogits.data.data
  for (let i = 0; i < sData.length; i++) {
    expect(fData[i]).toBeCloseTo(sData[i], 2)
  }
})

// --- Causal masking correctness ---

test('flash attention with causal mask: row 0 only attends to position 0', () => {
  // With causal masking, Q[h, 0, :] should only see K[h, 0, :]
  // So the output for row 0 should be V[h, 0, :] exactly
  const numHeads = 1, seqLen = 4, headDim = 4
  const Q = A.variable(T.ones([numHeads, seqLen, headDim]), { requiresGrad: false })
  const K = A.variable(T.ones([numHeads, seqLen, headDim]), { requiresGrad: false })

  // Set V so each row is distinct
  const vArr = new Float32Array(seqLen * headDim)
  for (let i = 0; i < seqLen; i++)
    for (let j = 0; j < headDim; j++)
      vArr[i * headDim + j] = i + 1 // row 0 = [1,1,1,1], row 1 = [2,2,2,2], etc.
  const V = A.variable(T.tensor(Array.from(vArr), [numHeads, seqLen, headDim]), { requiresGrad: false })

  let out
  A.noGrad(() => { out = A.flashAttention(Q, K, V, true) })

  // Row 0: only attends to position 0, so output = V[0] = [1,1,1,1]
  for (let j = 0; j < headDim; j++) {
    expect(out.data.data[j]).toBeCloseTo(1.0, 3)
  }
})
