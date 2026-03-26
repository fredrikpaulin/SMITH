// smith/src/muon.js
// MuonAdamW optimizer: Muon for 2D matrix params, AdamW for everything else.
// Ported from Karpathy's autoresearch train.py.
//
// Muon uses Newton-Schulz orthogonalization (polar express) to approximate
// the spectral norm of the gradient, producing better update directions for
// matrix-shaped parameters. Non-matrix params (embeddings, biases, norms)
// use standard AdamW.

import * as T from './tensor.js'
import { run, GROUP_1D } from './dispatch.js'
import { matmul2d } from './ops/matmul.js'
import { transpose } from './ops/transpose.js'
import { scale as gpuScale } from './ops/mul.js'

// Newton-Schulz polynomial coefficients (polar express)
// 5 sets of (a, b, c) for iterative orthogonalization
const POLAR_COEFFS = [
  [8.156554524902461, -22.48329292557795, 15.878769915207462],
  [4.042929935166739, -2.808917465908714, 0.5000178451051316],
  [3.8916678022926607, -2.772484153217685, 0.5060648178503393],
  [3.285753657755655, -2.3681294933425376, 0.46449024233003106],
  [2.3465413258596377, -1.7097828382687081, 0.42323551169305323],
]

// --- GPU kernel dispatch helpers ---

function nesterovMomentum(grad, momentumBuf, momentum) {
  const n = grad.size
  const buf = new ArrayBuffer(8)
  new Float32Array(buf, 0, 1)[0] = momentum
  new Uint32Array(buf, 4, 1)[0] = n
  run('muon_nesterov', [
    { buffer: grad.buffer, index: 0 },
    { buffer: momentumBuf.buffer, index: 1 },
  ], { x: n }, { x: Math.min(n, GROUP_1D) },
  { data: new Uint8Array(buf), index: 2 })
}

function nsPoly(A, AA, B, b, c) {
  const n = A.size
  const buf = new ArrayBuffer(12)
  const f = new Float32Array(buf, 0, 2)
  const u = new Uint32Array(buf, 8, 1)
  f[0] = b; f[1] = c; u[0] = n
  run('muon_ns_poly', [
    { buffer: A.buffer, index: 0 },
    { buffer: AA.buffer, index: 1 },
    { buffer: B.buffer, index: 2 },
  ], { x: n }, { x: Math.min(n, GROUP_1D) },
  { data: new Uint8Array(buf), index: 3 })
}

function nsCombine(X, product, a) {
  const n = X.size
  const buf = new ArrayBuffer(8)
  new Float32Array(buf, 0, 1)[0] = a
  new Uint32Array(buf, 4, 1)[0] = n
  run('muon_ns_combine', [
    { buffer: X.buffer, index: 0 },
    { buffer: product.buffer, index: 1 },
  ], { x: n }, { x: Math.min(n, GROUP_1D) },
  { data: new Uint8Array(buf), index: 2 })
}

function muonUpdate(param, g, lr, wd) {
  const n = param.size
  const buf = new ArrayBuffer(12)
  const f = new Float32Array(buf, 0, 2)
  const u = new Uint32Array(buf, 8, 1)
  f[0] = lr; f[1] = wd; u[0] = n
  run('muon_update', [
    { buffer: param.buffer, index: 0 },
    { buffer: g.buffer, index: 1 },
  ], { x: n }, { x: Math.min(n, GROUP_1D) },
  { data: new Uint8Array(buf), index: 2 })
}

// --- Frobenius norm via CPU (unified memory, zero-copy after GPU sync) ---

function frobeniusNorm(t) {
  const d = t.data
  let sum = 0
  for (let i = 0; i < d.length; i++) sum += d[i] * d[i]
  return Math.sqrt(sum)
}

// --- Newton-Schulz orthogonalization ---
// Approximates the polar decomposition X → U where X = U·S·V^T
// After NS iterations, X approximates the orthogonal factor U·V^T.
// All operations are GPU dispatches — no CPU reads/writes on GPU buffers.

function newtonSchulz(g, nsSteps) {
  const [rows, cols] = g.shape
  const tall = rows > cols  // strictly greater — matches reference (square → wide path)

  // Normalize: X = g / (||g||_F * 1.02 + eps)
  const norm = frobeniusNorm(g)
  if (norm < 1e-12) return g  // zero gradient, nothing to orthogonalize
  let X = gpuScale(g, 1.0 / (norm * 1.02 + 1e-6))

  for (let i = 0; i < nsSteps; i++) {
    const [a, b, c] = POLAR_COEFFS[i]

    // Compute Gram matrix A
    let A
    if (tall) {
      A = matmul2d(transpose(X), X)  // X^T @ X  [cols x cols]
    } else {
      A = matmul2d(X, transpose(X))  // X @ X^T  [rows x rows]
    }

    // AA = A @ A
    const AA = matmul2d(A, A)

    // B = b*A + c*AA (element-wise via GPU kernel)
    const B = T.create(A.shape, A.dtype)
    nsPoly(A, AA, B, b, c)

    // product = X@B (tall) or B@X (wide)
    const product = tall ? matmul2d(X, B) : matmul2d(B, X)

    // X = a*X + product (in-place on X via GPU kernel)
    nsCombine(X, product, a)
  }

  // Copy result back into g's buffer so caller sees the orthogonalized gradient
  // Use a GPU copy (scale by 1.0)
  run('elementwise_scale', [
    { buffer: X.buffer, index: 0 },
    { buffer: g.buffer, index: 1 },
  ], { x: g.size }, null,
  { data: new Float32Array([1.0]), index: 2 })
}

// --- NorMuon variance reduction ---
// Computes per-row (or per-col) second moment EMA, normalizes g to preserve
// the overall gradient norm while reducing per-direction variance.
// Reads GPU data via unified memory after all GPU ops complete (endSync ensures this).

function norMuonScale(g, secondMomentBuf, beta2, rows, cols) {
  const tall = rows >= cols
  const redDim = tall ? cols : rows  // reduce along this dim
  const otherDim = tall ? rows : cols
  const d = g.data
  const smBuf = secondMomentBuf.data

  // Compute per-row (or per-col) mean of squares
  const vMean = new Float32Array(otherDim)

  if (tall) {
    for (let r = 0; r < rows; r++) {
      let sum = 0
      const base = r * cols
      for (let c = 0; c < cols; c++) { const v = d[base + c]; sum += v * v }
      vMean[r] = sum / cols
    }
  } else {
    for (let c = 0; c < cols; c++) {
      let sum = 0
      for (let r = 0; r < rows; r++) { const v = d[r * cols + c]; sum += v * v }
      vMean[c] = sum / rows
    }
  }

  // v_norm = sqrt(sum(vMean * redDim))
  let vNormSq = 0
  for (let i = 0; i < otherDim; i++) vNormSq += vMean[i] * redDim
  const vNorm = Math.sqrt(vNormSq)

  // EMA update
  for (let i = 0; i < otherDim; i++) {
    smBuf[i] = beta2 * smBuf[i] + (1 - beta2) * vMean[i]
  }

  // step_size = rsqrt(clamp(secondMomentBuf, 1e-10))
  const stepSize = new Float32Array(otherDim)
  for (let i = 0; i < otherDim; i++) {
    stepSize[i] = 1.0 / Math.sqrt(Math.max(smBuf[i], 1e-10))
  }

  // v_norm_new = sqrt(sum(vMean * redDim * stepSize^2))
  let vNormNewSq = 0
  for (let i = 0; i < otherDim; i++) {
    vNormNewSq += vMean[i] * redDim * stepSize[i] * stepSize[i]
  }
  const vNormNew = Math.sqrt(Math.max(vNormNewSq, 1e-10))

  const globalScale = vNorm / vNormNew

  // Apply scale to g via CPU (after all GPU ops, unified memory is synced)
  if (tall) {
    for (let r = 0; r < rows; r++) {
      const s = stepSize[r] * globalScale
      const base = r * cols
      for (let c = 0; c < cols; c++) d[base + c] *= s
    }
  } else {
    for (let c = 0; c < cols; c++) {
      const s = stepSize[c] * globalScale
      for (let r = 0; r < rows; r++) d[r * cols + c] *= s
    }
  }
}

// --- MuonAdamW optimizer ---

function createMuonAdamW(groups) {
  const state = new Map()

  for (const group of groups) {
    for (const p of group.params) {
      if (group.kind === 'adamw') {
        state.set(p, {
          step: 0,
          m: T.zeros(p.data.shape, p.data.dtype),
          v: T.zeros(p.data.shape, p.data.dtype),
        })
      } else if (group.kind === 'muon') {
        const [rows, cols] = p.data.shape
        const tall = rows >= cols
        const otherDim = tall ? rows : cols
        state.set(p, {
          momentumBuf: T.zeros(p.data.shape, p.data.dtype),
          secondMomentBuf: { data: new Float32Array(otherDim) },
        })
      }
    }
  }

  return { groups, state, _adamwStep: 0 }
}

function muonAdamWStep(opt) {
  // Increment AdamW step counter once per optimizer step, not per group
  opt._adamwStep++
  for (const group of opt.groups) {
    if (group.kind === 'adamw') {
      stepAdamW(opt, group)
    } else if (group.kind === 'muon') {
      stepMuon(opt, group)
    }
  }
}

// --- AdamW step (reuses existing adam.metal kernel) ---

function stepAdamW(opt, group) {
  const { lr = 1e-3, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0.01 } = group
  const step = opt._adamwStep
  const bc1 = 1 - Math.pow(betas[0], step)
  const bc2 = 1 - Math.pow(betas[1], step)

  for (const p of group.params) {
    if (!p.grad) continue
    const s = opt.state.get(p)
    const n = p.data.size

    const buf = new ArrayBuffer(32)
    const f = new Float32Array(buf, 0, 7)
    const u = new Uint32Array(buf, 28, 1)
    f[0] = lr; f[1] = betas[0]; f[2] = betas[1]
    f[3] = eps; f[4] = weightDecay; f[5] = bc1; f[6] = bc2
    u[0] = n

    run('adamw_step', [
      { buffer: p.data.buffer, index: 0 },
      { buffer: p.grad.buffer, index: 1 },
      { buffer: s.m.buffer, index: 2 },
      { buffer: s.v.buffer, index: 3 },
    ], { x: n }, { x: Math.min(n, GROUP_1D) },
    { data: new Uint8Array(buf), index: 4 })
  }
}

// --- Muon step ---

function stepMuon(opt, group) {
  const {
    lr = 0.02,
    momentum = 0.95,
    beta2 = 0.7,
    weightDecay = 0.0,
    nsSteps = 5,
  } = group

  for (const p of group.params) {
    if (!p.grad) continue
    const s = opt.state.get(p)
    const [rows, cols] = p.data.shape

    // Work on a copy of the gradient (Nesterov modifies it in-place)
    const g = T.create(p.data.shape)
    // GPU copy: g = p.grad * 1.0
    run('elementwise_scale', [
      { buffer: p.grad.buffer, index: 0 },
      { buffer: g.buffer, index: 1 },
    ], { x: g.size }, null,
    { data: new Float32Array([1.0]), index: 2 })

    // 1. Nesterov momentum
    nesterovMomentum(g, s.momentumBuf, momentum)

    // 2. Newton-Schulz orthogonalization
    newtonSchulz(g, nsSteps)

    // 3. NorMuon variance reduction (if beta2 > 0)
    // NorMuon reads g.data on CPU — all prior GPU ops have completed (sync dispatch)
    if (beta2 > 0) {
      norMuonScale(g, s.secondMomentBuf, beta2, rows, cols)
    }

    // 4. Cautious weight decay + parameter update
    // lr scaled by sqrt(max(1, rows/cols)) as in reference
    const scaledLr = lr * Math.sqrt(Math.max(1.0, rows / cols))
    muonUpdate(p.data, g, scaledLr, weightDecay)
  }
}

export {
  createMuonAdamW,
  muonAdamWStep,
  POLAR_COEFFS,
}
