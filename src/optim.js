// smith/src/optim.js
// AdamW optimizer with fused GPU parameter update.
// Cosine LR schedule with linear warmup.
// Gradient clipping via global norm.
// Ported from TinyFormer's train.js.

import * as T from './tensor.js'
import { run, GROUP_1D } from './dispatch.js'

// --- AdamW Optimizer ---

function createAdamW(params, config = {}) {
  const { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01 } = config

  // Allocate moment buffers on GPU for each param
  const state = params.map(p => ({
    m: T.zeros(p.data.shape, p.data.dtype), // first moment
    v: T.zeros(p.data.shape, p.data.dtype), // second moment
  }))

  return { params, state, lr, beta1, beta2, eps, weightDecay, step: 0 }
}

// Build the AdamParams struct for the shader (8 floats = 32 bytes)
function adamParams(opt, n) {
  const bc1 = 1 - Math.pow(opt.beta1, opt.step)
  const bc2 = 1 - Math.pow(opt.beta2, opt.step)
  const buf = new ArrayBuffer(32)
  const f = new Float32Array(buf, 0, 7)
  const u = new Uint32Array(buf, 28, 1)
  f[0] = opt.lr
  f[1] = opt.beta1
  f[2] = opt.beta2
  f[3] = opt.eps
  f[4] = opt.weightDecay
  f[5] = bc1
  f[6] = bc2
  u[0] = n
  return new Uint8Array(buf)
}

function adamwStep(opt) {
  opt.step++

  for (let i = 0; i < opt.params.length; i++) {
    const p = opt.params[i]
    if (!p.grad) continue
    const s = opt.state[i]
    const n = p.data.size

    const params = adamParams(opt, n)

    run('adamw_step', [
      { buffer: p.data.buffer, index: 0 },
      { buffer: p.grad.buffer, index: 1 },
      { buffer: s.m.buffer, index: 2 },
      { buffer: s.v.buffer, index: 3 },
    ], { x: n }, { x: Math.min(n, GROUP_1D) },
    { data: params, index: 4 })
  }
}

// --- Learning Rate Schedule ---

function createSchedule(config = {}) {
  const { warmupSteps = 100, totalSteps = 1000, maxLr = 1e-3, minLr = 1e-5 } = config
  return { warmupSteps, totalSteps, maxLr, minLr }
}

function getLr(schedule, step) {
  const { warmupSteps, totalSteps, maxLr, minLr } = schedule
  if (step < warmupSteps) return maxLr * (step + 1) / warmupSteps
  const progress = (step - warmupSteps) / Math.max(1, totalSteps - warmupSteps)
  const cosDecay = 0.5 * (1 + Math.cos(Math.PI * Math.min(progress, 1)))
  return minLr + (maxLr - minLr) * cosDecay
}

// --- Gradient Clipping ---
// Reads grad data directly via unified memory (zero-copy).

function clipGradNorm(params, maxNorm) {
  let totalNormSq = 0
  for (const p of params) {
    if (!p.grad) continue
    const d = p.grad.data
    for (let i = 0; i < d.length; i++) totalNormSq += d[i] * d[i]
  }
  const totalNorm = Math.sqrt(totalNormSq)
  if (totalNorm > maxNorm) {
    const s = maxNorm / totalNorm
    for (const p of params) {
      if (!p.grad) continue
      const d = p.grad.data
      for (let i = 0; i < d.length; i++) d[i] *= s
    }
  }
  return totalNorm
}

// --- Zero Grad ---

function zeroGrad(params) {
  for (const p of params) p.grad = null
}

export {
  createAdamW, adamwStep,
  createSchedule, getLr,
  clipGradNorm, zeroGrad,
}
