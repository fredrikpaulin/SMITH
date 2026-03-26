// smith/src/ops/sampling.js
// GPU dispatch for sampling operations: argmax, penalties, top-k, multinomial.
// Full pipeline: gpuSample(logits, config) returns a single token index.
// Only 4 bytes cross the GPU→CPU boundary per token.

import * as T from '../tensor.js'
import * as device from '../device.js'
import { run, GROUP_1D } from '../dispatch.js'
import { isProfilingEnabled, recordKernel } from '../profile.js'
import { softmax } from './softmax.js'

// dispatch with multiple constant-buffer params (run() only supports one)
function runMultiParams(kernel, buffers, grid, group, paramsList) {
  const pso = device.pipeline(kernel)
  const enc = device.begin()
  device.setPipeline(enc, pso)
  for (const b of buffers) device.setBuffer(enc, b.buffer, b.index)
  for (const p of paramsList) device.setBytes(enc, p.data, p.data.byteLength, p.index)
  const gx = grid.x || 1, gy = grid.y || 1, gz = grid.z || 1
  const grpX = group?.x || Math.min(gx, GROUP_1D), grpY = group?.y || 1, grpZ = group?.z || 1
  device.dispatch(enc, gx, gy, gz, grpX, grpY, grpZ)
  if (isProfilingEnabled()) {
    const timing = device.endTimed(enc)
    recordKernel(kernel, timing.gpuMs)
  } else {
    device.endSync(enc)
  }
}

// Create a small scratch tensor, return both the tensor and a Uint32Array view of its data.
function u32Scratch(n) {
  const t = T.create([n])  // f32 buffer, 4 bytes per element = same size as uint32
  const view = new Uint32Array(t.data.buffer, t.data.byteOffset, n)
  return { tensor: t, u32: view }
}

// --- Argmax: parallel reduction, returns token index ---
function gpuArgmax(logits) {
  const size = logits.size
  const tpg = 1 << Math.ceil(Math.log2(Math.max(Math.min(size, GROUP_1D), 2)))
  const numGroups = Math.ceil(size / tpg)

  const partial = u32Scratch(numGroups)
  const partialValues = T.create([numGroups])

  run('argmax_reduce', [
    { buffer: logits.buffer, index: 0 },
    { buffer: partial.tensor.buffer, index: 1 },
    { buffer: partialValues.buffer, index: 2 },
  ], { x: numGroups * tpg }, { x: tpg },
  { data: new Uint32Array([size]), index: 3 })

  if (numGroups === 1) return partial.u32[0]

  // Pass 2
  const result = u32Scratch(1)
  const finalTpg = 1 << Math.ceil(Math.log2(Math.max(numGroups, 2)))

  run('argmax_reduce_final', [
    { buffer: partial.tensor.buffer, index: 0 },
    { buffer: partialValues.buffer, index: 1 },
    { buffer: result.tensor.buffer, index: 2 },
  ], { x: finalTpg }, { x: finalTpg },
  { data: new Uint32Array([numGroups]), index: 3 })

  return result.u32[0]
}

// --- Repetition penalty (in-place on logits) ---
function gpuApplyRepPenalty(logits, recentTokenIds, penalty) {
  if (penalty <= 1 || recentTokenIds.length === 0) return
  const n = recentTokenIds.length

  // Write recent token IDs into a GPU buffer
  const recent = u32Scratch(n)
  recent.u32.set(new Uint32Array(recentTokenIds))

  // PenaltyParams struct: { vocabSize: u32, numRecent: u32, temperature: f32, repPenalty: f32 }
  const params = new ArrayBuffer(16)
  new Uint32Array(params, 0, 2).set([logits.size, n])
  new Float32Array(params, 8, 2).set([0, penalty])

  run('apply_rep_penalty', [
    { buffer: logits.buffer, index: 0 },
    { buffer: recent.tensor.buffer, index: 1 },
  ], { x: n }, { x: Math.min(n, GROUP_1D) },
  { data: new Uint8Array(params), index: 2 })
}

// --- Temperature scaling (in-place) ---
function gpuApplyTemperature(logits, temperature) {
  if (temperature === 1) return

  runMultiParams('apply_temperature', [
    { buffer: logits.buffer, index: 0 },
  ], { x: logits.size }, { x: Math.min(logits.size, GROUP_1D) }, [
    { data: new Uint32Array([logits.size]), index: 1 },
    { data: new Float32Array([1 / temperature]), index: 2 },
  ])
}

// --- Top-K: find threshold + mask ---
function gpuApplyTopK(logits, k) {
  if (k <= 0 || k >= logits.size) return

  const threshold = T.create([1])

  run('topk_find_threshold', [
    { buffer: logits.buffer, index: 0 },
    { buffer: threshold.buffer, index: 1 },
  ], { x: 1 }, { x: 1 },
  { data: new Uint32Array([logits.size, k]), index: 2 })

  run('topk_mask', [
    { buffer: logits.buffer, index: 0 },
    { buffer: threshold.buffer, index: 1 },
  ], { x: logits.size }, { x: Math.min(logits.size, GROUP_1D) },
  { data: new Uint32Array([logits.size]), index: 2 })
}

// --- Top-P (nucleus) on CPU post-softmax ---
// After top-K filtering, the active set is small. Sequential cumulative sum
// is fast enough that GPU overhead isn't worth it.
function cpuApplyTopP(probs, p) {
  if (p >= 1) return

  const data = probs.data
  const n = probs.size
  const indices = Array.from({ length: n }, (_, i) => i)
  indices.sort((a, b) => data[b] - data[a])

  let cumulative = 0
  const kept = new Set()
  for (const i of indices) {
    cumulative += data[i]
    kept.add(i)
    if (cumulative >= p) break
  }

  // Zero out non-kept positions, renormalize
  let sum = 0
  for (let i = 0; i < n; i++) {
    if (!kept.has(i)) data[i] = 0
    else sum += data[i]
  }
  if (sum > 0) {
    const inv = 1 / sum
    for (let i = 0; i < n; i++) data[i] *= inv
  }
}

// --- Multinomial sample on GPU ---
function gpuMultinomialSample(probs, randomValue) {
  const result = u32Scratch(1)

  // SampleParams: { size: u32, randomValue: f32 }
  const params = new ArrayBuffer(8)
  new Uint32Array(params, 0, 1)[0] = probs.size
  new Float32Array(params, 4, 1)[0] = randomValue

  run('multinomial_sample', [
    { buffer: probs.buffer, index: 0 },
    { buffer: result.tensor.buffer, index: 1 },
  ], { x: 1 }, { x: 1 },
  { data: new Uint8Array(params), index: 2 })

  return result.u32[0]
}

// ============================================================
// gpuSample — full GPU sampling pipeline.
// Input: logits tensor [vocabSize], sampling config.
// Output: single token index (uint32).
//
// Pipeline: penalties → temperature → top-k → softmax → top-p → sample
// For greedy (temperature=0): penalties → argmax
// ============================================================

function gpuSample(logitsTensor, config = {}) {
  const {
    temperature = 1.0,
    topK = 0,
    topP = 1.0,
    repetitionPenalty = 1.0,
    recentTokens = [],
  } = config

  // Apply repetition penalty first (affects all paths)
  if (repetitionPenalty > 1 && recentTokens.length > 0) {
    gpuApplyRepPenalty(logitsTensor, recentTokens, repetitionPenalty)
  }

  // Greedy: argmax, skip everything else
  if (temperature === 0) return gpuArgmax(logitsTensor)

  // Stochastic path
  gpuApplyTemperature(logitsTensor, temperature)
  gpuApplyTopK(logitsTensor, topK)

  // Softmax → probabilities (existing GPU kernel, treats as 1 row of vocabSize)
  const probs = softmax(logitsTensor)

  // Top-P on CPU (sequential, fast after top-K filtering)
  cpuApplyTopP(probs, topP)

  // Sample
  return gpuMultinomialSample(probs, Math.random())
}

export {
  gpuArgmax,
  gpuApplyRepPenalty,
  gpuApplyTemperature,
  gpuApplyTopK,
  cpuApplyTopP,
  gpuMultinomialSample,
  gpuSample,
}
