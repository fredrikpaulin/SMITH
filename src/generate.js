// smith/src/generate.js
// Autoregressive text generation with sampling strategies.
// Ported from TinyFormer's generate.js.
// Temperature, top-k, top-p (nucleus), repetition penalty.

import * as A from './autograd.js'
import { forward, forwardCached } from './model.js'

// --- Sampling strategies (all pure JS, no GPU needed) ---

function applyTemperature(logits, temperature) {
  if (temperature === 0 || temperature === 1) return logits
  return logits.map(v => v / temperature)
}

function applyTopK(logits, k) {
  if (k <= 0 || k >= logits.length) return logits
  const sorted = Array.from(logits).sort((a, b) => b - a)
  const threshold = sorted[k - 1]
  return logits.map(v => v >= threshold ? v : -Infinity)
}

function applyTopP(logits, p) {
  if (p >= 1) return logits
  const indices = logits.map((_, i) => i)
  indices.sort((a, b) => logits[b] - logits[a])

  const maxLogit = Math.max(...logits)
  const exps = logits.map(v => Math.exp(v - maxLogit))
  const sumExp = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map(v => v / sumExp)

  let cumulative = 0
  const kept = new Set()
  for (const i of indices) {
    cumulative += probs[i]
    kept.add(i)
    if (cumulative >= p) break
  }
  return logits.map((v, i) => kept.has(i) ? v : -Infinity)
}

function applyRepetitionPenalty(logits, generatedIds, penalty) {
  if (penalty <= 1) return logits
  const result = Array.from(logits)
  const seen = new Set(generatedIds)
  for (const id of seen) {
    if (id < result.length) {
      result[id] = result[id] > 0 ? result[id] / penalty : result[id] * penalty
    }
  }
  return result
}

function sampleFromLogits(logits) {
  const maxLogit = Math.max(...logits)
  const exps = logits.map(v => Math.exp(v - maxLogit))
  const sumExp = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map(v => v / sumExp)

  const r = Math.random()
  let cumulative = 0
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i]
    if (r < cumulative) return i
  }
  return probs.length - 1
}

function argmax(logits) {
  let best = 0
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i
  }
  return best
}

// --- Generation (full context, no KV cache) ---

function generate(model, promptIds, config = {}, callbacks = {}) {
  const {
    maxTokens = 50,
    temperature = 1.0,
    topK = 0,
    topP = 1.0,
    repetitionPenalty = 1.0,
  } = config

  const maxSeqLen = model.config.maxSeqLen
  const vocabSize = model.config.vocabSize
  const generated = [...promptIds]

  for (let i = 0; i < maxTokens; i++) {
    const context = generated.length > maxSeqLen
      ? generated.slice(-maxSeqLen)
      : generated

    let logits
    A.noGrad(() => {
      const result = forward(model, context)
      logits = result.logits
    })

    // Get logits for the last position
    const lastPos = logits.data.shape[0] - 1
    let lastLogits = Array.from(logits.data.data.slice(lastPos * vocabSize, (lastPos + 1) * vocabSize))

    lastLogits = applyRepetitionPenalty(lastLogits, generated, repetitionPenalty)
    lastLogits = applyTemperature(lastLogits, temperature)
    lastLogits = applyTopK(lastLogits, topK)
    lastLogits = applyTopP(lastLogits, topP)

    const nextToken = temperature === 0 ? argmax(lastLogits) : sampleFromLogits(lastLogits)
    generated.push(nextToken)

    if (callbacks.onToken) {
      const shouldStop = callbacks.onToken(nextToken, generated.length - promptIds.length)
      if (shouldStop) break
    }
  }

  return generated
}

// --- Top-k predictions ---

function topKPredictions(model, tokenIds, k = 10) {
  let logits
  A.noGrad(() => {
    const result = forward(model, tokenIds)
    logits = result.logits
  })

  const vocabSize = model.config.vocabSize
  const lastPos = logits.data.shape[0] - 1
  const lastLogits = Array.from(logits.data.data.slice(lastPos * vocabSize, (lastPos + 1) * vocabSize))

  const maxL = Math.max(...lastLogits)
  const exps = lastLogits.map(v => Math.exp(v - maxL))
  const sumExp = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map(v => v / sumExp)

  return probs.map((p, id) => ({ id, prob: p }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, k)
}

// --- Generation with KV cache (O(1) per token after prompt) ---

function generateCached(model, promptIds, config = {}, callbacks = {}) {
  const {
    maxTokens = 50,
    temperature = 1.0,
    topK = 0,
    topP = 1.0,
    repetitionPenalty = 1.0,
  } = config

  const maxSeqLen = model.config.maxSeqLen
  const vocabSize = model.config.vocabSize
  const generated = [...promptIds]
  let kvCaches = null

  // Phase 1: Process prompt tokens one by one to fill the cache
  const promptToProcess = generated.length > maxSeqLen
    ? generated.slice(-maxSeqLen)
    : [...generated]
  const promptLen = promptToProcess.length

  for (let i = 0; i < promptLen; i++) {
    A.noGrad(() => {
      const result = forwardCached(model, promptToProcess[i], i, kvCaches)
      kvCaches = result.newCaches
      // Only sample from the last prompt token
      if (i === promptLen - 1) {
        let lastLogits = Array.from(result.logits.data.data.slice(0, vocabSize))
        lastLogits = applyRepetitionPenalty(lastLogits, generated, repetitionPenalty)
        lastLogits = applyTemperature(lastLogits, temperature)
        lastLogits = applyTopK(lastLogits, topK)
        lastLogits = applyTopP(lastLogits, topP)
        const nextToken = temperature === 0 ? argmax(lastLogits) : sampleFromLogits(lastLogits)
        generated.push(nextToken)
      }
    })
  }

  if (callbacks.onToken) {
    const shouldStop = callbacks.onToken(generated[generated.length - 1], 1)
    if (shouldStop) return generated
  }

  // Phase 2: Generate remaining tokens one at a time using cache
  for (let i = 1; i < maxTokens; i++) {
    const position = promptLen + i - 1
    if (position >= maxSeqLen) break

    const lastToken = generated[generated.length - 1]
    let nextToken

    A.noGrad(() => {
      const result = forwardCached(model, lastToken, position, kvCaches)
      kvCaches = result.newCaches
      let lastLogits = Array.from(result.logits.data.data.slice(0, vocabSize))
      lastLogits = applyRepetitionPenalty(lastLogits, generated, repetitionPenalty)
      lastLogits = applyTemperature(lastLogits, temperature)
      lastLogits = applyTopK(lastLogits, topK)
      lastLogits = applyTopP(lastLogits, topP)
      nextToken = temperature === 0 ? argmax(lastLogits) : sampleFromLogits(lastLogits)
    })

    generated.push(nextToken)

    if (callbacks.onToken) {
      const shouldStop = callbacks.onToken(nextToken, i + 1)
      if (shouldStop) break
    }
  }

  return generated
}

export {
  generate, generateCached, topKPredictions,
  applyTemperature, applyTopK, applyTopP, applyRepetitionPenalty,
  sampleFromLogits, argmax,
}
