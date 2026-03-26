// smith/src/gguf_cache.js
// KV cache for GGUF models (Llama, Phi). Supports GQA (fewer KV heads than Q heads).
// Pre-allocates fixed-size buffers and tracks current position.

import * as T from './tensor.js'
import * as A from './autograd.js'
import { linear } from './nn.js'
import { ropeForward } from './ops/rope.js'
import { rmsnormForward } from './ops/rmsnorm.js'
import { swigluForward } from './ops/swiglu.js'

// Allocate KV cache for a GGUF model
// Returns array of { k, v, len } per layer
function createGGUFCache(config) {
  const { numLayers, numKVHeads, maxSeqLen } = config
  const numHeads = numKVHeads ?? config.numHeads
  const headDim = config.dim / config.numHeads

  const caches = []
  for (let i = 0; i < numLayers; i++) {
    caches.push({
      k: T.zeros([numHeads, maxSeqLen, headDim]),  // [kvHeads, maxSeq, headDim]
      v: T.zeros([numHeads, maxSeqLen, headDim]),
      len: 0,
    })
  }
  return caches
}

// Reset cache (reuse buffers, just zero position)
function resetCache(caches) {
  for (const c of caches) c.len = 0
}

// Write a single position's K/V into the cache
// kNew, vNew: tensors [kvHeads, 1, headDim]
function cacheAppend(cache, kNew, vNew) {
  const pos = cache.len
  const kvHeads = kNew.shape[0]
  const headDim = kNew.shape[2]

  for (let h = 0; h < kvHeads; h++) {
    const srcKOff = h * headDim
    const srcVOff = h * headDim
    const dstOff = h * cache.k.shape[1] * headDim + pos * headDim
    cache.k.data.set(kNew.data.subarray(srcKOff, srcKOff + headDim), dstOff)
    cache.v.data.set(vNew.data.subarray(srcVOff, srcVOff + headDim), dstOff)
  }
  cache.len = pos + 1
}

// Write multiple positions' K/V into the cache (for prefill)
// kNew, vNew: tensors [kvHeads, seqLen, headDim]
function cachePrefill(cache, kNew, vNew, seqLen) {
  const kvHeads = kNew.shape[0]
  const headDim = kNew.shape[2]
  const maxSeqLen = cache.k.shape[1]

  for (let h = 0; h < kvHeads; h++) {
    const srcOff = h * seqLen * headDim
    const dstOff = h * maxSeqLen * headDim
    cache.k.data.set(kNew.data.subarray(srcOff, srcOff + seqLen * headDim), dstOff)
    cache.v.data.set(vNew.data.subarray(srcOff, srcOff + seqLen * headDim), dstOff)
  }
  cache.len = seqLen
}

// Get a view of the cached K/V up to current length
// Returns tensors [kvHeads, cachedLen, headDim]
function cacheSlice(cache) {
  const len = cache.len
  const kvHeads = cache.k.shape[0]
  const headDim = cache.k.shape[2]
  const maxSeqLen = cache.k.shape[1]

  const kSlice = T.create([kvHeads, len, headDim], cache.k.dtype)
  const vSlice = T.create([kvHeads, len, headDim], cache.v.dtype)

  for (let h = 0; h < kvHeads; h++) {
    const srcOff = h * maxSeqLen * headDim
    const dstOff = h * len * headDim
    kSlice.data.set(cache.k.data.subarray(srcOff, srcOff + len * headDim), dstOff)
    vSlice.data.set(cache.v.data.subarray(srcOff, srcOff + len * headDim), dstOff)
  }

  return { k: kSlice, v: vSlice }
}

// Repeat KV heads for GQA: [kvHeads, seqLen, headDim] → [numHeads, seqLen, headDim]
function repeatKV(kvTensor, numHeads, kvHeads) {
  if (kvHeads === numHeads) return kvTensor
  const repeats = numHeads / kvHeads
  const seqLen = kvTensor.shape[1]
  const headDim = kvTensor.shape[2]
  const out = T.create([numHeads, seqLen, headDim], kvTensor.dtype)

  for (let h = 0; h < kvHeads; h++) {
    const srcOff = h * seqLen * headDim
    for (let r = 0; r < repeats; r++) {
      const dstOff = (h * repeats + r) * seqLen * headDim
      out.data.set(kvTensor.data.subarray(srcOff, srcOff + seqLen * headDim), dstOff)
    }
  }
  return out
}

// --- Cached Llama forward: single token decode ---

function forwardLlamaCachedDecode(model, tokenId, position, caches) {
  const { dim, numHeads, headDim, numKVHeads } = model.config
  const kvHeads = numKVHeads ?? numHeads

  // Embed single token
  let x = A.embedding([tokenId], model.embedding.tokenWeight)  // [1, dim]

  for (let i = 0; i < model.blocks.length; i++) {
    const block = model.blocks[i]
    const cache = caches[i]

    // Pre-norm (RMSNorm)
    const normData = rmsnormForward(T.contiguous(x.data), T.contiguous(block.ln1Gamma.data), model.normEps)
    const norm1 = A.variable(normData, { requiresGrad: false })

    // Q, K, V projections
    const Q = linear(norm1, block.mha.qProj)  // [1, dim]
    const K = linear(norm1, block.mha.kProj)  // [1, kvDim]
    const V = linear(norm1, block.mha.vProj)  // [1, kvDim]

    // Apply RoPE at position offset
    const Qr = A.variable(ropeForward(T.contiguous(Q.data), model.rope, position), { requiresGrad: false })
    const Kr = A.variable(ropeForward(T.contiguous(K.data), model.rope, position), { requiresGrad: false })

    // Reshape to multi-head: [1, dim] → [numHeads, 1, headDim]
    const Qh = A.transpose(A.reshape(Qr, [1, numHeads, headDim]), [1, 0, 2])
    const Kh = A.transpose(A.reshape(Kr, [1, kvHeads, headDim]), [1, 0, 2])
    const Vh = A.transpose(A.reshape(V, [1, kvHeads, headDim]), [1, 0, 2])

    // Append to cache
    cacheAppend(cache, T.contiguous(Kh.data), T.contiguous(Vh.data))

    // Get full cached K/V
    const { k: cachedK, v: cachedV } = cacheSlice(cache)

    // GQA: repeat KV heads to match Q heads
    const fullK = A.variable(repeatKV(cachedK, numHeads, kvHeads), { requiresGrad: false })
    const fullV = A.variable(repeatKV(cachedV, numHeads, kvHeads), { requiresGrad: false })

    // Attention: Q [numHeads, 1, headDim] × K^T [numHeads, headDim, seqLen]
    const scaleFactor = 1 / Math.sqrt(headDim)
    const kAxes = [0, 2, 1]  // transpose last two dims of 3D tensor
    const scores = A.scale(A.matmul(Qh, A.transpose(fullK, kAxes)), scaleFactor)
    // No causal mask needed — Q is single position, all cached positions are valid
    const weights = A.softmax(scores, -1)
    const attnOut = A.matmul(weights, fullV)  // [numHeads, 1, headDim]

    // Concatenate heads: [numHeads, 1, headDim] → [1, dim]
    const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [1, dim])
    const projected = linear(concatenated, block.mha.outProj)
    const x2 = A.add(x, projected)

    // FFN with SwiGLU
    const norm2Data = rmsnormForward(T.contiguous(x2.data), T.contiguous(block.ln2Gamma.data), model.normEps)
    const norm2 = A.variable(norm2Data, { requiresGrad: false })

    const gateOut = linear(norm2, block.ffnGate)
    const upOut = linear(norm2, block.ffnUp)
    const fused = swigluForward(T.contiguous(gateOut.data), T.contiguous(upOut.data))
    const fusedVar = A.variable(fused, { requiresGrad: false })
    const ffnOut = linear(fusedVar, block.ffnDown)

    x = A.add(x2, ffnOut)
  }

  // Final norm
  const finalNorm = rmsnormForward(T.contiguous(x.data), T.contiguous(model.lnFGamma.data), model.normEps)
  x = A.variable(finalNorm, { requiresGrad: false })

  // Output projection
  let logits
  if (model.weightTied) {
    logits = A.matmul(x, A.transpose(model.embedding.tokenWeight))
  } else {
    logits = linear(x, { weight: model.lmHead, bias: null })
  }

  return { logits }
}

// --- Cached Llama forward: prefill (process full prompt at once) ---

function forwardLlamaCachedPrefill(model, tokenIds, caches) {
  const seqLen = tokenIds.length
  const { dim, numHeads, headDim, numKVHeads } = model.config
  const kvHeads = numKVHeads ?? numHeads

  // Embed all tokens
  let x = A.embedding(tokenIds, model.embedding.tokenWeight)  // [seqLen, dim]

  for (let i = 0; i < model.blocks.length; i++) {
    const block = model.blocks[i]
    const cache = caches[i]

    // Pre-norm
    const normData = rmsnormForward(T.contiguous(x.data), T.contiguous(block.ln1Gamma.data), model.normEps)
    const norm1 = A.variable(normData, { requiresGrad: false })

    // Q, K, V projections
    let Q = linear(norm1, block.mha.qProj)  // [seqLen, dim]
    let K = linear(norm1, block.mha.kProj)  // [seqLen, kvDim]
    let V = linear(norm1, block.mha.vProj)  // [seqLen, kvDim]

    // Apply RoPE (startPos = 0 for prefill)
    Q = A.variable(ropeForward(T.contiguous(Q.data), model.rope, 0), { requiresGrad: false })
    K = A.variable(ropeForward(T.contiguous(K.data), model.rope, 0), { requiresGrad: false })

    // Reshape: [seqLen, dim] → [numHeads, seqLen, headDim]
    const Qh = A.transpose(A.reshape(Q, [seqLen, numHeads, headDim]), [1, 0, 2])
    const Kh = A.transpose(A.reshape(K, [seqLen, kvHeads, headDim]), [1, 0, 2])
    const Vh = A.transpose(A.reshape(V, [seqLen, kvHeads, headDim]), [1, 0, 2])

    // Store in cache
    cachePrefill(cache, T.contiguous(Kh.data), T.contiguous(Vh.data), seqLen)

    // GQA: repeat KV heads
    const fullK = kvHeads < numHeads
      ? A.variable(repeatKV(T.contiguous(Kh.data), numHeads, kvHeads), { requiresGrad: false })
      : Kh
    const fullV = kvHeads < numHeads
      ? A.variable(repeatKV(T.contiguous(Vh.data), numHeads, kvHeads), { requiresGrad: false })
      : Vh

    // Flash attention (causal) for prefill — full sequence
    const attnOut = A.flashAttention(Qh, fullK, fullV, true)
    const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [seqLen, dim])
    const projected = linear(concatenated, block.mha.outProj)
    const x2 = A.add(x, projected)

    // FFN with SwiGLU
    const norm2Data = rmsnormForward(T.contiguous(x2.data), T.contiguous(block.ln2Gamma.data), model.normEps)
    const norm2 = A.variable(norm2Data, { requiresGrad: false })

    const gateOut = linear(norm2, block.ffnGate)
    const upOut = linear(norm2, block.ffnUp)
    const fused = swigluForward(T.contiguous(gateOut.data), T.contiguous(upOut.data))
    const fusedVar = A.variable(fused, { requiresGrad: false })
    const ffnOut = linear(fusedVar, block.ffnDown)

    x = A.add(x2, ffnOut)
  }

  // Final norm
  const finalNorm = rmsnormForward(T.contiguous(x.data), T.contiguous(model.lnFGamma.data), model.normEps)
  x = A.variable(finalNorm, { requiresGrad: false })

  // Output projection — only need logits for the last token
  // But return full for consistency, caller slices
  let logits
  if (model.weightTied) {
    logits = A.matmul(x, A.transpose(model.embedding.tokenWeight))
  } else {
    logits = linear(x, { weight: model.lmHead, bias: null })
  }

  return { logits }
}

// --- Generation with KV cache for GGUF models ---
// Prefill phase: process full prompt at once via flash attention
// Decode phase: generate tokens one at a time with cached K/V

import {
  applyTemperature, applyTopK, applyTopP,
  applyRepetitionPenalty, sampleFromLogits, argmax,
} from './generate.js'
import { gpuSample } from './ops/sampling.js'

// Extract logits for a single position as a 1D tensor [vocabSize].
// For GPU sampling, we need the tensor on the GPU — not copied to a JS array.
function extractLastLogits(logitsTensor, position, vocabSize) {
  const out = T.create([vocabSize], logitsTensor.dtype)
  const srcOff = position * vocabSize
  out.data.set(logitsTensor.data.subarray(srcOff, srcOff + vocabSize))
  return out
}

// Sample one token — GPU or CPU path
function sampleToken(logitsTensor, generated, config) {
  const { temperature, topK, topP, repetitionPenalty, gpuSampling } = config

  if (gpuSampling) {
    return gpuSample(logitsTensor, {
      temperature,
      topK,
      topP,
      repetitionPenalty,
      recentTokens: generated,
    })
  }

  // CPU path (original)
  let logits = Array.from(logitsTensor.data)
  logits = applyRepetitionPenalty(logits, generated, repetitionPenalty)
  logits = applyTemperature(logits, temperature)
  logits = applyTopK(logits, topK)
  logits = applyTopP(logits, topP)
  return temperature === 0 ? argmax(logits) : sampleFromLogits(logits)
}

function generateGGUF(model, promptIds, config = {}, callbacks = {}) {
  const {
    maxTokens = 50,
    temperature = 1.0,
    topK = 0,
    topP = 1.0,
    repetitionPenalty = 1.0,
    eosToken = null,
    gpuSampling = false,
  } = config

  const samplingConfig = { temperature, topK, topP, repetitionPenalty, gpuSampling }
  const vocabSize = model.config.vocabSize
  const maxSeqLen = model.config.maxSeqLen
  const generated = [...promptIds]

  // Truncate prompt if it exceeds max context
  const prompt = generated.length > maxSeqLen
    ? generated.slice(-maxSeqLen)
    : generated
  const promptLen = prompt.length

  // Allocate cache
  const caches = createGGUFCache(model.config)

  // Phase 1: Prefill — process entire prompt at once
  let nextToken
  A.noGrad(() => {
    const result = forwardLlamaCachedPrefill(model, prompt, caches)
    const lastLogits = extractLastLogits(result.logits.data, promptLen - 1, vocabSize)
    nextToken = sampleToken(lastLogits, generated, samplingConfig)
  })
  generated.push(nextToken)

  if (eosToken !== null && nextToken === eosToken) return generated
  if (callbacks.onToken) {
    const shouldStop = callbacks.onToken(nextToken, 1)
    if (shouldStop) return generated
  }

  // Phase 2: Decode — one token at a time using cached K/V
  for (let i = 1; i < maxTokens; i++) {
    const position = promptLen + i - 1
    if (position >= maxSeqLen) break

    const lastTok = generated[generated.length - 1]
    A.noGrad(() => {
      const result = forwardLlamaCachedDecode(model, lastTok, position, caches)
      const lastLogits = extractLastLogits(result.logits.data, 0, vocabSize)
      nextToken = sampleToken(lastLogits, generated, samplingConfig)
    })

    generated.push(nextToken)

    if (eosToken !== null && nextToken === eosToken) break
    if (callbacks.onToken) {
      const shouldStop = callbacks.onToken(nextToken, i + 1)
      if (shouldStop) break
    }
  }

  return generated
}

export {
  createGGUFCache, resetCache,
  cacheAppend, cachePrefill, cacheSlice, repeatKV,
  forwardLlamaCachedDecode, forwardLlamaCachedPrefill,
  generateGGUF,
}
