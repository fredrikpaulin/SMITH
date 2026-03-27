// examples/tts/predictor.js
// Qwen3-TTS Code Predictor: 5-layer transformer that generates groups 1-15.
// At each Talker step, the Code Predictor takes the talker hidden state +
// group-0 code embedding, and autoregressively generates 15 more codes.

import smith from '../../src/index.js'
import { rmsnormForward } from '../../src/ops/rmsnorm.js'
import { swigluForward } from '../../src/ops/swiglu.js'
import { precomputeRoPE } from '../../src/ops/rope.js'
import {
  linearNoBias,
  linearBias,
  addTensors,
  reshapeToHeads,
  reshapeFromHeads,
  repeatKV,
  sampleToken,
  qkNorm,
  ropePerHead,
} from './talker.js'

// Create KV cache for the Code Predictor
function createPredictorCache(config) {
  const maxSeq = 20
  const caches = []
  for (let i = 0; i < config.numLayers; i++) {
    caches.push({
      k: smith.zeros([config.numKVHeads, maxSeq, config.headDim]),
      v: smith.zeros([config.numKVHeads, maxSeq, config.headDim]),
      len: 0,
    })
  }
  return caches
}

// Cache ops
function cacheAppend(cache, kNew, vNew) {
  const pos = cache.len
  const kvHeads = kNew.shape[0]
  const headDim = kNew.shape[2]
  const maxSeq = cache.k.shape[1]

  for (let h = 0; h < kvHeads; h++) {
    const srcOff = h * headDim
    const dstOff = h * maxSeq * headDim + pos * headDim
    cache.k.data.set(kNew.data.subarray(srcOff, srcOff + headDim), dstOff)
    cache.v.data.set(vNew.data.subarray(srcOff, srcOff + headDim), dstOff)
  }
  cache.len = pos + 1
}

function cacheSlice(cache) {
  const len = cache.len
  const kvHeads = cache.k.shape[0]
  const headDim = cache.k.shape[2]
  const maxSeq = cache.k.shape[1]

  const kSlice = smith.zeros([kvHeads, len, headDim])
  const vSlice = smith.zeros([kvHeads, len, headDim])

  for (let h = 0; h < kvHeads; h++) {
    const srcOff = h * maxSeq * headDim
    const dstOff = h * len * headDim
    kSlice.data.set(cache.k.data.subarray(srcOff, srcOff + len * headDim), dstOff)
    vSlice.data.set(cache.v.data.subarray(srcOff, srcOff + len * headDim), dstOff)
  }
  return { k: kSlice, v: vSlice }
}

// Single block forward (decode mode, single token)
function predictorBlockDecode(x, block, cache, position, model) {
  const { dim, numHeads, numKVHeads, headDim } = model.config

  const norm1 = rmsnormForward(x, block.inputLayernorm, model.config.normEps)

  let Q = linearNoBias(norm1, block.qProj)
  let K = linearNoBias(norm1, block.kProj)
  const V = linearNoBias(norm1, block.vProj)

  // QK norm
  Q = qkNorm(Q, block.qNorm, numHeads, headDim, model.config.normEps)
  K = qkNorm(K, block.kNorm, numKVHeads, headDim, model.config.normEps)

  // Reshape first, then RoPE per-head
  const Qh = reshapeToHeads(Q, 1, numHeads, headDim)
  const Kh = reshapeToHeads(K, 1, numKVHeads, headDim)
  const Vh = reshapeToHeads(V, 1, numKVHeads, headDim)

  ropePerHead(Qh, numHeads, 1, headDim, model.rope, position)
  ropePerHead(Kh, numKVHeads, 1, headDim, model.rope, position)

  cacheAppend(cache, Kh, Vh)
  const { k: cachedK, v: cachedV } = cacheSlice(cache)
  const fullK = repeatKV(cachedK, numHeads, numKVHeads)
  const fullV = repeatKV(cachedV, numHeads, numKVHeads)

  // Attention
  const seqLen = cache.len
  const scale = 1 / Math.sqrt(headDim)
  const scores = smith.zeros([numHeads, 1, seqLen])
  for (let h = 0; h < numHeads; h++) {
    for (let s = 0; s < seqLen; s++) {
      let dot = 0
      for (let d = 0; d < headDim; d++) {
        dot += Qh.data[h * headDim + d] * fullK.data[h * seqLen * headDim + s * headDim + d]
      }
      scores.data[h * seqLen + s] = dot * scale
    }
  }

  for (let h = 0; h < numHeads; h++) {
    const off = h * seqLen
    let max = -Infinity
    for (let s = 0; s < seqLen; s++) max = Math.max(max, scores.data[off + s])
    let sum = 0
    for (let s = 0; s < seqLen; s++) {
      scores.data[off + s] = Math.exp(scores.data[off + s] - max)
      sum += scores.data[off + s]
    }
    for (let s = 0; s < seqLen; s++) scores.data[off + s] /= sum
  }

  const attnOut = smith.zeros([1, numHeads * headDim])
  for (let h = 0; h < numHeads; h++) {
    for (let d = 0; d < headDim; d++) {
      let acc = 0
      for (let s = 0; s < seqLen; s++) {
        acc += scores.data[h * seqLen + s] * fullV.data[h * seqLen * headDim + s * headDim + d]
      }
      attnOut.data[h * headDim + d] = acc
    }
  }

  const projected = linearNoBias(attnOut, block.oProj)
  const x2 = addTensors(x, projected)

  const norm2 = rmsnormForward(x2, block.postAttnLayernorm, model.config.normEps)
  const gate = linearNoBias(norm2, block.gateProj)
  const up = linearNoBias(norm2, block.upProj)
  const fused = swigluForward(gate, up)
  const ffnOut = linearNoBias(fused, block.downProj)

  return addTensors(x2, ffnOut)
}

// Project talker hidden (2048) → predictor space (1024) using bias
function projectHidden(hidden, projection, projectionBias) {
  return linearBias(hidden, projection, projectionBias)
}

// Prepare the predictor model
function preparePredictor(predictor) {
  predictor.rope = precomputeRoPE(predictor.config.headDim, 64, predictor.config.ropeTheta)
  return predictor
}

// Generate codes for groups 1-15 given the talker hidden state and group-0 code.
// Reference: predictor receives [talker_hidden, group0_embed] as 2-position prefix,
// then autoregressively generates 15 codes (one per group).
function predictCodes(predictor, talkerHidden, group0CodeId, talkerCodecEmbedding, config = {}) {
  const {
    temperature = 0.9,
    topK = 50,
  } = config

  const caches = createPredictorCache(predictor.config)

  // Position 0: projected talker hidden
  const projectedHidden = projectHidden(talkerHidden, predictor.projection, predictor.projectionBias)
  let x = projectedHidden
  for (let i = 0; i < predictor.blocks.length; i++) {
    x = predictorBlockDecode(x, predictor.blocks[i], caches[i], 0, predictor)
  }

  // Position 1: projected group-0 code embedding (from talker's codec_embedding)
  const group0Embed = embedCode(group0CodeId, talkerCodecEmbedding)
  const projectedGroup0 = projectHidden(group0Embed, predictor.projection, predictor.projectionBias)
  x = projectedGroup0
  for (let i = 0; i < predictor.blocks.length; i++) {
    x = predictorBlockDecode(x, predictor.blocks[i], caches[i], 1, predictor)
  }
  x = rmsnormForward(x, predictor.norm, predictor.config.normEps)

  // Generate group 1 code from position 1 output
  let logits = linearNoBias(x, predictor.lmHeads[0])
  let code = sampleToken(logits, temperature, topK)
  const codes = [code]

  // Generate groups 2-15 (positions 2..16)
  for (let g = 1; g < 15; g++) {
    // Embed previous code with that group's predictor embedding, then project to predictor dim
    let codeEmbed = embedCode(code, predictor.codecEmbeddings[g - 1])
    codeEmbed = projectHidden(codeEmbed, predictor.projection, predictor.projectionBias)

    x = codeEmbed
    for (let i = 0; i < predictor.blocks.length; i++) {
      x = predictorBlockDecode(x, predictor.blocks[i], caches[i], g + 1, predictor)
    }
    x = rmsnormForward(x, predictor.norm, predictor.config.normEps)
    logits = linearNoBias(x, predictor.lmHeads[g])
    code = sampleToken(logits, temperature, topK)
    codes.push(code)
  }

  return codes
}

// Embed a single code ID using an embedding weight matrix
function embedCode(codeId, embeddingWeight) {
  const dim = embeddingWeight.shape[1]
  const out = smith.zeros([1, dim])
  if (codeId >= 0 && codeId < embeddingWeight.shape[0]) {
    out.data.set(embeddingWeight.data.subarray(codeId * dim, (codeId + 1) * dim))
  }
  return out
}

export { preparePredictor, predictCodes, embedCode }
