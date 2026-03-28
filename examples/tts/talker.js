// examples/tts/talker.js
// Qwen3-TTS Talker: 28-layer Llama-style transformer with KV cache.
// Generates group-0 codec tokens autoregressively from text input.
// Follows the gguf_cache.js pattern: prefill + cached decode.

import smith from '../../src/index.js'
import * as T from '../../src/tensor.js'
import { matmul2d } from '../../src/ops/matmul.js'
import { transpose } from '../../src/ops/transpose.js'
import { rmsnormForward } from '../../src/ops/rmsnorm.js'
import { swigluForward } from '../../src/ops/swiglu.js'
import { ropeForward, precomputeRoPE } from '../../src/ops/rope.js'

// Create KV cache for the Talker
function createTalkerCache(config, maxSeqLen = 4096) {
  const caches = []
  for (let i = 0; i < config.numLayers; i++) {
    caches.push({
      k: smith.zeros([config.numKVHeads, maxSeqLen, config.headDim]),
      v: smith.zeros([config.numKVHeads, maxSeqLen, config.headDim]),
      len: 0,
    })
  }
  return caches
}

// Write K/V for a single position into cache
function cacheAppend(cache, kNew, vNew) {
  const pos = cache.len
  const kvHeads = kNew.shape[0]
  const headDim = kNew.shape[2]
  const maxSeqLen = cache.k.shape[1]

  for (let h = 0; h < kvHeads; h++) {
    const srcOff = h * headDim
    const dstOff = h * maxSeqLen * headDim + pos * headDim
    cache.k.data.set(kNew.data.subarray(srcOff, srcOff + headDim), dstOff)
    cache.v.data.set(vNew.data.subarray(srcOff, srcOff + headDim), dstOff)
  }
  cache.len = pos + 1
}

// Write multiple positions (prefill)
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

// Get cached K/V slice
function cacheSlice(cache) {
  const len = cache.len
  const kvHeads = cache.k.shape[0]
  const headDim = cache.k.shape[2]
  const maxSeqLen = cache.k.shape[1]

  const kSlice = smith.zeros([kvHeads, len, headDim])
  const vSlice = smith.zeros([kvHeads, len, headDim])

  for (let h = 0; h < kvHeads; h++) {
    const srcOff = h * maxSeqLen * headDim
    const dstOff = h * len * headDim
    kSlice.data.set(cache.k.data.subarray(srcOff, srcOff + len * headDim), dstOff)
    vSlice.data.set(cache.v.data.subarray(srcOff, srcOff + len * headDim), dstOff)
  }
  return { k: kSlice, v: vSlice }
}

// Repeat KV heads for GQA
function repeatKV(kvTensor, numHeads, kvHeads) {
  if (kvHeads === numHeads) return kvTensor
  const repeats = numHeads / kvHeads
  const seqLen = kvTensor.shape[1]
  const headDim = kvTensor.shape[2]
  const out = smith.zeros([numHeads, seqLen, headDim])

  for (let h = 0; h < kvHeads; h++) {
    const srcOff = h * seqLen * headDim
    for (let r = 0; r < repeats; r++) {
      const dstOff = (h * repeats + r) * seqLen * headDim
      out.data.set(kvTensor.data.subarray(srcOff, srcOff + seqLen * headDim), dstOff)
    }
  }
  return out
}

// Lazy-cache the contiguous transposed weight [outDim, inDim] → [inDim, outDim].
// Created once on first use, then reused for all subsequent calls.
function getWeightT(weight) {
  if (weight._t) return weight._t
  const view = transpose(weight) // virtual: [inDim, outDim], strided
  weight._t = T.contiguous(view) // materialize once into a new Metal buffer
  return weight._t
}

// Linear projection (no bias): output = input × weight^T
// input [seqLen, inDim], weight [outDim, inDim] → output [seqLen, outDim]
// GPU matmul: [seqLen, inDim] @ [inDim, outDim] = [seqLen, outDim]
function linearNoBias(input, weight) {
  return matmul2d(input, getWeightT(weight))
}

// Linear with bias: output = input × weight^T + bias
function linearBias(input, weight, bias) {
  const out = matmul2d(input, getWeightT(weight))
  const seqLen = out.shape[0]
  const outDim = out.shape[1]
  for (let s = 0; s < seqLen; s++) {
    const off = s * outDim
    for (let o = 0; o < outDim; o++) {
      out.data[off + o] += bias.data[o]
    }
  }
  return out
}

// Text projection: SiLU-gated MLP
// down: [hidden, text_hidden] → [hidden], up: [hidden, hidden] → [hidden]
function textProjection(input, proj) {
  const h = linearBias(input, proj.fc1.weight, proj.fc1.bias)
  // SiLU activation
  for (let i = 0; i < h.data.length; i++) {
    const x = h.data[i]
    h.data[i] = x / (1 + Math.exp(-x))
  }
  return linearBias(h, proj.fc2.weight, proj.fc2.bias)
}

// QK normalization: per-head RMSNorm on Q/K vectors
// input: [seqLen, numHeads * headDim], norm: [headDim]
function qkNorm(input, normWeight, numHeads, headDim, eps) {
  const seqLen = input.shape[0]
  const out = smith.zeros(input.shape)
  for (let s = 0; s < seqLen; s++) {
    for (let h = 0; h < numHeads; h++) {
      const base = s * numHeads * headDim + h * headDim
      // RMSNorm over headDim
      let sumSq = 0
      for (let d = 0; d < headDim; d++) {
        sumSq += input.data[base + d] * input.data[base + d]
      }
      const rms = 1 / Math.sqrt(sumSq / headDim + eps)
      for (let d = 0; d < headDim; d++) {
        out.data[base + d] = input.data[base + d] * rms * normWeight.data[d]
      }
    }
  }
  return out
}

// Embedding lookup: ids → [seqLen, dim]
// Supports lazy BF16 embeddings — converts only the rows actually needed.
function embed(ids, embeddingWeight) {
  const vocabSize = embeddingWeight.shape[0]
  const dim = embeddingWeight.shape[1]
  const seqLen = ids.length
  const out = smith.zeros([seqLen, dim])

  if (embeddingWeight.lazy) {
    // Lazy BF16: convert requested rows on the fly (dim elements each, ~8KB per row)
    const u16 = embeddingWeight.bf16
    for (let i = 0; i < seqLen; i++) {
      const id = ids[i]
      if (id >= 0 && id < vocabSize) {
        const rowStart = id * dim
        const outStart = i * dim
        const u32 = new Uint32Array(out.data.buffer, out.data.byteOffset + outStart * 4, dim)
        for (let d = 0; d < dim; d++) u32[d] = u16[rowStart + d] << 16
      }
    }
  } else {
    for (let i = 0; i < seqLen; i++) {
      const id = ids[i]
      if (id >= 0 && id < vocabSize) {
        out.data.set(embeddingWeight.data.subarray(id * dim, (id + 1) * dim), i * dim)
      }
    }
  }
  return out
}

// Single Llama block forward — decode (single token, KV cache)
function talkerBlockDecode(x, block, cache, position, model) {
  const { dim, numHeads, numKVHeads, headDim } = model.config
  const kvHeads = numKVHeads

  // Pre-norm
  const norm1 = rmsnormForward(x, block.inputLayernorm, model.config.normEps)

  // Q, K, V
  let Q = linearNoBias(norm1, block.qProj)      // [1, dim]
  let K = linearNoBias(norm1, block.kProj)      // [1, kvDim]
  const V = linearNoBias(norm1, block.vProj)      // [1, kvDim]

  // QK norm (per-head RMSNorm before RoPE)
  Q = qkNorm(Q, block.qNorm, numHeads, headDim, model.config.normEps)
  K = qkNorm(K, block.kNorm, kvHeads, headDim, model.config.normEps)

  // Reshape: [1, dim] → [numHeads, 1, headDim], then RoPE per-head
  const Qh = reshapeToHeads(Q, 1, numHeads, headDim)
  const Kh = reshapeToHeads(K, 1, kvHeads, headDim)
  const Vh = reshapeToHeads(V, 1, kvHeads, headDim)

  ropePerHead(Qh, numHeads, 1, headDim, model.rope, position)
  ropePerHead(Kh, kvHeads, 1, headDim, model.rope, position)

  // Append to cache
  cacheAppend(cache, Kh, Vh)

  // Get full cache
  const { k: cachedK, v: cachedV } = cacheSlice(cache)

  // GQA: repeat KV
  const fullK = repeatKV(cachedK, numHeads, kvHeads)
  const fullV = repeatKV(cachedV, numHeads, kvHeads)

  // Attention: Q [H, 1, D] @ K^T [H, D, seqLen]
  const scaleFactor = 1 / Math.sqrt(headDim)
  const seqLen = cache.len
  const attnOut = computeAttention(Qh, fullK, fullV, numHeads, headDim, seqLen, scaleFactor)

  // Output projection
  const projected = linearNoBias(attnOut, block.oProj)

  // Residual
  const x2 = addTensors(x, projected)

  // FFN: SwiGLU
  const norm2 = rmsnormForward(x2, block.postAttnLayernorm, model.config.normEps)
  const gate = linearNoBias(norm2, block.gateProj)
  const up = linearNoBias(norm2, block.upProj)
  const fused = swigluForward(gate, up)
  const ffnOut = linearNoBias(fused, block.downProj)

  return addTensors(x2, ffnOut)
}

// Prefill: process full sequence at once
function talkerBlockPrefill(x, block, cache, seqLen, model) {
  const { dim, numHeads, numKVHeads, headDim } = model.config
  const kvHeads = numKVHeads

  const norm1 = rmsnormForward(x, block.inputLayernorm, model.config.normEps)

  let Q = linearNoBias(norm1, block.qProj)
  let K = linearNoBias(norm1, block.kProj)
  const V = linearNoBias(norm1, block.vProj)

  // QK norm (per-head RMSNorm before RoPE)
  Q = qkNorm(Q, block.qNorm, numHeads, headDim, model.config.normEps)
  K = qkNorm(K, block.kNorm, kvHeads, headDim, model.config.normEps)

  // Reshape: [seqLen, dim] → [numHeads, seqLen, headDim], then RoPE per-head
  const Qh = reshapeToHeads(Q, seqLen, numHeads, headDim)
  const Kh = reshapeToHeads(K, seqLen, kvHeads, headDim)
  const Vh = reshapeToHeads(V, seqLen, kvHeads, headDim)

  ropePerHead(Qh, numHeads, seqLen, headDim, model.rope, 0)
  ropePerHead(Kh, kvHeads, seqLen, headDim, model.rope, 0)

  // Store in cache
  cachePrefill(cache, Kh, Vh, seqLen)

  // GQA
  const fullK = repeatKV(Kh, numHeads, kvHeads)
  const fullV = repeatKV(Vh, numHeads, kvHeads)

  // Flash attention (causal)
  const qV = smith.variable(Qh, { requiresGrad: false })
  const kV = smith.variable(fullK, { requiresGrad: false })
  const vV = smith.variable(fullV, { requiresGrad: false })
  const attnResult = smith.flashAttention(qV, kV, vV, true)
  const attnData = attnResult.data

  // Reshape back: [numHeads, seqLen, headDim] → [seqLen, dim]
  const attnOut = reshapeFromHeads(attnData, seqLen, numHeads, headDim, dim)
  const projected = linearNoBias(attnOut, block.oProj)
  const x2 = addTensors(x, projected)

  const norm2 = rmsnormForward(x2, block.postAttnLayernorm, model.config.normEps)
  const gate = linearNoBias(norm2, block.gateProj)
  const up = linearNoBias(norm2, block.upProj)
  const fused = swigluForward(gate, up)
  const ffnOut = linearNoBias(fused, block.downProj)

  return addTensors(x2, ffnOut)
}

// Apply RoPE per-head on a [numHeads, seqLen, headDim] tensor.
// Each head is a contiguous [seqLen, headDim] block — extract, rotate, write back.
function ropePerHead(heads, numHeads, seqLen, headDim, ropeTable, startPos) {
  const headSize = seqLen * headDim
  for (let h = 0; h < numHeads; h++) {
    const off = h * headSize
    const slice = smith.zeros([seqLen, headDim])
    slice.data.set(heads.data.subarray(off, off + headSize))
    const rotated = ropeForward(slice, ropeTable, startPos)
    heads.data.set(rotated.data, off)
  }
}

// Reshape [seqLen, dim] → [numHeads, seqLen, headDim]
function reshapeToHeads(tensor, seqLen, numHeads, headDim) {
  const out = smith.zeros([numHeads, seqLen, headDim])
  for (let s = 0; s < seqLen; s++) {
    for (let h = 0; h < numHeads; h++) {
      const srcOff = s * numHeads * headDim + h * headDim
      const dstOff = h * seqLen * headDim + s * headDim
      out.data.set(tensor.data.subarray(srcOff, srcOff + headDim), dstOff)
    }
  }
  return out
}

// Reshape [numHeads, seqLen, headDim] → [seqLen, dim]
function reshapeFromHeads(tensor, seqLen, numHeads, headDim, dim) {
  const out = smith.zeros([seqLen, dim])
  for (let s = 0; s < seqLen; s++) {
    for (let h = 0; h < numHeads; h++) {
      const srcOff = h * seqLen * headDim + s * headDim
      const dstOff = s * dim + h * headDim
      out.data.set(tensor.data.subarray(srcOff, srcOff + headDim), dstOff)
    }
  }
  return out
}

// Compute attention for single-token decode (no causal mask needed)
function computeAttention(Q, K, V, numHeads, headDim, seqLen, scale) {
  // Q: [H, 1, D], K: [H, seqLen, D], V: [H, seqLen, D]
  const scores = smith.zeros([numHeads, 1, seqLen])
  // Q @ K^T
  for (let h = 0; h < numHeads; h++) {
    for (let s = 0; s < seqLen; s++) {
      let dot = 0
      for (let d = 0; d < headDim; d++) {
        dot += Q.data[h * headDim + d] * K.data[h * seqLen * headDim + s * headDim + d]
      }
      scores.data[h * seqLen + s] = dot * scale
    }
  }

  // Softmax per head
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

  // Weighted sum: scores @ V → [H, 1, D]
  const out = smith.zeros([1, numHeads * headDim])
  for (let h = 0; h < numHeads; h++) {
    for (let d = 0; d < headDim; d++) {
      let acc = 0
      for (let s = 0; s < seqLen; s++) {
        acc += scores.data[h * seqLen + s] * V.data[h * seqLen * headDim + s * headDim + d]
      }
      out.data[h * headDim + d] = acc
    }
  }
  return out
}

// Element-wise add
function addTensors(a, b) {
  const out = smith.zeros(a.shape)
  for (let i = 0; i < a.data.length; i++) {
    out.data[i] = a.data[i] + b.data[i]
  }
  return out
}

// Prepare the Talker model for inference
function prepareTalker(talker) {
  // Precompute RoPE tables
  talker.rope = precomputeRoPE(talker.config.headDim, 8192, talker.config.ropeTheta)
  return talker
}

// Build input embeddings for the simplest case: text → speech, no voice clone
// Format: [role: <|im_start|>assistant\n] [codec_tags + codec_bos] [text_stream + codec_pad_stream]
function buildInputEmbeds(textTokenIds, talker) {
  const c = talker.config

  // Role prefix: first 3 tokens of the chat format (<|im_start|>, "assistant", \n)
  const roleIds = textTokenIds.slice(0, 3)
  const roleEmbed = textProjection(embed(roleIds, talker.textEmbedding), talker.textProjection)

  // Text tokens (after role, before <|im_end|>)
  // Format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
  // roleIds = [0:3], textIds = [3:end-5], suffix = [-5:]
  const textIds = textTokenIds.slice(3, -5)
  const suffixIds = textTokenIds.slice(-5) // <|im_end|>\n<|im_start|>assistant\n

  // Codec prefix tokens: [nothink, think_bos, think_eos, pad, bos]
  const codecPrefixIds = [c.codecNothinkId, c.codecThinkBosId, c.codecThinkEosId, c.codecPadId, c.codecBosId]
  const codecPrefixEmbed = embed(codecPrefixIds, talker.codecEmbedding)

  // TTS special embeddings
  const ttsSpecialIds = [151672, 151673, 151671] // bos, eos, pad
  const ttsSpecialEmbed = textProjection(
    embed(ttsSpecialIds, talker.textEmbedding),
    talker.textProjection,
  )
  const ttsBosEmbed = smith.zeros([1, c.dim])
  const ttsEosEmbed = smith.zeros([1, c.dim])
  const ttsPadEmbed = smith.zeros([1, c.dim])
  ttsBosEmbed.data.set(ttsSpecialEmbed.data.subarray(0, c.dim))
  ttsEosEmbed.data.set(ttsSpecialEmbed.data.subarray(c.dim, 2 * c.dim))
  ttsPadEmbed.data.set(ttsSpecialEmbed.data.subarray(2 * c.dim, 3 * c.dim))

  // Build combined embeddings:
  // Position 0-2: role text embed (no codec)
  // Position 3-6: tts_pad × (nTags-1) + tts_bos summed with codec prefix tokens[0:nTags-1] + codec_prefix[-1]
  // Position 7+: text embed + codec_pad, streaming

  // Codec prefix construction:
  // [nothink, think_bos, think_eos] tags (3 tokens) + [pad, bos] (2 tokens) = 5 positions
  // Text stream side: tts_pad × 3 + tts_bos summed with codec tags + bos
  const nTags = 3 // nothink, think_bos, think_eos
  const embedDim = c.dim

  // Build the codec prefix section: tags + pad + bos
  const codecSection = smith.zeros([nTags + 2, embedDim])
  // First nTags: tts_pad + codec_tag
  for (let i = 0; i < nTags; i++) {
    for (let d = 0; d < embedDim; d++) {
      codecSection.data[i * embedDim + d] = ttsPadEmbed.data[d] + codecPrefixEmbed.data[i * embedDim + d]
    }
  }
  // pad position: tts_pad + codec_pad
  for (let d = 0; d < embedDim; d++) {
    codecSection.data[nTags * embedDim + d] = ttsPadEmbed.data[d] + codecPrefixEmbed.data[nTags * embedDim + d]
  }
  // bos position: tts_bos + codec_bos
  for (let d = 0; d < embedDim; d++) {
    codecSection.data[(nTags + 1) * embedDim + d] = ttsBosEmbed.data[d] + codecPrefixEmbed.data[(nTags + 1) * embedDim + d]
  }

  // Non-streaming mode: text + eos + codec_pad overlay, then tts_pad + codec_bos at end
  const textEmbed = textProjection(embed(textIds, talker.textEmbedding), talker.textProjection)
  const textLen = textIds.length

  // Text stream: text[0] is first text token summed with first codec position after bos
  // For non-streaming: all text + eos embedded, overlaid with codec_pad
  const codecPadSingle = embed([c.codecPadId], talker.codecEmbedding)
  const codecBosSingle = embed([c.codecBosId], talker.codecEmbedding)

  // Build text+codec overlay (text_embed + codec_pad for each text position, then tts_eos + codec_pad, then tts_pad + codec_bos)
  const overlayLen = textLen + 1 + 1 // text + eos + bos
  const textOverlay = smith.zeros([overlayLen, embedDim])

  // Text positions: text_projection(text_embed) + codec_pad
  for (let i = 0; i < textLen; i++) {
    for (let d = 0; d < embedDim; d++) {
      textOverlay.data[i * embedDim + d] = textEmbed.data[i * embedDim + d] + codecPadSingle.data[d]
    }
  }
  // EOS position: tts_eos + codec_pad
  for (let d = 0; d < embedDim; d++) {
    textOverlay.data[textLen * embedDim + d] = ttsEosEmbed.data[d] + codecPadSingle.data[d]
  }
  // Final bos: tts_pad + codec_bos
  for (let d = 0; d < embedDim; d++) {
    textOverlay.data[(textLen + 1) * embedDim + d] = ttsPadEmbed.data[d] + codecBosSingle.data[d]
  }

  // Concatenate all: role (3) + codec section (5) + text overlay
  const totalLen = 3 + (nTags + 2) + overlayLen
  const inputEmbeds = smith.zeros([totalLen, embedDim])

  // Copy role
  inputEmbeds.data.set(roleEmbed.data.subarray(0, 3 * embedDim), 0)
  // Copy codec section
  inputEmbeds.data.set(codecSection.data, 3 * embedDim)
  // Copy text overlay
  inputEmbeds.data.set(textOverlay.data, (3 + nTags + 2) * embedDim)

  return { inputEmbeds, ttsPadEmbed, ttsEosEmbed, totalLen }
}

// Prefill the Talker on the input embeddings
function talkerPrefill(model, inputEmbeds, seqLen, caches) {
  let x = inputEmbeds

  for (let i = 0; i < model.blocks.length; i++) {
    x = talkerBlockPrefill(x, model.blocks[i], caches[i], seqLen, model)
  }

  // Final norm
  x = rmsnormForward(x, model.norm, model.config.normEps)

  // Logits for last position only
  const lastHidden = smith.zeros([1, model.config.dim])
  lastHidden.data.set(x.data.subarray((seqLen - 1) * model.config.dim, seqLen * model.config.dim))

  const logits = linearNoBias(lastHidden, model.codecHead)
  const hidden = smith.zeros([1, model.config.dim])
  hidden.data.set(lastHidden.data)

  return { logits, hidden }
}

// Decode a single token
function talkerDecode(model, inputEmbed, position, caches) {
  let x = inputEmbed // [1, dim]

  for (let i = 0; i < model.blocks.length; i++) {
    x = talkerBlockDecode(x, model.blocks[i], caches[i], position, model)
  }

  x = rmsnormForward(x, model.norm, model.config.normEps)
  const logits = linearNoBias(x, model.codecHead)

  return { logits, hidden: x }
}

// Sample a token from logits
function sampleToken(logits, temperature = 0.9, topK = 50, repetitionPenalty = 1.05, generated = []) {
  const vocabSize = logits.shape[1] || logits.shape[0]
  const data = new Float32Array(vocabSize)
  data.set(logits.data.subarray(0, vocabSize))

  // Repetition penalty
  if (repetitionPenalty !== 1.0) {
    for (const id of generated) {
      if (id < vocabSize) {
        data[id] = data[id] > 0 ? data[id] / repetitionPenalty : data[id] * repetitionPenalty
      }
    }
  }

  // Temperature
  if (temperature !== 1.0 && temperature > 0) {
    for (let i = 0; i < vocabSize; i++) data[i] /= temperature
  }

  // Top-K
  if (topK > 0 && topK < vocabSize) {
    const indices = Array.from({ length: vocabSize }, (_, i) => i)
    indices.sort((a, b) => data[b] - data[a])
    const threshold = data[indices[topK - 1]]
    for (let i = 0; i < vocabSize; i++) {
      if (data[i] < threshold) data[i] = -Infinity
    }
  }

  // Softmax
  let max = -Infinity
  for (let i = 0; i < vocabSize; i++) max = Math.max(max, data[i])
  let sum = 0
  for (let i = 0; i < vocabSize; i++) {
    data[i] = Math.exp(data[i] - max)
    sum += data[i]
  }
  for (let i = 0; i < vocabSize; i++) data[i] /= sum

  // Sample
  const r = Math.random()
  let cumsum = 0
  for (let i = 0; i < vocabSize; i++) {
    cumsum += data[i]
    if (cumsum >= r) return i
  }
  return vocabSize - 1
}

export {
  prepareTalker,
  createTalkerCache,
  buildInputEmbeds,
  talkerPrefill,
  talkerDecode,
  sampleToken,
  embed,
  textProjection,
  linearNoBias,
  linearBias,
  qkNorm,
  addTensors,
  reshapeToHeads,
  reshapeFromHeads,
  repeatKV,
  ropePerHead,
}
