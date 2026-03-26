// smith/src/nn.js
// Neural network building blocks: linear, multi-head attention, transformer block.
// Ported from TinyFormer's attention.js + transformer.js.

import * as T from './tensor.js'
import * as A from './autograd.js'

// --- Linear projection ---

function createLinear(inDim, outDim, useBias = true) {
  const s = Math.sqrt(2 / (inDim + outDim))
  const weight = A.variable(T.tensor(
    Array.from({ length: inDim * outDim }, () => (Math.random() * 2 - 1) * s),
    [inDim, outDim]
  ), { requiresGrad: true })
  const bias = useBias
    ? A.variable(T.zeros([outDim]), { requiresGrad: true })
    : null
  return { weight, bias, inDim, outDim }
}

function linear(x, layer) {
  const out = A.matmul(x, layer.weight)
  return layer.bias ? A.add(out, layer.bias) : out
}

function linearParams(layer) {
  return layer.bias ? [layer.weight, layer.bias] : [layer.weight]
}

// --- Causal mask ---

function createCausalMask(seqLen) {
  const data = new Float32Array(seqLen * seqLen)
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      data[i * seqLen + j] = j > i ? -Infinity : 0
    }
  }
  return T.tensor(Array.from(data), [seqLen, seqLen])
}

// --- Scaled dot-product attention ---

function scaledDotProductAttention(Q, K, V, mask) {
  const dk = Q.data.shape[Q.data.shape.length - 1]
  const scaleFactor = 1 / Math.sqrt(dk)

  const ndim = K.data.shape.length
  const kAxes = []
  for (let i = 0; i < ndim - 2; i++) kAxes.push(i)
  kAxes.push(ndim - 1, ndim - 2)

  const scores = A.scale(A.matmul(Q, A.transpose(K, kAxes)), scaleFactor)

  let masked = scores
  if (mask) {
    const maskVar = A.variable(mask, { requiresGrad: false })
    masked = A.add(scores, maskVar)
  }

  const weights = A.softmax(masked, -1)
  const out = A.matmul(weights, V)
  return { output: out, weights }
}

// --- Multi-head attention ---

function createMultiHeadAttention(dim, numHeads) {
  if (dim % numHeads !== 0) throw new Error(`dim (${dim}) must be divisible by numHeads (${numHeads})`)
  const headDim = dim / numHeads
  return {
    qProj: createLinear(dim, dim),
    kProj: createLinear(dim, dim),
    vProj: createLinear(dim, dim),
    outProj: createLinear(dim, dim),
    numHeads,
    headDim,
    dim,
  }
}

function multiHeadAttention(x, layer, mask) {
  const seqLen = x.data.shape[0]
  const { numHeads, headDim, dim } = layer

  const Q = linear(x, layer.qProj)
  const K = linear(x, layer.kProj)
  const V = linear(x, layer.vProj)

  // [seqLen, dim] → [numHeads, seqLen, headDim]
  const Qh = A.transpose(A.reshape(Q, [seqLen, numHeads, headDim]), [1, 0, 2])
  const Kh = A.transpose(A.reshape(K, [seqLen, numHeads, headDim]), [1, 0, 2])
  const Vh = A.transpose(A.reshape(V, [seqLen, numHeads, headDim]), [1, 0, 2])

  const { output: attnOut } = scaledDotProductAttention(Qh, Kh, Vh, mask)

  // [numHeads, seqLen, headDim] → [seqLen, dim]
  const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [seqLen, dim])
  return { output: linear(concatenated, layer.outProj) }
}

// --- Cross-attention ---
// Q comes from x, K and V come from a separate source (e.g. encoder output).
// Uses the same layer structure as self-attention.

function multiHeadCrossAttention(x, kv, layer, mask) {
  // x:  [seqLen, dim] — query source (decoder)
  // kv: [kvLen, dim]  — key/value source (encoder output)
  const seqLen = x.data.shape[0]
  const kvLen = kv.data.shape[0]
  const { numHeads, headDim, dim } = layer

  const Q = linear(x, layer.qProj)
  const K = linear(kv, layer.kProj)
  const V = linear(kv, layer.vProj)

  const Qh = A.transpose(A.reshape(Q, [seqLen, numHeads, headDim]), [1, 0, 2])
  const Kh = A.transpose(A.reshape(K, [kvLen, numHeads, headDim]), [1, 0, 2])
  const Vh = A.transpose(A.reshape(V, [kvLen, numHeads, headDim]), [1, 0, 2])

  const { output: attnOut } = scaledDotProductAttention(Qh, Kh, Vh, mask)

  const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [seqLen, dim])
  return { output: linear(concatenated, layer.outProj) }
}

// --- Cached cross-attention (for generation with pre-computed encoder output) ---

function multiHeadCrossAttentionCached(x, encoderKV, layer) {
  // x:  [1, dim] — single decoder token
  // encoderKV: { k: [numHeads, kvLen, headDim], v: same } — pre-computed from encoder
  const { numHeads, headDim, dim } = layer

  const Q = linear(x, layer.qProj)
  const Qh = A.transpose(A.reshape(Q, [1, numHeads, headDim]), [1, 0, 2])

  // Attention: Q [H, 1, D] @ K^T [H, D, kvLen] → [H, 1, kvLen]
  const scaleFactor = 1 / Math.sqrt(headDim)
  const ndim = encoderKV.k.data.shape.length
  const kAxes = []
  for (let i = 0; i < ndim - 2; i++) kAxes.push(i)
  kAxes.push(ndim - 1, ndim - 2)

  const scores = A.scale(A.matmul(Qh, A.transpose(encoderKV.k, kAxes)), scaleFactor)
  const weights = A.softmax(scores, -1)
  const attnOut = A.matmul(weights, encoderKV.v)

  const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [1, dim])
  return { output: linear(concatenated, layer.outProj) }
}

function mhaParams(layer) {
  return [
    ...linearParams(layer.qProj),
    ...linearParams(layer.kProj),
    ...linearParams(layer.vProj),
    ...linearParams(layer.outProj),
  ]
}

// --- Transformer block (pre-norm, like GPT-2) ---

function createTransformerBlock(dim, numHeads, ffnDim) {
  if (!ffnDim) ffnDim = dim * 4
  return {
    ln1Gamma: A.variable(T.ones([dim]), { requiresGrad: true }),
    ln1Beta: A.variable(T.zeros([dim]), { requiresGrad: true }),
    ln2Gamma: A.variable(T.ones([dim]), { requiresGrad: true }),
    ln2Beta: A.variable(T.zeros([dim]), { requiresGrad: true }),
    mha: createMultiHeadAttention(dim, numHeads),
    ffn1: createLinear(dim, ffnDim),
    ffn2: createLinear(ffnDim, dim),
    dim, numHeads, ffnDim,
  }
}

function transformerBlock(x, block, mask) {
  // Attention sublayer with pre-norm + residual
  const norm1 = A.layernorm(x, block.ln1Gamma, block.ln1Beta)
  const { output: attnOut } = multiHeadAttention(norm1, block.mha, mask)
  const x2 = A.add(x, attnOut)

  // FFN sublayer with pre-norm + residual
  const norm2 = A.layernorm(x2, block.ln2Gamma, block.ln2Beta)
  const ffnHidden = A.gelu(linear(norm2, block.ffn1))
  const ffnOut = linear(ffnHidden, block.ffn2)
  return { output: A.add(x2, ffnOut) }
}

function blockParams(block) {
  return [
    block.ln1Gamma, block.ln1Beta,
    block.ln2Gamma, block.ln2Beta,
    ...mhaParams(block.mha),
    ...linearParams(block.ffn1),
    ...linearParams(block.ffn2),
  ]
}

function countParams(params) {
  let total = 0
  for (const p of params) total += T.shapeSize(p.data.shape)
  return total
}

// --- Sinusoidal positional embedding ---
// Used in the original Transformer, BERT, Whisper encoder.
// Returns a tensor (not variable) of shape [maxLen, dim].

function sinusoidalPE(maxLen, dim) {
  const pe = new Float32Array(maxLen * dim)
  for (let pos = 0; pos < maxLen; pos++) {
    for (let i = 0; i < dim; i += 2) {
      const angle = pos / Math.pow(10000, i / dim)
      pe[pos * dim + i] = Math.sin(angle)
      if (i + 1 < dim) pe[pos * dim + i + 1] = Math.cos(angle)
    }
  }
  return T.tensor(Array.from(pe), [maxLen, dim])
}

// --- Flash multi-head attention ---
// Uses fused flash attention kernel: O(n) memory, no materialized score matrix.
// Replaces the decomposed Q@K^T → softmax → @V pipeline with a single GPU dispatch.

function multiHeadAttentionFlash(x, layer, causal = true) {
  const seqLen = x.data.shape[0]
  const { numHeads, headDim, dim } = layer

  const Q = linear(x, layer.qProj)
  const K = linear(x, layer.kProj)
  const V = linear(x, layer.vProj)

  // [seqLen, dim] → [numHeads, seqLen, headDim]
  const Qh = A.transpose(A.reshape(Q, [seqLen, numHeads, headDim]), [1, 0, 2])
  const Kh = A.transpose(A.reshape(K, [seqLen, numHeads, headDim]), [1, 0, 2])
  const Vh = A.transpose(A.reshape(V, [seqLen, numHeads, headDim]), [1, 0, 2])

  // Single fused flash attention op (replaces matmul→scale→mask→softmax→matmul)
  const attnOut = A.flashAttention(Qh, Kh, Vh, causal)

  // [numHeads, seqLen, headDim] → [seqLen, dim]
  const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [seqLen, dim])
  return { output: linear(concatenated, layer.outProj) }
}

function transformerBlockFlash(x, block) {
  const norm1 = A.layernorm(x, block.ln1Gamma, block.ln1Beta)
  const { output: attnOut } = multiHeadAttentionFlash(norm1, block.mha, true)
  const x2 = A.add(x, attnOut)

  const norm2 = A.layernorm(x2, block.ln2Gamma, block.ln2Beta)
  const ffnHidden = A.gelu(linear(norm2, block.ffn1))
  const ffnOut = linear(ffnHidden, block.ffn2)
  return { output: A.add(x2, ffnOut) }
}

// --- Cached multi-head attention (for generation with KV cache) ---

// Concatenate along axis 1: [H, oldLen, D] + [H, 1, D] → [H, oldLen+1, D]
function catAlongAxis1(cached, newTensor, numHeads, headDim) {
  const oldLen = cached.data.shape[1]
  const newLen = oldLen + 1
  const result = T.create([numHeads, newLen, headDim], cached.data.dtype)
  const oldData = T.contiguous(cached.data)
  const newData = T.contiguous(newTensor.data)

  for (let h = 0; h < numHeads; h++) {
    const oldStart = h * oldLen * headDim
    const dstStart = h * newLen * headDim
    result.data.set(oldData.data.subarray(oldStart, oldStart + oldLen * headDim), dstStart)
    const srcStart = h * headDim
    result.data.set(newData.data.subarray(srcStart, srcStart + headDim), dstStart + oldLen * headDim)
  }

  return A.variable(result, { requiresGrad: false })
}

function multiHeadAttentionCached(x, layer, cache) {
  // x: Variable [1, dim] (single new token)
  // cache: { k: Variable [numHeads, cachedLen, headDim], v: same } or null
  const { numHeads, headDim, dim } = layer

  const Q = linear(x, layer.qProj)
  const K = linear(x, layer.kProj)
  const V = linear(x, layer.vProj)

  // Reshape to [numHeads, 1, headDim]
  const Qh = A.transpose(A.reshape(Q, [1, numHeads, headDim]), [1, 0, 2])
  const Kh = A.transpose(A.reshape(K, [1, numHeads, headDim]), [1, 0, 2])
  const Vh = A.transpose(A.reshape(V, [1, numHeads, headDim]), [1, 0, 2])

  // Append new K, V to cache
  let fullK, fullV
  if (cache && cache.k) {
    fullK = catAlongAxis1(cache.k, Kh, numHeads, headDim)
    fullV = catAlongAxis1(cache.v, Vh, numHeads, headDim)
  } else {
    fullK = Kh
    fullV = Vh
  }

  // Attention: Q [numHeads, 1, headDim] @ K^T [numHeads, headDim, seqLen]
  // No causal mask needed — Q has length 1, all cached positions are valid
  const scaleFactor = 1 / Math.sqrt(headDim)
  const ndim = fullK.data.shape.length
  const kAxes = []
  for (let i = 0; i < ndim - 2; i++) kAxes.push(i)
  kAxes.push(ndim - 1, ndim - 2)

  const scores = A.scale(A.matmul(Qh, A.transpose(fullK, kAxes)), scaleFactor)
  const weights = A.softmax(scores, -1)
  const attnOut = A.matmul(weights, fullV)

  // [numHeads, 1, headDim] → [1, dim]
  const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [1, dim])
  const out = linear(concatenated, layer.outProj)

  return { output: out, weights, newCache: { k: fullK, v: fullV } }
}

// --- Cached transformer block (for generation with KV cache) ---

function transformerBlockCached(x, block, cache) {
  // x: Variable [1, dim] (single token)
  // cache: { k, v } for this block's attention, or null
  const norm1 = A.layernorm(x, block.ln1Gamma, block.ln1Beta)
  const { output: attnOut, weights: attnWeights, newCache } = multiHeadAttentionCached(norm1, block.mha, cache)
  const x2 = A.add(x, attnOut)

  const norm2 = A.layernorm(x2, block.ln2Gamma, block.ln2Beta)
  const ffnHidden = A.gelu(linear(norm2, block.ffn1))
  const ffnOut = linear(ffnHidden, block.ffn2)
  const x3 = A.add(x2, ffnOut)

  return { output: x3, weights: attnWeights, newCache }
}

export {
  createLinear, linear, linearParams,
  createCausalMask,
  scaledDotProductAttention,
  createMultiHeadAttention, multiHeadAttention, multiHeadAttentionFlash, multiHeadAttentionCached, mhaParams,
  multiHeadCrossAttention, multiHeadCrossAttentionCached,
  createTransformerBlock, transformerBlock, transformerBlockFlash, transformerBlockCached, blockParams,
  countParams,
  sinusoidalPE,
}
