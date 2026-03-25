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

export {
  createLinear, linear, linearParams,
  createCausalMask,
  scaledDotProductAttention,
  createMultiHeadAttention, multiHeadAttention, mhaParams,
  createTransformerBlock, transformerBlock, blockParams,
  countParams,
}
