// smith/src/model.js
// GPT-style decoder-only transformer language model.
// Ported from TinyFormer's model.js.

import * as T from './tensor.js'
import * as A from './autograd.js'
import {
  createTransformerBlock, transformerBlock, transformerBlockFlash, transformerBlockCached, blockParams, countParams,
  createCausalMask,
} from './nn.js'

// --- Model configs ---

const CONFIGS = {
  tiny:   { numLayers: 2, numHeads: 2, dim: 64,  maxSeqLen: 128 },
  small:  { numLayers: 4, numHeads: 4, dim: 128, maxSeqLen: 256 },
  medium: { numLayers: 6, numHeads: 6, dim: 192, maxSeqLen: 256 },
}

// --- Embedding layer ---

function createEmbedding(config) {
  const { vocabSize, dim, maxSeqLen } = config
  // N(0, 0.02) init like GPT-2
  const s = 0.02
  const tokenWeight = A.variable(T.tensor(
    Array.from({ length: vocabSize * dim }, () => (Math.random() * 2 - 1) * s),
    [vocabSize, dim]
  ), { requiresGrad: true })

  const posWeight = A.variable(T.tensor(
    Array.from({ length: maxSeqLen * dim }, () => (Math.random() * 2 - 1) * s),
    [maxSeqLen, dim]
  ), { requiresGrad: true })

  return { tokenWeight, posWeight, dim }
}

function embed(indices, layer) {
  const seqLen = indices.length
  const tok = A.embedding(indices, layer.tokenWeight)
  const posIndices = Array.from({ length: seqLen }, (_, i) => i)
  const pos = A.embedding(posIndices, layer.posWeight)
  return A.add(tok, pos)
}

function embeddingParams(layer) {
  return [layer.tokenWeight, layer.posWeight]
}

// --- Model creation ---

function createModel(config) {
  const { vocabSize, numLayers, numHeads, dim, maxSeqLen } = config
  const embeddingLayer = createEmbedding({ vocabSize, dim, maxSeqLen })

  const blocks = []
  const residualScale = 1 / Math.sqrt(numLayers)
  for (let i = 0; i < numLayers; i++) {
    const block = createTransformerBlock(dim, numHeads)
    // GPT-2 residual scaling on output projections
    const outW = block.mha.outProj.weight.data
    for (let j = 0; j < outW.data.length; j++) outW.data[j] *= residualScale
    const ffn2W = block.ffn2.weight.data
    for (let j = 0; j < ffn2W.data.length; j++) ffn2W.data[j] *= residualScale
    blocks.push(block)
  }

  const lnFGamma = A.variable(T.ones([dim]), { requiresGrad: true })
  const lnFBeta = A.variable(T.zeros([dim]), { requiresGrad: true })

  return {
    embedding: embeddingLayer,
    blocks,
    lnFGamma, lnFBeta,
    config: { vocabSize, numLayers, numHeads, dim, maxSeqLen },
  }
}

function forward(model, tokenIds) {
  const seqLen = tokenIds.length
  const mask = createCausalMask(seqLen)

  let x = embed(tokenIds, model.embedding)

  for (const block of model.blocks) {
    const result = transformerBlock(x, block, mask)
    x = result.output
  }

  x = A.layernorm(x, model.lnFGamma, model.lnFBeta)

  // Weight-tied output head: logits = x @ tokenEmbed^T
  const logits = A.matmul(x, A.transpose(model.embedding.tokenWeight))

  return { logits }
}

function modelParams(model) {
  const params = []
  params.push(...embeddingParams(model.embedding))
  for (const block of model.blocks) params.push(...blockParams(block))
  params.push(model.lnFGamma, model.lnFBeta)
  return params
}

function modelInfo(model) {
  const params = modelParams(model)
  const total = countParams(params)
  const { vocabSize, numLayers, numHeads, dim, maxSeqLen } = model.config
  return { vocabSize, numLayers, numHeads, dim, maxSeqLen, totalParams: total, paramCount: params.length }
}

// --- Flash forward pass (O(n) memory attention) ---

function forwardFlash(model, tokenIds) {
  let x = embed(tokenIds, model.embedding)

  for (const block of model.blocks) {
    const result = transformerBlockFlash(x, block)
    x = result.output
  }

  x = A.layernorm(x, model.lnFGamma, model.lnFBeta)
  const logits = A.matmul(x, A.transpose(model.embedding.tokenWeight))

  return { logits }
}

// --- Cached forward pass (single token, for generation with KV cache) ---

function forwardCached(model, tokenId, position, kvCaches) {
  // tokenId: single int, position: int (absolute position in sequence)
  // kvCaches: array of { k, v } per block, or null
  // Returns { logits: Variable [1, vocabSize], newCaches }

  // Embed single token at the given position
  const tokEmb = A.embedding([tokenId], model.embedding.tokenWeight) // [1, dim]
  const posEmb = A.embedding([position], model.embedding.posWeight)  // [1, dim]
  let x = A.add(tokEmb, posEmb) // [1, dim]

  // Run through blocks with cache
  const newCaches = []
  for (let i = 0; i < model.blocks.length; i++) {
    const cache = kvCaches ? kvCaches[i] : null
    const result = transformerBlockCached(x, model.blocks[i], cache)
    x = result.output
    newCaches.push(result.newCache)
  }

  // Final layer norm + weight-tied logits
  x = A.layernorm(x, model.lnFGamma, model.lnFBeta)
  const logits = A.matmul(x, A.transpose(model.embedding.tokenWeight))

  return { logits, newCaches }
}

export {
  CONFIGS, createModel, forward, forwardFlash, forwardCached, modelParams, modelInfo,
  createEmbedding, embed, embeddingParams,
}
