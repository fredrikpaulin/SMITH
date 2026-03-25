// smith/src/gguf_loader.js
// Load GGUF models into Smith. Maps weight names from various architectures
// (Llama, Phi, GPT-2) to Smith's model structure.
// Supports dequantized (f32) and quantized (Q4/Q8) weight loading.

import * as T from './tensor.js'
import * as A from './autograd.js'
import * as device from './device.js'
import {
  createLinear, linear, linearParams,
  createTransformerBlock, blockParams,
  createMultiHeadAttention, mhaParams,
  countParams,
} from './nn.js'
import {
  parseGGUF, extractConfig, dequantizeTensor, readTensorData,
  GGML_TYPE, GGML_TYPE_INFO,
} from './gguf.js'
import { matmulQ4, matmulQ8, Q8_BLOCK_SIZE, Q8_BLOCK_BYTES } from './ops/quantize.js'

// --- Weight name mappings ---
// Each architecture maps GGUF tensor names → Smith layer paths

const LLAMA_MAP = {
  'token_embd.weight':                     'embedding.tokenWeight',
  'output_norm.weight':                    'lnFGamma',
  'output.weight':                         'lmHead',
  // Per-layer patterns (use {L} placeholder)
  'blk.{L}.attn_norm.weight':             'blocks.{L}.ln1Gamma',
  'blk.{L}.ffn_norm.weight':              'blocks.{L}.ln2Gamma',
  'blk.{L}.attn_q.weight':               'blocks.{L}.mha.qProj.weight',
  'blk.{L}.attn_k.weight':               'blocks.{L}.mha.kProj.weight',
  'blk.{L}.attn_v.weight':               'blocks.{L}.mha.vProj.weight',
  'blk.{L}.attn_output.weight':          'blocks.{L}.mha.outProj.weight',
  'blk.{L}.ffn_gate.weight':             'blocks.{L}.ffnGate.weight',
  'blk.{L}.ffn_up.weight':               'blocks.{L}.ffnUp.weight',
  'blk.{L}.ffn_down.weight':             'blocks.{L}.ffnDown.weight',
}

const PHI_MAP = {
  'token_embd.weight':                     'embedding.tokenWeight',
  'output_norm.weight':                    'lnFGamma',
  'output_norm.bias':                      'lnFBeta',
  'output.weight':                         'lmHead',
  'blk.{L}.attn_norm.weight':             'blocks.{L}.ln1Gamma',
  'blk.{L}.attn_norm.bias':               'blocks.{L}.ln1Beta',
  'blk.{L}.ffn_norm.weight':              'blocks.{L}.ln2Gamma',
  'blk.{L}.ffn_norm.bias':                'blocks.{L}.ln2Beta',
  'blk.{L}.attn_qkv.weight':             'blocks.{L}.mha.qkvProj.weight',
  'blk.{L}.attn_qkv.bias':               'blocks.{L}.mha.qkvProj.bias',
  'blk.{L}.attn_output.weight':          'blocks.{L}.mha.outProj.weight',
  'blk.{L}.attn_output.bias':            'blocks.{L}.mha.outProj.bias',
  'blk.{L}.ffn_up.weight':               'blocks.{L}.ffn1.weight',
  'blk.{L}.ffn_up.bias':                  'blocks.{L}.ffn1.bias',
  'blk.{L}.ffn_down.weight':             'blocks.{L}.ffn2.weight',
  'blk.{L}.ffn_down.bias':                'blocks.{L}.ffn2.bias',
}

const GPT2_MAP = {
  'token_embd.weight':                     'embedding.tokenWeight',
  'position_embd.weight':                  'embedding.posWeight',
  'output_norm.weight':                    'lnFGamma',
  'output_norm.bias':                      'lnFBeta',
  'blk.{L}.attn_norm.weight':             'blocks.{L}.ln1Gamma',
  'blk.{L}.attn_norm.bias':               'blocks.{L}.ln1Beta',
  'blk.{L}.ffn_norm.weight':              'blocks.{L}.ln2Gamma',
  'blk.{L}.ffn_norm.bias':                'blocks.{L}.ln2Beta',
  'blk.{L}.attn_qkv.weight':             'blocks.{L}.mha.qkvFused.weight',
  'blk.{L}.attn_qkv.bias':               'blocks.{L}.mha.qkvFused.bias',
  'blk.{L}.attn_output.weight':          'blocks.{L}.mha.outProj.weight',
  'blk.{L}.attn_output.bias':            'blocks.{L}.mha.outProj.bias',
  'blk.{L}.ffn_up.weight':               'blocks.{L}.ffn1.weight',
  'blk.{L}.ffn_up.bias':                  'blocks.{L}.ffn1.bias',
  'blk.{L}.ffn_down.weight':             'blocks.{L}.ffn2.weight',
  'blk.{L}.ffn_down.bias':                'blocks.{L}.ffn2.bias',
}

function getWeightMap(arch) {
  if (arch === 'llama') return LLAMA_MAP
  if (arch === 'phi' || arch === 'phi2' || arch === 'phi3') return PHI_MAP
  if (arch === 'gpt2') return GPT2_MAP
  throw new Error(`Unsupported GGUF architecture: ${arch}`)
}

// Resolve a GGUF tensor name to a Smith path using the weight map
function resolveWeight(name, weightMap) {
  // Direct match first
  if (weightMap[name]) return weightMap[name]

  // Try pattern match with layer number
  const layerMatch = name.match(/^blk\.(\d+)\.(.+)$/)
  if (layerMatch) {
    const layerIdx = layerMatch[1]
    const suffix = layerMatch[2]
    const pattern = `blk.{L}.${suffix}`
    if (weightMap[pattern]) return weightMap[pattern].replace('{L}', layerIdx)
  }

  return null // unknown weight
}

// --- RoPE precomputation ---
// Llama-style rotary position embeddings: precompute cos/sin tables

function precomputeRoPE(dim, maxSeqLen, freqBase = 10000) {
  const halfDim = dim / 2
  const freqs = new Float32Array(halfDim)
  for (let i = 0; i < halfDim; i++) {
    freqs[i] = 1.0 / Math.pow(freqBase, (2 * i) / dim)
  }

  const cos = new Float32Array(maxSeqLen * halfDim)
  const sin = new Float32Array(maxSeqLen * halfDim)
  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const angle = pos * freqs[i]
      cos[pos * halfDim + i] = Math.cos(angle)
      sin[pos * halfDim + i] = Math.sin(angle)
    }
  }

  return {
    cos: T.tensor(Array.from(cos), [maxSeqLen, halfDim]),
    sin: T.tensor(Array.from(sin), [maxSeqLen, halfDim]),
  }
}

// Apply RoPE to Q or K tensor: [seqLen, dim] → [seqLen, dim]
// Rotates pairs: (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
function applyRoPE(x, ropeTable, startPos = 0) {
  const xData = T.contiguous(x.data)
  const seqLen = xData.shape[0]
  const dim = xData.shape[1]
  const halfDim = dim / 2

  const out = T.create(xData.shape, xData.dtype)
  for (let s = 0; s < seqLen; s++) {
    const pos = startPos + s
    for (let i = 0; i < halfDim; i++) {
      const x0 = xData.data[s * dim + i]
      const x1 = xData.data[s * dim + halfDim + i]
      const c = ropeTable.cos.data[pos * halfDim + i]
      const sn = ropeTable.sin.data[pos * halfDim + i]
      out.data[s * dim + i] = x0 * c - x1 * sn
      out.data[s * dim + halfDim + i] = x0 * sn + x1 * c
    }
  }

  return A.variable(out, { requiresGrad: false })
}

// --- SwiGLU FFN (Llama-style) ---
// out = down(silu(gate(x)) * up(x))
// Unlike GPT-2's MLP: gelu(fc1(x)) → fc2

function swiGLUForward(x, gate, up, down) {
  const gateOut = linear(x, gate)    // [seqLen, ffnDim]
  const upOut = linear(x, up)        // [seqLen, ffnDim]
  // SiLU activation on gate path
  // silu(x) = x * sigmoid(x), we'll use autograd ops
  // For now compute on CPU (activations are small relative to matmul)
  const gateData = T.contiguous(gateOut.data)
  const activated = T.create(gateData.shape, gateData.dtype)
  for (let i = 0; i < gateData.size; i++) {
    const v = gateData.data[i]
    activated.data[i] = v / (1 + Math.exp(-v)) // silu
  }
  const activatedVar = A.variable(activated, { requiresGrad: false })
  const gated = A.mul(activatedVar, upOut)
  return linear(gated, down)
}

// --- RMSNorm (Llama-style) ---
// RMSNorm(x) = x * gamma / sqrt(mean(x²) + eps)

function rmsNorm(x, gamma, eps = 1e-5) {
  const xData = T.contiguous(x.data)
  const shape = xData.shape
  const rows = shape.length > 1 ? shape[0] : 1
  const cols = shape[shape.length - 1]

  const out = T.create(shape, xData.dtype)
  const gammaData = T.contiguous(gamma.data)

  for (let r = 0; r < rows; r++) {
    let sumSq = 0
    for (let c = 0; c < cols; c++) {
      const v = xData.data[r * cols + c]
      sumSq += v * v
    }
    const rms = Math.sqrt(sumSq / cols + eps)
    for (let c = 0; c < cols; c++) {
      out.data[r * cols + c] = xData.data[r * cols + c] * gammaData.data[c] / rms
    }
  }

  return A.variable(out, { requiresGrad: false })
}

// --- Create model from GGUF config ---

function createGGUFModel(config) {
  const {
    arch, vocabSize, dim, numLayers, numHeads, numKVHeads,
    maxSeqLen, ffnDim, ropeFreqBase, normEps,
  } = config

  const headDim = dim / numHeads
  const kvHeads = numKVHeads ?? numHeads
  const kvDim = kvHeads * headDim
  const actualFfnDim = ffnDim ?? Math.round(dim * 8 / 3 / 256) * 256 // Llama default

  // Embedding (no position embedding for RoPE models)
  const tokenWeight = A.variable(T.zeros([vocabSize, dim]), { requiresGrad: false })

  // Blocks
  const blocks = []
  for (let i = 0; i < numLayers; i++) {
    const block = {
      ln1Gamma: A.variable(T.ones([dim]), { requiresGrad: false }),
      ln2Gamma: A.variable(T.ones([dim]), { requiresGrad: false }),
      mha: {
        qProj:   { weight: A.variable(T.zeros([dim, dim]), { requiresGrad: false }), bias: null },
        kProj:   { weight: A.variable(T.zeros([dim, kvDim]), { requiresGrad: false }), bias: null },
        vProj:   { weight: A.variable(T.zeros([dim, kvDim]), { requiresGrad: false }), bias: null },
        outProj: { weight: A.variable(T.zeros([dim, dim]), { requiresGrad: false }), bias: null },
        numHeads, numKVHeads: kvHeads, headDim, dim,
      },
    }

    if (arch === 'llama') {
      // SwiGLU: gate + up + down
      block.ffnGate = { weight: A.variable(T.zeros([dim, actualFfnDim]), { requiresGrad: false }), bias: null }
      block.ffnUp   = { weight: A.variable(T.zeros([dim, actualFfnDim]), { requiresGrad: false }), bias: null }
      block.ffnDown  = { weight: A.variable(T.zeros([actualFfnDim, dim]), { requiresGrad: false }), bias: null }
    } else {
      block.ffn1 = { weight: A.variable(T.zeros([dim, actualFfnDim]), { requiresGrad: false }), bias: null }
      block.ffn2 = { weight: A.variable(T.zeros([actualFfnDim, dim]), { requiresGrad: false }), bias: null }
    }

    blocks.push(block)
  }

  const lnFGamma = A.variable(T.ones([dim]), { requiresGrad: false })
  const lnFBeta = arch !== 'llama' ? A.variable(T.zeros([dim]), { requiresGrad: false }) : null

  // Separate lm_head if not weight-tied
  const lmHead = A.variable(T.zeros([dim, vocabSize]), { requiresGrad: false })

  // RoPE tables (for non-GPT2 architectures)
  const rope = arch !== 'gpt2' ? precomputeRoPE(headDim, maxSeqLen, ropeFreqBase) : null

  return {
    embedding: { tokenWeight, posWeight: null },
    blocks,
    lnFGamma, lnFBeta,
    lmHead,
    rope,
    config: { ...config, headDim, kvDim, ffnDim: actualFfnDim },
    arch,
    normEps,
  }
}

// --- Load weights from parsed GGUF into model ---

function loadWeights(parsed, model) {
  const weightMap = getWeightMap(model.arch)
  const loaded = []
  const skipped = []

  for (const tensorInfo of parsed.tensors) {
    const path = resolveWeight(tensorInfo.name, weightMap)
    if (!path) {
      skipped.push(tensorInfo.name)
      continue
    }

    // Navigate to the target variable
    const target = resolvePath(model, path)
    if (!target) {
      skipped.push(`${tensorInfo.name} → ${path} (not found in model)`)
      continue
    }

    // Dequantize and load
    const data = dequantizeTensor(parsed, tensorInfo)

    // GGUF stores weights in row-major, but some need transposing
    // Weight matrices in GGUF for Llama are [outDim, inDim] — we need [inDim, outDim]
    const isWeight = path.endsWith('.weight') && !path.includes('embd') && !path.includes('norm') && !path.includes('Gamma')
    const targetData = target.data

    if (isWeight && tensorInfo.shape.length === 2) {
      // Transpose: GGUF [outDim, inDim] → Smith [inDim, outDim]
      const [rows, cols] = tensorInfo.shape
      const expected = targetData.shape
      if (expected[0] === cols && expected[1] === rows) {
        // Need transpose
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            targetData.data[c * rows + r] = data[r * cols + c]
          }
        }
      } else {
        // Shape matches, direct copy
        targetData.data.set(data)
      }
    } else {
      targetData.data.set(data)
    }

    loaded.push(tensorInfo.name)
  }

  // If no separate lm_head was loaded, weight-tie with token embedding
  if (!loaded.includes('output.weight')) {
    const tokData = model.embedding.tokenWeight.data
    model.lmHead.data = tokData // share buffer
    model.weightTied = true
  }

  return { loaded, skipped }
}

// Navigate dotted path like "blocks.0.mha.qProj.weight" into an object
function resolvePath(obj, path) {
  const parts = path.split('.')
  let current = obj
  for (const part of parts) {
    if (current === null || current === undefined) return null
    if (/^\d+$/.test(part)) {
      current = current[parseInt(part)]
    } else {
      current = current[part]
    }
  }
  return current
}

// --- Forward pass for Llama-style models ---

function forwardLlama(model, tokenIds) {
  const seqLen = tokenIds.length
  const { dim, numHeads, headDim } = model.config
  const kvHeads = model.config.numKVHeads

  // Token embedding (no position embedding — RoPE handles positions)
  let x = A.embedding(tokenIds, model.embedding.tokenWeight)

  for (let i = 0; i < model.blocks.length; i++) {
    const block = model.blocks[i]

    // Pre-norm (RMSNorm for Llama)
    const norm1 = rmsNorm(x, block.ln1Gamma, model.normEps)

    // Q, K, V projections
    let Q = linear(norm1, block.mha.qProj) // [seqLen, dim]
    let K = linear(norm1, block.mha.kProj) // [seqLen, kvDim]
    let V = linear(norm1, block.mha.vProj) // [seqLen, kvDim]

    // Apply RoPE to Q and K
    // Reshape to [seqLen, headDim] per head, apply RoPE, reshape back
    // For simplicity, apply to the full Q and K
    Q = applyRoPE(Q, model.rope, 0)
    K = applyRoPE(K, model.rope, 0)

    // Reshape to multi-head: [seqLen, dim] → [numHeads, seqLen, headDim]
    const Qh = A.transpose(A.reshape(Q, [seqLen, numHeads, headDim]), [1, 0, 2])

    // GQA: K and V have fewer heads, need to repeat
    let Kh, Vh
    if (kvHeads < numHeads) {
      // Reshape K: [seqLen, kvDim] → [kvHeads, seqLen, headDim]
      const kShaped = A.transpose(A.reshape(K, [seqLen, kvHeads, headDim]), [1, 0, 2])
      const vShaped = A.transpose(A.reshape(V, [seqLen, kvHeads, headDim]), [1, 0, 2])
      // Repeat KV heads to match Q heads
      Kh = repeatKVHeads(kShaped, numHeads / kvHeads)
      Vh = repeatKVHeads(vShaped, numHeads / kvHeads)
    } else {
      Kh = A.transpose(A.reshape(K, [seqLen, numHeads, headDim]), [1, 0, 2])
      Vh = A.transpose(A.reshape(V, [seqLen, numHeads, headDim]), [1, 0, 2])
    }

    // Flash attention (causal)
    const attnOut = A.flashAttention(Qh, Kh, Vh, true)
    const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [seqLen, dim])
    const projected = linear(concatenated, block.mha.outProj)
    const x2 = A.add(x, projected)

    // FFN with SwiGLU
    const norm2 = rmsNorm(x2, block.ln2Gamma, model.normEps)
    const ffnOut = swiGLUForward(norm2, block.ffnGate, block.ffnUp, block.ffnDown)
    x = A.add(x2, ffnOut)
  }

  // Final norm
  x = rmsNorm(x, model.lnFGamma, model.normEps)

  // Output projection
  let logits
  if (model.weightTied) {
    logits = A.matmul(x, A.transpose(model.embedding.tokenWeight))
  } else {
    logits = linear(x, { weight: model.lmHead, bias: null })
  }

  return { logits }
}

// Repeat KV heads for GQA: [kvHeads, seqLen, headDim] → [numHeads, seqLen, headDim]
function repeatKVHeads(kv, repeats) {
  if (repeats === 1) return kv
  const kvData = T.contiguous(kv.data)
  const [kvHeads, seqLen, headDim] = kvData.shape
  const numHeads = kvHeads * repeats
  const out = T.create([numHeads, seqLen, headDim], kvData.dtype)

  for (let h = 0; h < kvHeads; h++) {
    const srcOff = h * seqLen * headDim
    for (let r = 0; r < repeats; r++) {
      const dstOff = (h * repeats + r) * seqLen * headDim
      out.data.set(kvData.data.subarray(srcOff, srcOff + seqLen * headDim), dstOff)
    }
  }

  return A.variable(out, { requiresGrad: false })
}

// --- Top-level load function ---

async function loadGGUF(path) {
  const fileData = await Bun.file(path).arrayBuffer()
  const parsed = parseGGUF(fileData)
  const config = extractConfig(parsed.metadata)
  const model = createGGUFModel(config)
  const { loaded, skipped } = loadWeights(parsed, model)

  return {
    model,
    config,
    metadata: parsed.metadata,
    loaded,
    skipped,
    forward: (tokenIds) => {
      if (config.arch === 'llama') return forwardLlama(model, tokenIds)
      // For GPT-2 style models, fall back to the standard forward
      throw new Error(`Forward not implemented for ${config.arch} via GGUF loader`)
    },
  }
}

export {
  loadGGUF, parseGGUF, createGGUFModel, loadWeights,
  precomputeRoPE, applyRoPE, rmsNorm, swiGLUForward,
  repeatKVHeads, forwardLlama,
  getWeightMap, resolveWeight, resolvePath,
  LLAMA_MAP, PHI_MAP, GPT2_MAP,
}
