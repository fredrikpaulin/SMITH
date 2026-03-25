// smith/src/clip.js
// CLIP (Contrastive Language-Image Pre-training) model builder + weight loader.
// Supports ViT-B/32, ViT-B/16, ViT-L/14 vision encoders + transformer text encoder.
// Loads from OpenAI-format safetensors files.

import * as T from './tensor.js'
import * as A from './autograd.js'
import { createLinear, linear, createMultiHeadAttention, linearParams } from './nn.js'
import { parseSafetensors, readTensor } from './safetensors.js'

// --- Weight loading helpers ---

function f16ToF32(h) {
  const sign = (h >> 15) & 1
  const exp = (h >> 10) & 0x1f
  const mant = h & 0x3ff
  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024)
  }
  if (exp === 31) return mant === 0 ? (sign ? -Infinity : Infinity) : NaN
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024)
}

function getF32(parsed, name) {
  const t = readTensor(parsed, name)
  if (t.dtype === 'F32') return { data: t.data, shape: t.shape }
  if (t.dtype === 'F16') {
    const out = new Float32Array(t.data.length)
    for (let i = 0; i < t.data.length; i++) out[i] = f16ToF32(t.data[i])
    return { data: out, shape: t.shape }
  }
  throw new Error(`Cannot convert ${t.dtype} to f32 for tensor ${name}`)
}

function loadIntoParam(param, srcData) {
  if (param.data.size !== srcData.length) {
    throw new Error(`Size mismatch: param has ${param.data.size}, source has ${srcData.length}`)
  }
  if (srcData instanceof Float32Array) param.data.data.set(srcData)
  else for (let i = 0; i < srcData.length; i++) param.data.data[i] = srcData[i]
}

// --- CLIP configurations ---

const CLIP_CONFIGS = {
  'ViT-B/32': {
    vision: { dim: 768, layers: 12, heads: 12, patchSize: 32, imageSize: 224 },
    text: { dim: 512, layers: 12, heads: 8, contextLen: 77, vocabSize: 49408 },
    embedDim: 512,
  },
  'ViT-B/16': {
    vision: { dim: 768, layers: 12, heads: 12, patchSize: 16, imageSize: 224 },
    text: { dim: 512, layers: 12, heads: 8, contextLen: 77, vocabSize: 49408 },
    embedDim: 512,
  },
  'ViT-L/14': {
    vision: { dim: 1024, layers: 24, heads: 16, patchSize: 14, imageSize: 224 },
    text: { dim: 768, layers: 12, heads: 12, contextLen: 77, vocabSize: 49408 },
    embedDim: 768,
  },
}

// --- ViT (Vision Transformer) builder ---

function createVisionTransformer(cfg) {
  const { dim, layers, heads, patchSize, imageSize } = cfg
  const numPatches = (imageSize / patchSize) ** 2

  // Patch embedding: conv with kernel=patchSize, stride=patchSize
  // Equivalent to splitting image into non-overlapping patches and projecting
  const patchConvWeight = A.variable(
    T.randn([dim, 3, patchSize, patchSize]),
    { requiresGrad: true }
  )
  const patchConvBias = A.variable(T.zeros([dim]), { requiresGrad: true })

  // Class token + positional embedding
  const classToken = A.variable(T.randn([1, dim]), { requiresGrad: true })
  const posEmbed = A.variable(T.randn([numPatches + 1, dim]), { requiresGrad: true })

  // Pre-projection layernorm
  const lnPre = {
    gamma: A.variable(T.ones([dim]), { requiresGrad: true }),
    beta: A.variable(T.zeros([dim]), { requiresGrad: true }),
  }

  // Transformer blocks (pre-norm, no causal mask)
  const blocks = []
  for (let i = 0; i < layers; i++) {
    blocks.push({
      ln1Gamma: A.variable(T.ones([dim]), { requiresGrad: true }),
      ln1Beta: A.variable(T.zeros([dim]), { requiresGrad: true }),
      ln2Gamma: A.variable(T.ones([dim]), { requiresGrad: true }),
      ln2Beta: A.variable(T.zeros([dim]), { requiresGrad: true }),
      mha: createMultiHeadAttention(dim, heads),
      ffn1: createLinear(dim, dim * 4),
      ffn2: createLinear(dim * 4, dim),
    })
  }

  // Post-projection layernorm
  const lnPost = {
    gamma: A.variable(T.ones([dim]), { requiresGrad: true }),
    beta: A.variable(T.zeros([dim]), { requiresGrad: true }),
  }

  // Visual projection
  const projection = A.variable(T.randn([dim, CLIP_CONFIGS['ViT-B/32'].embedDim]), { requiresGrad: true })

  return {
    patchConvWeight, patchConvBias,
    classToken, posEmbed, lnPre, blocks, lnPost, projection,
    config: cfg,
  }
}

// --- Text Transformer builder ---

function createTextTransformer(cfg) {
  const { dim, layers, heads, contextLen, vocabSize } = cfg

  const tokenEmbed = A.variable(T.randn([vocabSize, dim]), { requiresGrad: true })
  const posEmbed = A.variable(T.randn([contextLen, dim]), { requiresGrad: true })

  const blocks = []
  for (let i = 0; i < layers; i++) {
    blocks.push({
      ln1Gamma: A.variable(T.ones([dim]), { requiresGrad: true }),
      ln1Beta: A.variable(T.zeros([dim]), { requiresGrad: true }),
      ln2Gamma: A.variable(T.ones([dim]), { requiresGrad: true }),
      ln2Beta: A.variable(T.zeros([dim]), { requiresGrad: true }),
      mha: createMultiHeadAttention(dim, heads),
      ffn1: createLinear(dim, dim * 4),
      ffn2: createLinear(dim * 4, dim),
    })
  }

  const lnFinal = {
    gamma: A.variable(T.ones([dim]), { requiresGrad: true }),
    beta: A.variable(T.zeros([dim]), { requiresGrad: true }),
  }

  const textProjection = A.variable(T.randn([dim, 512]), { requiresGrad: true })

  return {
    tokenEmbed, posEmbed, blocks, lnFinal, textProjection,
    config: cfg,
  }
}

// --- CLIP model ---

function createCLIP(variant = 'ViT-B/32') {
  const cfg = CLIP_CONFIGS[variant]
  if (!cfg) throw new Error(`Unknown CLIP variant: ${variant}. Use: ${Object.keys(CLIP_CONFIGS).join(', ')}`)

  const visual = createVisionTransformer(cfg.vision)
  // Fix projection dims
  visual.projection = A.variable(T.randn([cfg.vision.dim, cfg.embedDim]), { requiresGrad: true })

  const text = createTextTransformer(cfg.text)
  text.textProjection = A.variable(T.randn([cfg.text.dim, cfg.embedDim]), { requiresGrad: true })

  const logitScale = A.variable(T.scalar(Math.log(1 / 0.07)), { requiresGrad: true })

  return {
    visual, text, logitScale,
    config: { variant, ...cfg },
  }
}

// --- Vision forward ---

function forwardVision(model, x) {
  // x: tensor [N, 3, imageSize, imageSize] — NCHW (raw tensor, not variable)
  const v = model.visual
  const { patchSize } = v.config
  // Accept both raw tensor and variable
  const tensor = x.shape ? x : x.data
  const [n, , h, w] = tensor.shape
  const numPatches = (h / patchSize) * (w / patchSize)

  const xVar = x.shape ? A.variable(x, { requiresGrad: false }) : x

  // Patch embedding via conv2d: [N, 3, H, W] → [N, dim, gridH, gridW]
  let patches = A.conv2d(
    xVar, v.patchConvWeight, v.patchConvBias,
    { stride: patchSize, padding: 0 }
  )

  // Reshape to [N, dim, numPatches] → [N, numPatches, dim]
  const dim = v.config.dim
  patches = A.reshape(patches, [n, dim, numPatches])
  patches = A.transpose(patches, [0, 2, 1])  // [N, numPatches, dim]

  // Process each sample (our transformer blocks don't support batch)
  const results = []
  for (let i = 0; i < n; i++) {
    // Extract sample: [numPatches, dim]
    let seq = extractSample(patches, i, numPatches, dim)

    // Prepend class token: [numPatches+1, dim]
    seq = prependClassToken(seq, v.classToken, numPatches, dim)

    // Add positional embedding
    seq = A.add(seq, v.posEmbed)

    // Pre-layernorm
    seq = A.layernorm(seq, v.lnPre.gamma, v.lnPre.beta)

    // Transformer blocks (non-causal attention)
    for (const block of v.blocks) {
      const norm1 = A.layernorm(seq, block.ln1Gamma, block.ln1Beta)
      const { output: attnOut } = A.flashAttention
        ? flashMHA(norm1, block.mha, false)
        : stdMHA(norm1, block.mha, null)
      seq = A.add(seq, attnOut)

      const norm2 = A.layernorm(seq, block.ln2Gamma, block.ln2Beta)
      const ffnHidden = A.gelu(linear(norm2, block.ffn1))
      const ffnOut = linear(ffnHidden, block.ffn2)
      seq = A.add(seq, ffnOut)
    }

    // Post-layernorm on class token only
    const classOut = extractRow(seq, 0, dim)
    const normed = A.layernorm(classOut, v.lnPost.gamma, v.lnPost.beta)

    // Project to embed space
    const projected = A.matmul(normed, v.projection)
    results.push(projected)
  }

  // Stack results [N, embedDim]
  return stackRows(results)
}

// --- Text forward ---

function forwardText(model, tokenIds) {
  // tokenIds: array of ints [seqLen] (single sequence)
  const t = model.text
  const seqLen = tokenIds.length
  const dim = t.config.dim

  // Token embedding: embedding(indices, weight)
  let x = A.embedding(tokenIds, t.tokenEmbed)

  // Add positional embedding (slice to seqLen)
  const posSlice = sliceRows(t.posEmbed, 0, seqLen, dim)
  x = A.add(x, posSlice)

  // Causal mask
  const mask = T.tensor(
    Array.from({ length: seqLen * seqLen }, (_, idx) => {
      const i = Math.floor(idx / seqLen), j = idx % seqLen
      return j > i ? -Infinity : 0
    }),
    [seqLen, seqLen]
  )

  // Transformer blocks (causal attention)
  for (const block of t.blocks) {
    const norm1 = A.layernorm(x, block.ln1Gamma, block.ln1Beta)
    const { output: attnOut } = stdMHA(norm1, block.mha, mask)
    x = A.add(x, attnOut)

    const norm2 = A.layernorm(x, block.ln2Gamma, block.ln2Beta)
    const ffnHidden = A.gelu(linear(norm2, block.ffn1))
    const ffnOut = linear(ffnHidden, block.ffn2)
    x = A.add(x, ffnOut)
  }

  // Final layernorm
  x = A.layernorm(x, t.lnFinal.gamma, t.lnFinal.beta)

  // Take features from the EOT token (last token)
  const eotIdx = seqLen - 1
  const eotFeatures = extractRow(x, eotIdx, dim)

  // Project to embed space
  return A.matmul(eotFeatures, t.textProjection)
}

// --- Standard MHA (replicates nn.js pattern without batch) ---

function stdMHA(x, layer, mask) {
  const seqLen = x.data.shape[0]
  const { numHeads, headDim, dim } = layer

  const Q = linear(x, layer.qProj)
  const K = linear(x, layer.kProj)
  const V = linear(x, layer.vProj)

  const Qh = A.transpose(A.reshape(Q, [seqLen, numHeads, headDim]), [1, 0, 2])
  const Kh = A.transpose(A.reshape(K, [seqLen, numHeads, headDim]), [1, 0, 2])
  const Vh = A.transpose(A.reshape(V, [seqLen, numHeads, headDim]), [1, 0, 2])

  const dk = headDim
  const scaleFactor = 1 / Math.sqrt(dk)
  const KhT = A.transpose(Kh, [0, 2, 1])
  let scores = A.scale(A.matmul(Qh, KhT), scaleFactor)

  if (mask) {
    const maskVar = A.variable(mask, { requiresGrad: false })
    scores = A.add(scores, maskVar)
  }

  const weights = A.softmax(scores, -1)
  const attnOut = A.matmul(weights, Vh)

  const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [seqLen, dim])
  return { output: linear(concatenated, layer.outProj) }
}

// Flash MHA (non-causal for vision)
function flashMHA(x, layer, causal) {
  const seqLen = x.data.shape[0]
  const { numHeads, headDim, dim } = layer

  const Q = linear(x, layer.qProj)
  const K = linear(x, layer.kProj)
  const V = linear(x, layer.vProj)

  const Qh = A.transpose(A.reshape(Q, [seqLen, numHeads, headDim]), [1, 0, 2])
  const Kh = A.transpose(A.reshape(K, [seqLen, numHeads, headDim]), [1, 0, 2])
  const Vh = A.transpose(A.reshape(V, [seqLen, numHeads, headDim]), [1, 0, 2])

  const attnOut = A.flashAttention(Qh, Kh, Vh, causal)
  const concatenated = A.reshape(A.transpose(attnOut, [1, 0, 2]), [seqLen, dim])
  return { output: linear(concatenated, layer.outProj) }
}

// --- Tensor helpers ---

function extractSample(batched, idx, seqLen, dim) {
  // Extract [seqLen, dim] from [N, seqLen, dim] at batch index idx
  const data = new Float32Array(seqLen * dim)
  const src = batched.data.data
  const offset = idx * seqLen * dim
  data.set(src.subarray(offset, offset + seqLen * dim))
  return A.variable(T.tensor(Array.from(data), [seqLen, dim]), { requiresGrad: false })
}

function prependClassToken(seq, classToken, numPatches, dim) {
  // [numPatches, dim] → [numPatches+1, dim] with class token at position 0
  const newLen = numPatches + 1
  const data = new Float32Array(newLen * dim)
  data.set(classToken.data.data.subarray(0, dim), 0)
  data.set(seq.data.data.subarray(0, numPatches * dim), dim)
  return A.variable(T.tensor(Array.from(data), [newLen, dim]), { requiresGrad: false })
}

function extractRow(x, rowIdx, dim) {
  // Extract [1, dim] from [seqLen, dim]
  const data = new Float32Array(dim)
  data.set(x.data.data.subarray(rowIdx * dim, (rowIdx + 1) * dim))
  return A.variable(T.tensor(Array.from(data), [1, dim]), { requiresGrad: false })
}

function sliceRows(param, start, count, dim) {
  const data = new Float32Array(count * dim)
  data.set(param.data.data.subarray(start * dim, (start + count) * dim))
  return A.variable(T.tensor(Array.from(data), [count, dim]), { requiresGrad: false })
}

function stackRows(rows) {
  // Stack array of [1, embedDim] → [N, embedDim]
  const n = rows.length
  const embedDim = rows[0].data.shape[1]
  const data = new Float32Array(n * embedDim)
  for (let i = 0; i < n; i++) {
    data.set(rows[i].data.data.subarray(0, embedDim), i * embedDim)
  }
  return A.variable(T.tensor(Array.from(data), [n, embedDim]), { requiresGrad: false })
}

// --- CLIP similarity ---

function clipSimilarity(imageFeatures, textFeatures, logitScale) {
  // Normalize
  const imgNorm = l2Normalize(imageFeatures)
  const txtNorm = l2Normalize(textFeatures)

  // logit_scale * (image @ text^T)
  const scale = Math.exp(logitScale.data.data[0])
  const txtT = A.transpose(txtNorm, [1, 0])
  const similarity = A.scale(A.matmul(imgNorm, txtT), scale)
  return similarity
}

function l2Normalize(x) {
  // x: [N, D] → normalize each row
  const [n, d] = x.data.shape
  const data = new Float32Array(n * d)
  const src = x.data.data
  for (let i = 0; i < n; i++) {
    let sumSq = 0
    for (let j = 0; j < d; j++) sumSq += src[i * d + j] ** 2
    const norm = Math.sqrt(sumSq) + 1e-8
    for (let j = 0; j < d; j++) data[i * d + j] = src[i * d + j] / norm
  }
  return A.variable(T.tensor(Array.from(data), [n, d]), { requiresGrad: false })
}

// --- Collect all parameters ---

function clipParams(model) {
  const params = []

  function addBlock(block) {
    params.push(block.ln1Gamma, block.ln1Beta, block.ln2Gamma, block.ln2Beta)
    params.push(...linearParams(block.mha.qProj))
    params.push(...linearParams(block.mha.kProj))
    params.push(...linearParams(block.mha.vProj))
    params.push(...linearParams(block.mha.outProj))
    params.push(...linearParams(block.ffn1))
    params.push(...linearParams(block.ffn2))
  }

  // Vision
  const v = model.visual
  params.push(v.patchConvWeight, v.patchConvBias, v.classToken, v.posEmbed)
  params.push(v.lnPre.gamma, v.lnPre.beta)
  for (const block of v.blocks) addBlock(block)
  params.push(v.lnPost.gamma, v.lnPost.beta, v.projection)

  // Text
  const t = model.text
  params.push(t.tokenEmbed, t.posEmbed)
  for (const block of t.blocks) addBlock(block)
  params.push(t.lnFinal.gamma, t.lnFinal.beta, t.textProjection)

  params.push(model.logitScale)
  return params
}

// --- Weight loading (OpenAI CLIP safetensors format) ---

function loadLinearWeights(parsed, prefix, layer) {
  // OpenAI CLIP linear: weight is [outDim, inDim], we need [inDim, outDim]
  const w = getF32(parsed, `${prefix}.weight`)
  const [outDim, inDim] = w.shape
  const transposed = new Float32Array(inDim * outDim)
  for (let r = 0; r < outDim; r++) {
    for (let c = 0; c < inDim; c++) {
      transposed[c * outDim + r] = w.data[r * inDim + c]
    }
  }
  loadIntoParam(layer.weight, transposed)
  if (layer.bias && parsed.tensors[`${prefix}.bias`]) {
    const b = getF32(parsed, `${prefix}.bias`)
    loadIntoParam(layer.bias, b.data)
  }
}

function loadMHAWeights(parsed, prefix, mha) {
  const dim = mha.dim
  // OpenAI CLIP uses in_proj_weight [3*dim, dim] and in_proj_bias [3*dim]
  if (parsed.tensors[`${prefix}.in_proj_weight`]) {
    const w = getF32(parsed, `${prefix}.in_proj_weight`)
    const b = parsed.tensors[`${prefix}.in_proj_bias`]
      ? getF32(parsed, `${prefix}.in_proj_bias`) : null

    // Split [3*dim, dim] → Q [dim, dim], K [dim, dim], V [dim, dim]
    // Source is [outDim, inDim], Smith needs [inDim, outDim]
    for (const [proj, offset] of [['qProj', 0], ['kProj', dim], ['vProj', 2 * dim]]) {
      const t = new Float32Array(dim * dim)
      for (let r = 0; r < dim; r++) {
        for (let c = 0; c < dim; c++) {
          t[c * dim + r] = w.data[(offset + r) * dim + c]
        }
      }
      loadIntoParam(mha[proj].weight, t)
      if (b && mha[proj].bias) {
        loadIntoParam(mha[proj].bias, b.data.subarray(offset, offset + dim))
      }
    }
  }

  // out_proj
  loadLinearWeights(parsed, `${prefix}.out_proj`, mha.outProj)
}

function loadBlockWeights(parsed, prefix, block) {
  // Layernorms
  const ln1w = getF32(parsed, `${prefix}.ln_1.weight`)
  const ln1b = getF32(parsed, `${prefix}.ln_1.bias`)
  loadIntoParam(block.ln1Gamma, ln1w.data)
  loadIntoParam(block.ln1Beta, ln1b.data)

  const ln2w = getF32(parsed, `${prefix}.ln_2.weight`)
  const ln2b = getF32(parsed, `${prefix}.ln_2.bias`)
  loadIntoParam(block.ln2Gamma, ln2w.data)
  loadIntoParam(block.ln2Beta, ln2b.data)

  // MHA
  loadMHAWeights(parsed, `${prefix}.attn`, block.mha)

  // FFN (OpenAI CLIP uses c_fc and c_proj)
  loadLinearWeights(parsed, `${prefix}.mlp.c_fc`, block.ffn1)
  loadLinearWeights(parsed, `${prefix}.mlp.c_proj`, block.ffn2)
}

function mapCLIPWeights(parsed, model) {
  const v = model.visual
  const t = model.text

  // --- Vision ---
  // Patch embedding conv
  const convW = getF32(parsed, 'visual.conv1.weight')
  loadIntoParam(v.patchConvWeight, convW.data)
  if (parsed.tensors['visual.conv1.bias']) {
    const convB = getF32(parsed, 'visual.conv1.bias')
    loadIntoParam(v.patchConvBias, convB.data)
  }

  // Class token
  const cls = getF32(parsed, 'visual.class_embedding')
  loadIntoParam(v.classToken, cls.data)

  // Positional embedding
  const vPos = getF32(parsed, 'visual.positional_embedding')
  loadIntoParam(v.posEmbed, vPos.data)

  // Pre-LN
  const lnPreW = getF32(parsed, 'visual.ln_pre.weight')
  const lnPreB = getF32(parsed, 'visual.ln_pre.bias')
  loadIntoParam(v.lnPre.gamma, lnPreW.data)
  loadIntoParam(v.lnPre.beta, lnPreB.data)

  // Vision transformer blocks
  for (let i = 0; i < v.blocks.length; i++) {
    loadBlockWeights(parsed, `visual.transformer.resblocks.${i}`, v.blocks[i])
  }

  // Post-LN
  const lnPostW = getF32(parsed, 'visual.ln_post.weight')
  const lnPostB = getF32(parsed, 'visual.ln_post.bias')
  loadIntoParam(v.lnPost.gamma, lnPostW.data)
  loadIntoParam(v.lnPost.beta, lnPostB.data)

  // Visual projection
  const vProj = getF32(parsed, 'visual.proj')
  loadIntoParam(v.projection, vProj.data)

  // --- Text ---
  // Token embedding
  const tokEmb = getF32(parsed, 'token_embedding.weight')
  loadIntoParam(t.tokenEmbed, tokEmb.data)

  // Positional embedding
  const tPos = getF32(parsed, 'positional_embedding')
  loadIntoParam(t.posEmbed, tPos.data)

  // Text transformer blocks
  for (let i = 0; i < t.blocks.length; i++) {
    loadBlockWeights(parsed, `transformer.resblocks.${i}`, t.blocks[i])
  }

  // Final LN
  const lnFW = getF32(parsed, 'ln_final.weight')
  const lnFB = getF32(parsed, 'ln_final.bias')
  loadIntoParam(t.lnFinal.gamma, lnFW.data)
  loadIntoParam(t.lnFinal.beta, lnFB.data)

  // Text projection
  const tProj = getF32(parsed, 'text_projection')
  loadIntoParam(t.textProjection, tProj.data)

  // Logit scale
  if (parsed.tensors['logit_scale']) {
    const ls = getF32(parsed, 'logit_scale')
    model.logitScale.data.data[0] = ls.data[0]
  }
}

// --- High-level loader ---

async function loadCLIP(path, opts = {}) {
  const { variant = 'ViT-B/32' } = opts
  const buf = await Bun.file(path).arrayBuffer()
  const parsed = parseSafetensors(buf)

  const model = createCLIP(variant)
  mapCLIPWeights(parsed, model)

  return {
    model,
    config: model.config,
    encodeImage: (x) => forwardVision(model, x),
    encodeText: (tokenIds) => forwardText(model, tokenIds),
    similarity: (imgFeats, txtFeats) => clipSimilarity(imgFeats, txtFeats, model.logitScale),
    params: () => clipParams(model),
  }
}

export {
  createCLIP, forwardVision, forwardText,
  clipSimilarity, l2Normalize, clipParams,
  mapCLIPWeights, loadCLIP,
  CLIP_CONFIGS,
}
