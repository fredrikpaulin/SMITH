// examples/autoresearch/model.js
// GPT model ported from Karpathy's autoresearch train.py.
// Uses Smith's autograd: RoPE, RMSNorm, GQA flash attention,
// ReluSquared MLP, value embeddings, logit soft-capping.

import smith from '../../src/index.js'

const {
  variable, param, tensor, zeros, ones,
  embedding, rmsNorm, rope, flashAttention,
  reluSquared, sigmoid, tanh,
  matmul, add, mul, scale, sum, reshape, transpose,
  crossEntropy, backward, zeroGrad, noGrad,
  precomputeRoPE,
} = smith

// --- Config ---

const defaultConfig = {
  seqLen: 512,
  vocabSize: 4096,
  nLayer: 4,
  nHead: 4,
  nKVHead: 4,
  nEmbd: 256,
  windowPattern: 'SSSL',
}

// --- Helpers ---

function hasVE(layerIdx, nLayer) {
  return layerIdx % 2 === (nLayer - 1) % 2
}

// Tile RoPE cos/sin tables so [nHeads, T, headDim] flattened to [nHeads*T, headDim]
// gets correct per-position rotations (positions 0..T-1 repeated per head).
function tileRoPETable(ropeTable, seqLen, numHeads) {
  if (numHeads === 1) return ropeTable
  const halfDim = ropeTable.cos.shape[1]
  const srcCos = ropeTable.cos.data  // Float32Array
  const srcSin = ropeTable.sin.data
  const totalRows = numHeads * seqLen
  const cosData = new Float32Array(totalRows * halfDim)
  const sinData = new Float32Array(totalRows * halfDim)
  for (let h = 0; h < numHeads; h++) {
    const off = h * seqLen * halfDim
    cosData.set(srcCos.subarray(0, seqLen * halfDim), off)
    sinData.set(srcSin.subarray(0, seqLen * halfDim), off)
  }
  return {
    cos: tensor(cosData, [totalRows, halfDim]),
    sin: tensor(sinData, [totalRows, halfDim]),
  }
}

function computeWindowSizes(config) {
  const pattern = config.windowPattern.toUpperCase()
  const longWin = config.seqLen
  const shortWin = Math.floor(longWin / 2)
  const sizes = []
  for (let i = 0; i < config.nLayer; i++) {
    const c = pattern[i % pattern.length]
    sizes.push(c === 'L' ? longWin : shortWin)
  }
  sizes[sizes.length - 1] = longWin  // last layer always full
  return sizes
}

// --- Model creation ---

function createModel(config = {}) {
  const cfg = { ...defaultConfig, ...config }
  const { vocabSize, nEmbd, nLayer, nHead, nKVHead, seqLen } = cfg
  const headDim = Math.floor(nEmbd / nHead)
  const kvDim = nKVHead * headDim
  const ffnDim = Math.floor(8 * nEmbd / 3)
  const veGateChannels = Math.min(32, nEmbd)

  // Embedding
  const wte = param([vocabSize, nEmbd], () => smith.randn([vocabSize, nEmbd]))

  // Per-layer scalars
  const residLambdas = variable(ones([nLayer]), { requiresGrad: true })
  const x0Lambdas = variable(tensor(new Array(nLayer).fill(0.1), [nLayer]), { requiresGrad: true })

  // Blocks
  const blocks = []
  for (let i = 0; i < nLayer; i++) {
    const block = {
      // Attention projections
      cQ: param([nEmbd, nHead * headDim]),
      cK: param([nEmbd, kvDim]),
      cV: param([nEmbd, kvDim]),
      cProj: param([nHead * headDim, nEmbd]),
      // MLP (SwiGLU)
      cFcUp: param([nEmbd, ffnDim]),
      cFcGate: param([nEmbd, ffnDim]),
      cMlpProj: param([ffnDim, nEmbd]),
      // RMSNorm gammas
      normAttn: variable(ones([nEmbd]), { requiresGrad: true }),
      normMlp: variable(ones([nEmbd]), { requiresGrad: true }),
    }
    if (hasVE(i, nLayer)) {
      block.veEmbed = param([vocabSize, kvDim], () => smith.randn([vocabSize, kvDim]))
      block.veGate = param([veGateChannels, nKVHead])
    }
    blocks.push(block)
  }

  // LM head + final norm
  const lmHead = param([nEmbd, vocabSize])
  const normF = variable(ones([nEmbd]), { requiresGrad: true })

  // RoPE base table (headDim-sized, seqLen positions)
  const ropeBase = precomputeRoPE(headDim, seqLen)

  // Window sizes
  const windowSizes = computeWindowSizes(cfg)

  return {
    config: cfg, headDim, kvDim, ffnDim, veGateChannels,
    wte, residLambdas, x0Lambdas,
    blocks, lmHead, normF,
    ropeBase, windowSizes,
  }
}

// --- Init weights (matches reference) ---

function initWeights(model) {
  const { config, blocks } = model
  const s = Math.sqrt(3) * Math.pow(config.nEmbd, -0.5)

  // Embedding: normal(0, 1)
  const wteD = model.wte.data.data
  for (let i = 0; i < wteD.length; i++) wteD[i] = randn()

  // LM head: normal(0, 0.001)
  const lmD = model.lmHead.data.data
  for (let i = 0; i < lmD.length; i++) lmD[i] = 0.001 * randn()

  // Per-layer scalars
  const rl = model.residLambdas.data.data
  const xl = model.x0Lambdas.data.data
  for (let i = 0; i < config.nLayer; i++) { rl[i] = 1.0; xl[i] = 0.1 }

  for (const block of blocks) {
    // Q, K, V: uniform(-s, s)
    uniformInit(block.cQ.data.data, s)
    uniformInit(block.cK.data.data, s)
    uniformInit(block.cV.data.data, s)
    // Proj: zeros
    block.cProj.data.data.fill(0)
    // MLP fc: uniform, proj: zeros
    uniformInit(block.cFcUp.data.data, s)
    uniformInit(block.cFcGate.data.data, s)
    block.cMlpProj.data.data.fill(0)
    // Norm gammas: ones (already set by T.ones)
    // VE
    if (block.veEmbed) uniformInit(block.veEmbed.data.data, s)
    if (block.veGate) block.veGate.data.data.fill(0)  // sigmoid(0)=0.5, *2=1.0 → neutral
  }
}

function randn() {
  // Box-Muller transform
  const u1 = Math.random(), u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2)
}

function uniformInit(arr, s) {
  for (let i = 0; i < arr.length; i++) arr[i] = (Math.random() * 2 - 1) * s
}

// --- Forward pass ---
// Processes one sequence at a time (no batching — Apple Silicon friendly).
// tokens: Int32Array or Array of token IDs, length T
// Returns { logits, loss } where logits is Variable [T, vocabSize]

function forward(model, tokens, targets = null) {
  const { config, headDim, kvDim, veGateChannels, blocks, ropeBase, windowSizes } = model
  const { nHead, nKVHead, nEmbd } = config
  const T = tokens.length

  // Tile RoPE tables for this sequence length (tiled for multi-head flatten)
  const ropeQ = tileRoPETable(ropeBase, T, nHead)
  const ropeK = tileRoPETable(ropeBase, T, nKVHead)

  // Token embedding
  let x = embedding(tokens, model.wte)
  const x0 = x

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i]

    // Pre-norm + residual scaling
    const rl = sliceScalar(model.residLambdas, i)
    const xl = sliceScalar(model.x0Lambdas, i)
    x = add(scale(x, rl), scale(x0, xl))

    // --- Attention ---
    const xNorm = rmsNorm(x, block.normAttn)

    // Q, K, V projections: [T, nEmbd] @ [nEmbd, dim] → [T, dim]
    let q = matmul(xNorm, block.cQ)  // [T, nHead*headDim]
    let k = matmul(xNorm, block.cK)  // [T, kvDim]
    let v = matmul(xNorm, block.cV)  // [T, kvDim]

    // Value residual (ResFormer)
    if (block.veEmbed) {
      const ve = embedding(tokens, block.veEmbed)  // [T, kvDim]
      // Gate: sigmoid(x[:, :veGateChannels] @ veGate) * 2
      const xSlice = sliceCols(xNorm, 0, veGateChannels)  // [T, veGateChannels]
      const gate = scale(sigmoid(matmul(xSlice, block.veGate)), 2.0)  // [T, nKVHead]
      // v = v + gate.unsqueeze(-1) * ve
      // Since gate is [T, nKVHead] and ve is [T, kvDim=nKVHead*headDim],
      // we need to broadcast gate across headDim
      v = add(v, mulGateVE(gate, ve, nKVHead, headDim))
    }

    // Reshape to 3D [nHead, T, headDim] for flash attention
    let q3 = transpose(reshape(q, [T, nHead, headDim]), [1, 0, 2])   // [nHead, T, headDim]
    let k3 = transpose(reshape(k, [T, nKVHead, headDim]), [1, 0, 2]) // [nKVHead, T, headDim]
    let v3 = transpose(reshape(v, [T, nKVHead, headDim]), [1, 0, 2]) // [nKVHead, T, headDim]

    // RoPE — flatten to 2D [nHead*T, headDim], apply with tiled table, reshape back
    q3 = reshape(rope(reshape(q3, [nHead * T, headDim]), ropeQ), [nHead, T, headDim])
    k3 = reshape(rope(reshape(k3, [nKVHead * T, headDim]), ropeK), [nKVHead, T, headDim])

    // Q/K norm — per-head RMSNorm on the headDim axis (works on 3D: last dim = headDim)
    const qNormGamma = variable(ones([headDim]), { requiresGrad: false })
    const kNormGamma = variable(ones([headDim]), { requiresGrad: false })
    q3 = rmsNorm(q3, qNormGamma)
    k3 = rmsNorm(k3, kNormGamma)

    // Flash attention with GQA + sliding window
    const attnOut3 = flashAttention(q3, k3, v3, {
      causal: true,
      numKVHeads: nKVHead,
      windowSize: windowSizes[i],
    })

    // Reshape back to 2D [T, nHead*headDim]
    const attnOut = reshape(transpose(attnOut3, [1, 0, 2]), [T, nHead * headDim])

    // Output projection
    const attnProj = matmul(attnOut, block.cProj)
    x = add(x, attnProj)

    // --- MLP (SwiGLU) ---
    const xMlpNorm = rmsNorm(x, block.normMlp)
    const up = matmul(xMlpNorm, block.cFcUp)
    const gate = matmul(xMlpNorm, block.cFcGate)
    const h = smith.swiglu(up, gate)
    const mlpOut = matmul(h, block.cMlpProj)
    x = add(x, mlpOut)
  }

  // Final norm
  x = rmsNorm(x, model.normF)

  // LM head
  const logits = matmul(x, model.lmHead)  // [T, vocabSize]

  if (targets !== null) {
    const loss = crossEntropy(logits, targets)
    return { logits, loss }
  }
  return { logits }
}

// --- Scalar slicing (extract layer scalar from residLambdas/x0Lambdas) ---
// Since Smith doesn't have per-element indexing as a differentiable op,
// we read the value directly and return it as a JS number for scale().
function sliceScalar(lambdas, idx) {
  return lambdas.data.data[idx]
}

// --- Column slicing: extract first `n` columns from [T, D] ---
function sliceCols(x, start, end) {
  const [rows, cols] = x.data.shape
  const n = end - start
  const out = zeros([rows, n])
  const src = x.data.data, dst = out.data
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < n; c++) {
      dst[r * n + c] = src[r * cols + (start + c)]
    }
  }
  // Non-differentiable slice (VE gate input doesn't need grad through slice)
  return variable(out, { requiresGrad: false })
}

// --- Gate * VE broadcasting ---
// gate: [T, nKVHead], ve: [T, kvDim=nKVHead*headDim]
// output: [T, kvDim] where each head's channels are scaled by gate for that head
function mulGateVE(gate, ve, nKVHead, headDim) {
  const T_len = gate.data.shape[0]
  const kvDim = nKVHead * headDim
  const out = zeros([T_len, kvDim])
  const gd = gate.data.data, vd = ve.data.data, od = out.data
  for (let t = 0; t < T_len; t++) {
    for (let h = 0; h < nKVHead; h++) {
      const g = gd[t * nKVHead + h]
      const base = t * kvDim + h * headDim
      for (let d = 0; d < headDim; d++) {
        od[base + d] = g * vd[base + d]
      }
    }
  }
  return variable(out, {
    _deps: [gate, ve],
    _backward: (grad) => {
      // grad wrt ve: gate * grad (broadcast)
      // grad wrt gate: sum(ve * grad, dim=-1) per head
      const gGrad = zeros([T_len, nKVHead])
      const vGrad = zeros([T_len, kvDim])
      const gg = gGrad.data, vg = vGrad.data
      for (let t = 0; t < T_len; t++) {
        for (let h = 0; h < nKVHead; h++) {
          const g = gd[t * nKVHead + h]
          const base = t * kvDim + h * headDim
          let gSum = 0
          for (let d = 0; d < headDim; d++) {
            const gr = grad.data[base + d]
            vg[base + d] = g * gr
            gSum += vd[base + d] * gr
          }
          gg[t * nKVHead + h] = gSum
        }
      }
      smith.addGrad(gate, gGrad)
      smith.addGrad(ve, vGrad)
    },
  })
}

// --- Collect all params for optimizer ---

function getParamGroups(model) {
  const { config, blocks } = model
  const matrixParams = []
  const embeddingParams = [model.wte]
  const veParams = []
  const lmHeadParams = [model.lmHead]
  const scalarParams = [model.residLambdas, model.x0Lambdas]

  for (const block of blocks) {
    matrixParams.push(block.cQ, block.cK, block.cV, block.cProj)
    matrixParams.push(block.cFcUp, block.cFcGate, block.cMlpProj)
    if (block.veGate) matrixParams.push(block.veGate)
    if (block.veEmbed) veParams.push(block.veEmbed)
  }

  // Group norm gammas as 1D adamw params
  const normParams = [model.normF]
  for (const block of blocks) {
    normParams.push(block.normAttn, block.normMlp)
  }

  return { matrixParams, embeddingParams, veParams, lmHeadParams, scalarParams, normParams }
}

function setupOptimizer(model, opts = {}) {
  const {
    embeddingLr = 0.2,
    unembeddingLr = 0.004,
    matrixLr = 0.02,
    scalarLr = 0.5,
    weightDecay = 0.0,
    adamBetas = [0.8, 0.95],
  } = opts

  const { matrixParams, embeddingParams, veParams, lmHeadParams, scalarParams, normParams } = getParamGroups(model)
  const dim = model.config.nEmbd
  const dmodelScale = Math.pow(dim / 768, -0.5)

  const groups = [
    { kind: 'adamw', params: lmHeadParams, lr: unembeddingLr * dmodelScale, betas: adamBetas, eps: 1e-10, weightDecay: 0.0 },
    { kind: 'adamw', params: embeddingParams, lr: embeddingLr * dmodelScale, betas: adamBetas, eps: 1e-10, weightDecay: 0.0 },
    { kind: 'adamw', params: veParams, lr: embeddingLr * dmodelScale, betas: adamBetas, eps: 1e-10, weightDecay: 0.0 },
    { kind: 'adamw', params: scalarParams, lr: scalarLr, betas: [0.96, 0.95], eps: 1e-10, weightDecay: 0.0 },
    { kind: 'adamw', params: normParams, lr: scalarLr * 0.01, betas: adamBetas, eps: 1e-10, weightDecay: 0.0 },
  ]

  // Group matrix params by shape for Muon
  const byShape = new Map()
  for (const p of matrixParams) {
    const key = p.data.shape.join(',')
    if (!byShape.has(key)) byShape.set(key, [])
    byShape.get(key).push(p)
  }
  for (const params of byShape.values()) {
    groups.push({
      kind: 'muon', params, lr: matrixLr,
      momentum: 0.95, nsSteps: 5, beta2: 0.95, weightDecay,
    })
  }

  // Filter empty groups
  const nonEmpty = groups.filter(g => g.params.length > 0)
  const opt = smith.createMuonAdamW(nonEmpty)

  // Store initial LRs for scheduling
  for (const g of nonEmpty) g.initialLr = g.lr

  return opt
}

function countModelParams(model) {
  const groups = getParamGroups(model)
  let total = 0
  for (const list of Object.values(groups)) {
    for (const p of list) total += p.data.size
  }
  return total
}

function allParams(model) {
  const groups = getParamGroups(model)
  const all = []
  for (const list of Object.values(groups)) all.push(...list)
  return all
}

export {
  defaultConfig,
  createModel,
  initWeights,
  forward,
  getParamGroups,
  setupOptimizer,
  countModelParams,
  allParams,
}
