// smith/src/safetensors.js
// Safetensors file parser and GPT-2 weight loader.
// Format: 8-byte LE u64 header length + JSON header + raw tensor data.
// No dependencies — parsed with DataView.

import * as T from './tensor.js'
import * as A from './autograd.js'
import { createModel, modelParams } from './model.js'

// --- Dtype mapping ---

const DTYPE_INFO = {
  F32: { bytes: 4, array: Float32Array },
  F16: { bytes: 2, array: Uint16Array },
  BF16: { bytes: 2, array: Uint16Array },
  I32: { bytes: 4, array: Int32Array },
  I64: { bytes: 8, array: BigInt64Array },
  U8: { bytes: 1, array: Uint8Array },
}

// --- Parse safetensors file ---

function parseSafetensors(buffer) {
  const view = new DataView(buffer)

  // First 8 bytes: little-endian u64 = JSON header length
  const headerLen = Number(view.getBigUint64(0, true))
  if (headerLen > 100_000_000) throw new Error('Safetensors header exceeds 100MB limit')

  // Decode JSON header
  const headerBytes = new Uint8Array(buffer, 8, headerLen)
  const headerStr = new TextDecoder().decode(headerBytes)
  const header = JSON.parse(headerStr)

  // Data section starts after the header
  const dataOffset = 8 + headerLen

  // Extract tensor metadata (skip __metadata__)
  const tensors = {}
  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue
    const info = DTYPE_INFO[meta.dtype]
    if (!info) throw new Error(`Unsupported dtype: ${meta.dtype} for tensor ${name}`)
    tensors[name] = {
      dtype: meta.dtype,
      shape: meta.shape,
      dataStart: dataOffset + meta.data_offsets[0],
      dataEnd: dataOffset + meta.data_offsets[1],
      byteSize: meta.data_offsets[1] - meta.data_offsets[0],
    }
  }

  return { tensors, buffer, header }
}

// Read a tensor's raw data as a typed array
// Handles unaligned byte offsets by copying into a fresh aligned buffer when needed
function readTensor(parsed, name) {
  const meta = parsed.tensors[name]
  if (!meta) throw new Error(`Tensor not found: ${name}`)
  const info = DTYPE_INFO[meta.dtype]
  const count = meta.byteSize / info.bytes
  let data
  if (meta.dataStart % info.bytes === 0) {
    // Aligned — view directly into the buffer
    data = new info.array(parsed.buffer, meta.dataStart, count)
  } else {
    // Unaligned — copy bytes into a fresh aligned buffer
    const raw = new Uint8Array(parsed.buffer, meta.dataStart, meta.byteSize)
    const aligned = new ArrayBuffer(meta.byteSize)
    new Uint8Array(aligned).set(raw)
    data = new info.array(aligned, 0, count)
  }
  return { data, shape: meta.shape, dtype: meta.dtype }
}

// List all tensor names and their shapes
function listTensors(parsed) {
  return Object.entries(parsed.tensors).map(([name, meta]) => ({
    name, shape: meta.shape, dtype: meta.dtype,
  }))
}

// --- GPT-2 weight mapping ---
// GPT-2 uses Conv1D [in, out], matching Smith's convention.
// c_attn is a fused QKV projection [dim, 3*dim] that we split.

function mapGPT2Weights(parsed, model) {
  const { dim, numLayers } = model.config
  const params = modelParams(model)
  const errors = []

  // Helper: copy f32 data from safetensors into a Smith parameter
  function loadParam(param, srcData) {
    if (param.data.size !== srcData.length) {
      errors.push(`Size mismatch: param has ${param.data.size}, source has ${srcData.length}`)
      return
    }
    // Write directly into GPU-backed typed array via unified memory
    if (srcData instanceof Float32Array) {
      param.data.data.set(srcData)
    } else {
      // Convert to f32 if needed (e.g., from f16)
      for (let i = 0; i < srcData.length; i++) {
        param.data.data[i] = srcData[i]
      }
    }
  }

  // Helper: read a safetensors tensor as Float32Array
  function getF32(name) {
    const t = readTensor(parsed, name)
    if (t.dtype === 'F32') return t.data
    if (t.dtype === 'F16') {
      // Convert f16 to f32
      const out = new Float32Array(t.data.length)
      for (let i = 0; i < t.data.length; i++) {
        out[i] = f16ToF32(t.data[i])
      }
      return out
    }
    throw new Error(`Cannot convert ${t.dtype} to f32 for tensor ${name}`)
  }

  // f16 → f32 conversion (IEEE 754 half precision)
  function f16ToF32(h) {
    const sign = (h >> 15) & 1
    const exp = (h >> 10) & 0x1f
    const mant = h & 0x3ff
    if (exp === 0) {
      if (mant === 0) return sign ? -0 : 0
      // Subnormal
      return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024)
    }
    if (exp === 31) return mant === 0 ? (sign ? -Infinity : Infinity) : NaN
    return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024)
  }

  // --- Token + position embeddings ---
  loadParam(model.embedding.tokenWeight, getF32('transformer.wte.weight'))
  loadParam(model.embedding.posWeight, getF32('transformer.wpe.weight'))

  // --- Transformer blocks ---
  for (let i = 0; i < numLayers; i++) {
    const block = model.blocks[i]
    const prefix = `transformer.h.${i}`

    // Layer norms
    loadParam(block.ln1Gamma, getF32(`${prefix}.ln_1.weight`))
    loadParam(block.ln1Beta, getF32(`${prefix}.ln_1.bias`))
    loadParam(block.ln2Gamma, getF32(`${prefix}.ln_2.weight`))
    loadParam(block.ln2Beta, getF32(`${prefix}.ln_2.bias`))

    // Fused c_attn [dim, 3*dim] → split into Q, K, V projections [dim, dim] each
    const cAttnW = getF32(`${prefix}.attn.c_attn.weight`) // [dim, 3*dim] flattened
    const cAttnB = getF32(`${prefix}.attn.c_attn.bias`)   // [3*dim]

    // Split weight: stored as [dim, 3*dim] row-major
    // Row i contains [q_i_0..q_i_(dim-1), k_i_0..k_i_(dim-1), v_i_0..v_i_(dim-1)]
    const qW = new Float32Array(dim * dim)
    const kW = new Float32Array(dim * dim)
    const vW = new Float32Array(dim * dim)
    for (let row = 0; row < dim; row++) {
      const rowStart = row * 3 * dim
      qW.set(cAttnW.subarray(rowStart, rowStart + dim), row * dim)
      kW.set(cAttnW.subarray(rowStart + dim, rowStart + 2 * dim), row * dim)
      vW.set(cAttnW.subarray(rowStart + 2 * dim, rowStart + 3 * dim), row * dim)
    }

    loadParam(block.mha.qProj.weight, qW)
    loadParam(block.mha.kProj.weight, kW)
    loadParam(block.mha.vProj.weight, vW)

    // Split bias: [3*dim] → 3x [dim]
    loadParam(block.mha.qProj.bias, cAttnB.subarray(0, dim))
    loadParam(block.mha.kProj.bias, cAttnB.subarray(dim, 2 * dim))
    loadParam(block.mha.vProj.bias, cAttnB.subarray(2 * dim, 3 * dim))

    // Output projection
    loadParam(block.mha.outProj.weight, getF32(`${prefix}.attn.c_proj.weight`))
    loadParam(block.mha.outProj.bias, getF32(`${prefix}.attn.c_proj.bias`))

    // FFN
    loadParam(block.ffn1.weight, getF32(`${prefix}.mlp.c_fc.weight`))
    loadParam(block.ffn1.bias, getF32(`${prefix}.mlp.c_fc.bias`))
    loadParam(block.ffn2.weight, getF32(`${prefix}.mlp.c_proj.weight`))
    loadParam(block.ffn2.bias, getF32(`${prefix}.mlp.c_proj.bias`))
  }

  // --- Final layer norm ---
  loadParam(model.lnFGamma, getF32('transformer.ln_f.weight'))
  loadParam(model.lnFBeta, getF32('transformer.ln_f.bias'))

  // lm_head.weight is tied with wte.weight in GPT-2 — skip it
  if (errors.length > 0) throw new Error(`Weight loading errors:\n${errors.join('\n')}`)
}

// --- High-level loader ---

async function loadSafetensors(path, config) {
  const buf = await Bun.file(path).arrayBuffer()
  const parsed = parseSafetensors(buf)
  return { parsed, tensors: parsed.tensors }
}

// Load a GPT-2 model from a safetensors file
async function loadGPT2Safetensors(path, configOverrides = {}) {
  const buf = await Bun.file(path).arrayBuffer()
  const parsed = parseSafetensors(buf)

  // Infer config from tensor shapes
  const wte = parsed.tensors['transformer.wte.weight']
  const wpe = parsed.tensors['transformer.wpe.weight']
  if (!wte || !wpe) throw new Error('Not a GPT-2 model: missing wte or wpe')

  const vocabSize = wte.shape[0]
  const dim = wte.shape[1]
  const maxSeqLen = wpe.shape[0]

  // Count layers
  let numLayers = 0
  while (parsed.tensors[`transformer.h.${numLayers}.ln_1.weight`]) numLayers++
  if (numLayers === 0) throw new Error('No transformer blocks found')

  // Infer numHeads from c_attn shape: [dim, 3*dim], headDim is typically 64
  const headDim = 64
  const numHeads = dim / headDim

  const config = {
    vocabSize, numLayers, numHeads, dim, maxSeqLen,
    ...configOverrides,
  }

  const model = createModel(config)
  mapGPT2Weights(parsed, model)

  return model
}

// --- Export to safetensors ---

function exportSafetensors(model) {
  const { dim, numLayers, vocabSize, maxSeqLen } = model.config

  // Collect all tensors with their GPT-2 names
  const tensorEntries = []

  function addTensor(name, param) {
    const data = T.contiguous(param.data)
    tensorEntries.push({ name, shape: data.shape, data: data.data })
  }

  function addRawTensor(name, shape, data) {
    tensorEntries.push({ name, shape, data })
  }

  // Embeddings
  addTensor('transformer.wte.weight', model.embedding.tokenWeight)
  addTensor('transformer.wpe.weight', model.embedding.posWeight)

  for (let i = 0; i < numLayers; i++) {
    const block = model.blocks[i]
    const prefix = `transformer.h.${i}`

    addTensor(`${prefix}.ln_1.weight`, block.ln1Gamma)
    addTensor(`${prefix}.ln_1.bias`, block.ln1Beta)

    // Fuse Q, K, V back into c_attn [dim, 3*dim]
    const qW = T.contiguous(block.mha.qProj.weight.data)
    const kW = T.contiguous(block.mha.kProj.weight.data)
    const vW = T.contiguous(block.mha.vProj.weight.data)
    const cAttnW = new Float32Array(dim * 3 * dim)
    for (let row = 0; row < dim; row++) {
      const dst = row * 3 * dim
      cAttnW.set(qW.data.subarray(row * dim, (row + 1) * dim), dst)
      cAttnW.set(kW.data.subarray(row * dim, (row + 1) * dim), dst + dim)
      cAttnW.set(vW.data.subarray(row * dim, (row + 1) * dim), dst + 2 * dim)
    }
    addRawTensor(`${prefix}.attn.c_attn.weight`, [dim, 3 * dim], cAttnW)

    // Fuse biases
    const qB = T.contiguous(block.mha.qProj.bias.data)
    const kB = T.contiguous(block.mha.kProj.bias.data)
    const vB = T.contiguous(block.mha.vProj.bias.data)
    const cAttnB = new Float32Array(3 * dim)
    cAttnB.set(qB.data, 0)
    cAttnB.set(kB.data, dim)
    cAttnB.set(vB.data, 2 * dim)
    addRawTensor(`${prefix}.attn.c_attn.bias`, [3 * dim], cAttnB)

    addTensor(`${prefix}.attn.c_proj.weight`, block.mha.outProj.weight)
    addTensor(`${prefix}.attn.c_proj.bias`, block.mha.outProj.bias)

    addTensor(`${prefix}.ln_2.weight`, block.ln2Gamma)
    addTensor(`${prefix}.ln_2.bias`, block.ln2Beta)

    addTensor(`${prefix}.mlp.c_fc.weight`, block.ffn1.weight)
    addTensor(`${prefix}.mlp.c_fc.bias`, block.ffn1.bias)
    addTensor(`${prefix}.mlp.c_proj.weight`, block.ffn2.weight)
    addTensor(`${prefix}.mlp.c_proj.bias`, block.ffn2.bias)
  }

  addTensor('transformer.ln_f.weight', model.lnFGamma)
  addTensor('transformer.ln_f.bias', model.lnFBeta)

  // Build binary
  // Calculate data offsets
  let dataSize = 0
  const headerObj = {}
  for (const entry of tensorEntries) {
    const byteLen = entry.data.length * 4 // f32 = 4 bytes
    headerObj[entry.name] = {
      dtype: 'F32',
      shape: entry.shape,
      data_offsets: [dataSize, dataSize + byteLen],
    }
    dataSize += byteLen
  }

  const headerStr = JSON.stringify(headerObj)
  const headerBytes = new TextEncoder().encode(headerStr)
  const headerLen = headerBytes.length

  // Pad header to 4-byte alignment so tensor data is aligned for Float32Array views
  const padding = (4 - (headerLen % 4)) % 4
  const paddedHeaderLen = headerLen + padding

  // Recalculate with padded header (padding bytes are spaces, per safetensors convention)
  const totalSize = 8 + paddedHeaderLen + dataSize
  const output = new ArrayBuffer(totalSize)
  const view = new DataView(output)

  // Write padded header length as LE u64
  view.setBigUint64(0, BigInt(paddedHeaderLen), true)

  // Write header JSON + padding spaces
  const headerWithPadding = new Uint8Array(paddedHeaderLen)
  headerWithPadding.set(headerBytes)
  for (let i = headerLen; i < paddedHeaderLen; i++) headerWithPadding[i] = 0x20 // space
  new Uint8Array(output, 8, paddedHeaderLen).set(headerWithPadding)

  // Write tensor data (offset is now 4-byte aligned)
  let offset = 8 + paddedHeaderLen
  for (const entry of tensorEntries) {
    new Float32Array(output, offset, entry.data.length).set(entry.data)
    offset += entry.data.length * 4
  }

  return output
}

async function saveSafetensors(model, path) {
  const buf = exportSafetensors(model)
  await Bun.write(path, buf)
}

export {
  parseSafetensors, readTensor, listTensors,
  loadSafetensors, loadGPT2Safetensors,
  exportSafetensors, saveSafetensors,
  mapGPT2Weights,
}
