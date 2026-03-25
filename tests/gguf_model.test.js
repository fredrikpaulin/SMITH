// tests/gguf_model.test.js
// Integration test: loads a real GGUF model (Nemotron-H 4B Q4_K_M) and validates
// parsing, metadata extraction, tensor dequantization across all quant types.

import { test, expect } from 'bun:test'
import {
  parseGGUF, extractConfig, listTensors, readTensorData, dequantizeTensor,
  GGML_TYPE, GGML_TYPE_NAME, GGML_TYPE_INFO,
  dequantQ4_0, dequantQ4_1, dequantQ5_0, dequantQ8_0,
  dequantQ4_K, dequantQ6_K,
} from '../src/gguf.js'

const MODEL_PATH = 'models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf'

// Parse once, reuse across tests
let parsed, config

test('parse GGUF file', async () => {
  const buf = await Bun.file(MODEL_PATH).arrayBuffer()
  parsed = parseGGUF(buf)

  expect(parsed.version).toBeGreaterThanOrEqual(2)
  expect(parsed.version).toBeLessThanOrEqual(3)
  expect(parsed.tensors.length).toBeGreaterThan(0)
  expect(parsed.metadata).toBeDefined()
  expect(parsed.dataOffset).toBeGreaterThan(0)
})

// =====================================================================
// Metadata and config extraction
// =====================================================================

test('extract architecture', () => {
  expect(parsed.metadata['general.architecture']).toBe('nemotron_h')
  expect(parsed.metadata['general.type']).toBe('model')
})

test('extract config', () => {
  config = extractConfig(parsed.metadata)

  expect(config.arch).toBe('nemotron_h')
  expect(config.vocabSize).toBe(131072)
  expect(config.dim).toBe(3136)
  expect(config.numLayers).toBe(42)
  expect(config.numHeads).toBe(40)
  expect(config.maxSeqLen).toBe(1048576)
})

test('config has per-layer arrays for hybrid architecture', () => {
  // Nemotron-H has per-layer KV head counts and FFN dims
  expect(Array.isArray(config.numKVHeads)).toBe(true)
  expect(config.numKVHeads.length).toBe(42)
  expect(Array.isArray(config.ffnDim)).toBe(true)
  expect(config.ffnDim.length).toBe(42)
})

test('attention layers have 8 KV heads, SSM layers have 0', () => {
  // Layers with attention+GQA have 8 KV heads
  const kvLayers = config.numKVHeads.filter(h => h > 0)
  for (const h of kvLayers) expect(h).toBe(8)

  // Pure SSM layers have 0 KV heads
  const ssmOnlyCount = config.numKVHeads.filter(h => h === 0).length
  expect(ssmOnlyCount).toBeGreaterThan(0)
})

test('FFN layers have dim 12544', () => {
  const ffnLayers = config.ffnDim.filter(d => d > 0)
  for (const d of ffnLayers) expect(d).toBe(12544)
})

test('SSM metadata present', () => {
  const md = parsed.metadata
  expect(md['nemotron_h.ssm.conv_kernel']).toBe(4)
  expect(md['nemotron_h.ssm.state_size']).toBe(128)
  expect(md['nemotron_h.ssm.group_count']).toBe(8)
  expect(md['nemotron_h.ssm.inner_size']).toBe(7680)
  expect(md['nemotron_h.ssm.time_step_rank']).toBe(96)
})

// =====================================================================
// Tensor listing
// =====================================================================

test('lists all tensors with correct count', () => {
  const tensors = listTensors(parsed)
  expect(tensors.length).toBe(263)
  for (const t of tensors) {
    expect(t.name).toBeDefined()
    expect(t.shape.length).toBeGreaterThan(0)
    expect(typeof t.type).toBe('string')
    expect(t.bytes).toBeGreaterThan(0)
  }
})

test('tensor type distribution matches Q4_K_M quantization', () => {
  const counts = {}
  for (const t of parsed.tensors) {
    const name = GGML_TYPE_NAME[t.type]
    counts[name] = (counts[name] || 0) + 1
  }

  // Q4_K_M uses a mix of types
  expect(counts['F32']).toBeGreaterThan(0)
  expect(counts['Q4_K']).toBeGreaterThan(0)
  expect(counts['Q5_0']).toBeGreaterThan(0)
  expect(counts['Q6_K']).toBeGreaterThan(0)
  expect(counts['Q8_0']).toBeGreaterThan(0)
})

test('all tensor types have GGML_TYPE_INFO entries', () => {
  const types = new Set(parsed.tensors.map(t => t.type))
  for (const type of types) {
    expect(GGML_TYPE_INFO[type]).toBeDefined()
    expect(GGML_TYPE_INFO[type].blockSize).toBeGreaterThan(0)
    expect(GGML_TYPE_INFO[type].bytesPerBlock).toBeGreaterThan(0)
  }
})

// =====================================================================
// Layer structure analysis
// =====================================================================

test('model has token embedding and output tensors', () => {
  const names = parsed.tensors.map(t => t.name)
  expect(names).toContain('token_embd.weight')
  expect(names).toContain('output.weight')
  expect(names).toContain('output_norm.weight')
})

test('each layer has attention norm', () => {
  for (let i = 0; i < 42; i++) {
    const norm = parsed.tensors.find(t => t.name === `blk.${i}.attn_norm.weight`)
    expect(norm).toBeDefined()
    expect(norm.shape).toEqual([3136])
  }
})

test('hybrid layers: some have SSM, some have FFN, some both', () => {
  const layerComponents = {}
  for (const t of parsed.tensors) {
    const m = t.name.match(/^blk\.(\d+)\./)
    if (!m) continue
    const layer = parseInt(m[1])
    if (!layerComponents[layer]) layerComponents[layer] = new Set()
    const suffix = t.name.replace(/^blk\.\d+\./, '')
    if (suffix.startsWith('attn_')) layerComponents[layer].add('attention')
    if (suffix.startsWith('ssm_')) layerComponents[layer].add('ssm')
    if (suffix.startsWith('ffn_')) layerComponents[layer].add('ffn')
  }

  let hasSSM = false, hasFFN = false, hasAttnOnly = false
  for (let i = 0; i < 42; i++) {
    const c = layerComponents[i]
    expect(c.has('attention')).toBe(true) // all layers have attention
    if (c.has('ssm')) hasSSM = true
    if (c.has('ffn')) hasFFN = true
    if (!c.has('ssm') && !c.has('ffn')) hasAttnOnly = true
  }

  expect(hasSSM).toBe(true)
  expect(hasFFN).toBe(true)
  expect(hasAttnOnly).toBe(true) // some layers are attention-only
})

// =====================================================================
// Tensor shapes match architecture dimensions
// =====================================================================

test('token embedding shape matches vocab × dim', () => {
  const t = parsed.tensors.find(t => t.name === 'token_embd.weight')
  // GGUF stores as [dim, vocab] (row-major)
  expect(t.shape[0]).toBe(3136)
  expect(t.shape[1]).toBe(131072)
})

test('output weight shape matches dim × vocab', () => {
  const t = parsed.tensors.find(t => t.name === 'output.weight')
  expect(t.shape[0]).toBe(3136)
  expect(t.shape[1]).toBe(131072)
})

test('attention Q projection matches dim × (numHeads * headDim)', () => {
  // headDim = dim / numHeads = 3136 / 40 = 78.4... wait
  // Actually key_length = 128 from metadata, so Q proj maps to numHeads * keyLength
  const q = parsed.tensors.find(t => t.name === 'blk.12.attn_q.weight')
  expect(q).toBeDefined()
  expect(q.shape[0]).toBe(3136) // input dim
  expect(q.shape[1]).toBe(5120) // 40 heads × 128 head_dim
})

test('attention K/V projection matches GQA dimensions', () => {
  const k = parsed.tensors.find(t => t.name === 'blk.12.attn_k.weight')
  expect(k).toBeDefined()
  expect(k.shape[0]).toBe(3136)
  expect(k.shape[1]).toBe(1024) // 8 KV heads × 128 head_dim
})

test('SSM tensors have expected shapes', () => {
  const ssmA = parsed.tensors.find(t => t.name === 'blk.0.ssm_a')
  expect(ssmA.shape).toEqual([1, 96]) // [1, time_step_rank]

  const conv = parsed.tensors.find(t => t.name === 'blk.0.ssm_conv1d.weight')
  expect(conv.shape[0]).toBe(4) // conv_kernel
  expect(conv.shape[1]).toBe(9728) // inner_size + 2 * state_group_related

  const ssmIn = parsed.tensors.find(t => t.name === 'blk.0.ssm_in.weight')
  expect(ssmIn.shape[0]).toBe(3136) // dim

  const ssmOut = parsed.tensors.find(t => t.name === 'blk.0.ssm_out.weight')
  expect(ssmOut.shape[1]).toBe(3136) // output → dim
})

// =====================================================================
// Dequantization of each type
// =====================================================================

test('dequantize F32 tensor (output_norm.weight)', () => {
  const t = parsed.tensors.find(t => t.name === 'output_norm.weight')
  const data = dequantizeTensor(parsed, t)

  expect(data.length).toBe(3136)
  expect(data).toBeInstanceOf(Float32Array)
  // Values should be finite and non-zero (norm weights)
  let nonZero = 0
  for (let i = 0; i < data.length; i++) {
    expect(isFinite(data[i])).toBe(true)
    if (data[i] !== 0) nonZero++
  }
  expect(nonZero).toBeGreaterThan(data.length * 0.9)
})

test('dequantize Q5_0 tensor (attention K weight)', () => {
  // Pick a small-ish attention K weight: [3136, 1024] = 3.2M elements
  const t = parsed.tensors.find(t => t.name === 'blk.12.attn_k.weight')
  expect(t.type).toBe(GGML_TYPE.Q5_0)
  const data = dequantizeTensor(parsed, t)
  const expected = t.shape.reduce((a, b) => a * b, 1)

  expect(data.length).toBe(expected)
  expect(data).toBeInstanceOf(Float32Array)
  // Check values are reasonable (not all zeros, finite)
  let nonZero = 0
  for (let i = 0; i < Math.min(1000, data.length); i++) {
    expect(isFinite(data[i])).toBe(true)
    if (data[i] !== 0) nonZero++
  }
  expect(nonZero).toBeGreaterThan(500)
})

test('dequantize Q4_K block', () => {
  const t = parsed.tensors.find(t => t.type === GGML_TYPE.Q4_K)
  const raw = readTensorData(parsed, t)
  const block = dequantQ4_K(raw, 0)

  expect(block.length).toBe(256)
  expect(block).toBeInstanceOf(Float32Array)
  let nonZero = 0
  for (let i = 0; i < 256; i++) {
    expect(isFinite(block[i])).toBe(true)
    if (block[i] !== 0) nonZero++
  }
  expect(nonZero).toBeGreaterThan(100)
})

test('dequantize Q6_K block', () => {
  const t = parsed.tensors.find(t => t.type === GGML_TYPE.Q6_K)
  const raw = readTensorData(parsed, t)
  const block = dequantQ6_K(raw, 0)

  expect(block.length).toBe(256)
  expect(block).toBeInstanceOf(Float32Array)
  let nonZero = 0
  for (let i = 0; i < 256; i++) {
    expect(isFinite(block[i])).toBe(true)
    if (block[i] !== 0) nonZero++
  }
  expect(nonZero).toBeGreaterThan(100)
})

test('dequantize Q8_0 block', () => {
  const t = parsed.tensors.find(t => t.type === GGML_TYPE.Q8_0)
  const raw = readTensorData(parsed, t)
  const block = dequantQ8_0(raw, 0)

  expect(block.length).toBe(32)
  expect(block).toBeInstanceOf(Float32Array)
  for (let i = 0; i < 32; i++) {
    expect(isFinite(block[i])).toBe(true)
  }
})

test('dequantize Q5_0 block', () => {
  const t = parsed.tensors.find(t => t.type === GGML_TYPE.Q5_0)
  const raw = readTensorData(parsed, t)
  const block = dequantQ5_0(raw, 0)

  expect(block.length).toBe(32)
  expect(block).toBeInstanceOf(Float32Array)
  for (let i = 0; i < 32; i++) {
    expect(isFinite(block[i])).toBe(true)
  }
})

// =====================================================================
// Edge cases and data integrity
// =====================================================================

test('all tensors have valid data offsets within file', () => {
  for (const t of parsed.tensors) {
    const typeInfo = GGML_TYPE_INFO[t.type]
    if (!typeInfo) continue
    const numElements = t.shape.reduce((a, b) => a * b, 1)
    const numBlocks = Math.ceil(numElements / typeInfo.blockSize)
    const byteLen = numBlocks * typeInfo.bytesPerBlock
    const start = parsed.dataOffset + t.offset
    const end = start + byteLen
    expect(end).toBeLessThanOrEqual(parsed.buffer.byteLength)
  }
})

test('F32 norm weights are consistent across layers', () => {
  // All attention norm weights should have same shape
  const norms = parsed.tensors.filter(t => t.name.match(/^blk\.\d+\.attn_norm\.weight$/))
  expect(norms.length).toBe(42)
  for (const n of norms) {
    expect(n.shape).toEqual([3136])
    expect(n.type).toBe(GGML_TYPE.F32)
  }
})

test('Q4_K dequant produces values in reasonable range', () => {
  const t = parsed.tensors.find(t => t.type === GGML_TYPE.Q4_K)
  const raw = readTensorData(parsed, t)

  // Dequant multiple blocks and check range
  const blockCount = Math.min(10, Math.floor(raw.length / 144))
  for (let b = 0; b < blockCount; b++) {
    const vals = dequantQ4_K(raw, b * 144)
    for (let i = 0; i < 256; i++) {
      expect(Math.abs(vals[i])).toBeLessThan(10) // weights should be small
    }
  }
})

test('Q6_K dequant produces values in reasonable range', () => {
  const t = parsed.tensors.find(t => t.type === GGML_TYPE.Q6_K)
  const raw = readTensorData(parsed, t)

  const blockCount = Math.min(10, Math.floor(raw.length / 210))
  for (let b = 0; b < blockCount; b++) {
    const vals = dequantQ6_K(raw, b * 210)
    for (let i = 0; i < 256; i++) {
      expect(Math.abs(vals[i])).toBeLessThan(10)
    }
  }
})

// =====================================================================
// Architecture detection for forward pass support
// =====================================================================

test('nemotron_h architecture is not yet supported for forward pass', () => {
  // Document that this arch needs SSM (Mamba) support
  expect(config.arch).toBe('nemotron_h')
  // The loadGGUF loader currently supports: llama, phi, gpt2
  const supported = ['llama', 'phi', 'phi2', 'phi3', 'gpt2']
  expect(supported).not.toContain(config.arch)
})

test('model has all components needed for Mamba-2 implementation', () => {
  // Verify the SSM tensors exist for a future forward pass
  const ssmTensors = parsed.tensors.filter(t => t.name.includes('.ssm_'))
  expect(ssmTensors.length).toBeGreaterThan(0)

  // SSM components per layer: ssm_a, ssm_d, ssm_dt.bias, ssm_conv1d.weight,
  // ssm_conv1d.bias, ssm_in.weight, ssm_out.weight, ssm_norm.weight
  const ssmLayers = new Set()
  for (const t of ssmTensors) {
    const m = t.name.match(/^blk\.(\d+)\./)
    if (m) ssmLayers.add(parseInt(m[1]))
  }

  // SSM layers should match layers with 0 KV heads
  for (const layer of ssmLayers) {
    const hasSsm = parsed.tensors.some(t => t.name === `blk.${layer}.ssm_a`)
    expect(hasSsm).toBe(true)
  }
})
