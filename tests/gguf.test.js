// smith/tests/gguf.test.js
// Phase 9: GGUF parser and loader tests
import { test, expect } from 'bun:test'
import {
  parseGGUF, listTensors, extractConfig, dequantizeTensor,
  GGML_TYPE, GGUF_TYPE, dequantQ4_0, dequantQ8_0,
} from '../src/gguf.js'
import {
  precomputeRoPE, applyRoPE, rmsNorm,
  resolveWeight, resolvePath,
  LLAMA_MAP, GPT2_MAP,
} from '../src/gguf_loader.js'
import * as T from '../src/tensor.js'
import * as A from '../src/autograd.js'

function expectClose(a, b, tol = 1e-4) {
  expect(Math.abs(a - b)).toBeLessThan(tol)
}

// --- Build a minimal synthetic GGUF file ---

function buildGGUF(opts = {}) {
  const {
    version = 3,
    metadata = {},
    tensors = [],
  } = opts

  // Calculate sizes
  const parts = []

  function writeU32(v) { const b = new ArrayBuffer(4); new DataView(b).setUint32(0, v, true); parts.push(new Uint8Array(b)) }
  function writeU64(v) {
    const b = new ArrayBuffer(8)
    const dv = new DataView(b)
    dv.setUint32(0, v & 0xFFFFFFFF, true)
    dv.setUint32(4, (v / 0x100000000) | 0, true)
    parts.push(new Uint8Array(b))
  }
  function writeF32(v) { const b = new ArrayBuffer(4); new DataView(b).setFloat32(0, v, true); parts.push(new Uint8Array(b)) }
  function writeString(s) {
    const enc = new TextEncoder().encode(s)
    writeU64(enc.length)
    parts.push(enc)
  }
  function writeU8(v) { parts.push(new Uint8Array([v])) }

  // Magic + version
  writeU32(0x46475547) // "GGUF"
  writeU32(version)

  // Tensor count and metadata count
  const metaKeys = Object.entries(metadata)
  writeU64(tensors.length)
  writeU64(metaKeys.length)

  // Metadata KV pairs
  for (const [key, { type, value }] of metaKeys) {
    writeString(key)
    writeU32(type)
    if (type === GGUF_TYPE.STRING) writeString(value)
    else if (type === GGUF_TYPE.UINT32) writeU32(value)
    else if (type === GGUF_TYPE.FLOAT32) writeF32(value)
    else if (type === GGUF_TYPE.BOOL) writeU8(value ? 1 : 0)
    else if (type === GGUF_TYPE.UINT64) writeU64(value)
    else if (type === GGUF_TYPE.ARRAY) {
      writeU32(value.elemType)
      writeU64(value.items.length)
      for (const item of value.items) {
        if (value.elemType === GGUF_TYPE.STRING) writeString(item)
        else if (value.elemType === GGUF_TYPE.UINT32) writeU32(item)
        else if (value.elemType === GGUF_TYPE.FLOAT32) writeF32(item)
      }
    }
  }

  // Tensor info
  let dataOffset = 0
  for (const t of tensors) {
    writeString(t.name)
    writeU32(t.shape.length)
    for (const d of t.shape) writeU64(d)
    writeU32(t.type)
    writeU64(dataOffset)
    dataOffset += t.data.byteLength
  }

  // Concatenate header
  let headerSize = 0
  for (const p of parts) headerSize += p.byteLength

  // Align to 32 bytes
  const alignment = 32
  const alignedOffset = Math.ceil(headerSize / alignment) * alignment
  const padding = alignedOffset - headerSize

  // Build final buffer
  const total = alignedOffset + dataOffset
  const buf = new ArrayBuffer(total)
  const view = new Uint8Array(buf)

  let pos = 0
  for (const p of parts) { view.set(p, pos); pos += p.byteLength }
  pos += padding // skip padding

  // Write tensor data
  for (const t of tensors) {
    view.set(new Uint8Array(t.data), pos)
    pos += t.data.byteLength
  }

  return buf
}

// --- Parser tests ---

test('parse GGUF magic and version', () => {
  const buf = buildGGUF({ metadata: {}, tensors: [] })
  const parsed = parseGGUF(buf)
  expect(parsed.version).toBe(3)
  expect(parsed.tensors.length).toBe(0)
})

test('parse GGUF metadata strings', () => {
  const buf = buildGGUF({
    metadata: {
      'general.architecture': { type: GGUF_TYPE.STRING, value: 'llama' },
      'general.name': { type: GGUF_TYPE.STRING, value: 'test-model' },
    },
  })
  const parsed = parseGGUF(buf)
  expect(parsed.metadata['general.architecture']).toBe('llama')
  expect(parsed.metadata['general.name']).toBe('test-model')
})

test('parse GGUF metadata numeric types', () => {
  const buf = buildGGUF({
    metadata: {
      'llama.embedding_length': { type: GGUF_TYPE.UINT32, value: 4096 },
      'llama.rope.freq_base': { type: GGUF_TYPE.FLOAT32, value: 10000 },
      'llama.block_count': { type: GGUF_TYPE.UINT32, value: 32 },
    },
  })
  const parsed = parseGGUF(buf)
  expect(parsed.metadata['llama.embedding_length']).toBe(4096)
  expect(parsed.metadata['llama.block_count']).toBe(32)
  expectClose(parsed.metadata['llama.rope.freq_base'], 10000)
})

test('parse GGUF metadata arrays', () => {
  const buf = buildGGUF({
    metadata: {
      'test.values': {
        type: GGUF_TYPE.ARRAY,
        value: { elemType: GGUF_TYPE.UINT32, items: [1, 2, 3, 4] },
      },
    },
  })
  const parsed = parseGGUF(buf)
  expect(parsed.metadata['test.values']).toEqual([1, 2, 3, 4])
})

test('parse GGUF with f32 tensor', () => {
  const data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  const buf = buildGGUF({
    tensors: [{
      name: 'test.weight',
      shape: [2, 3],
      type: GGML_TYPE.F32,
      data: data.buffer,
    }],
  })
  const parsed = parseGGUF(buf)
  expect(parsed.tensors.length).toBe(1)
  expect(parsed.tensors[0].name).toBe('test.weight')
  expect(parsed.tensors[0].shape).toEqual([2, 3])
  expect(parsed.tensors[0].type).toBe(GGML_TYPE.F32)

  const result = dequantizeTensor(parsed, parsed.tensors[0])
  expect(result.length).toBe(6)
  expectClose(result[0], 1.0)
  expectClose(result[5], 6.0)
})

test('list tensors from parsed GGUF', () => {
  const data = new Float32Array(8)
  const buf = buildGGUF({
    tensors: [
      { name: 'a.weight', shape: [4, 2], type: GGML_TYPE.F32, data: data.buffer },
    ],
  })
  const parsed = parseGGUF(buf)
  const list = listTensors(parsed)
  expect(list.length).toBe(1)
  expect(list[0].name).toBe('a.weight')
  expect(list[0].type).toBe('F32')
  expect(list[0].bytes).toBe(32) // 8 floats * 4 bytes
})

test('reject non-GGUF file', () => {
  const buf = new ArrayBuffer(64)
  new Uint8Array(buf).fill(0)
  expect(() => parseGGUF(buf)).toThrow('Not a GGUF file')
})

// --- Dequantization tests ---

test('dequant Q4_0 block', () => {
  // Build a Q4_0 block: 2 bytes fp16 scale + 16 bytes nibbles
  const block = new Uint8Array(18)
  // Scale = 0.5 as fp16 (0x3800)
  block[0] = 0x00
  block[1] = 0x38
  // All nibbles = 8 → val = 0.5 * (8 - 8) = 0
  block.fill(0x88, 2, 18) // each byte = two nibbles of 8

  const values = dequantQ4_0(block, 0)
  for (let i = 0; i < 32; i++) {
    expectClose(values[i], 0.0, 0.01) // 0.5 * (8 - 8) = 0
  }

  // Nibble = 15 → val = 0.5 * (15 - 8) = 3.5
  block.fill(0xFF, 2, 18) // all nibbles = 15
  const values2 = dequantQ4_0(block, 0)
  expectClose(values2[0], 3.5, 0.01)
})

test('dequant Q8_0 block', () => {
  // Build a Q8_0 block: 2 bytes fp16 scale + 32 bytes int8
  const block = new Uint8Array(34)
  // Scale = 1.0 as fp16 (0x3C00)
  block[0] = 0x00
  block[1] = 0x3C
  // int8 values: alternating +1, -1
  for (let i = 0; i < 32; i++) {
    block[2 + i] = i % 2 === 0 ? 1 : 255 // 255 = -1 as uint8 → int8
  }

  const values = dequantQ8_0(block, 0)
  expectClose(values[0], 1.0, 0.01)
  expectClose(values[1], -1.0, 0.01)
  expectClose(values[2], 1.0, 0.01)
})

// --- Config extraction ---

test('extract Llama config from metadata', () => {
  const metadata = {
    'general.architecture': 'llama',
    'general.name': 'TinyLlama',
    'llama.embedding_length': 2048,
    'llama.block_count': 22,
    'llama.attention.head_count': 32,
    'llama.attention.head_count_kv': 4,
    'llama.context_length': 2048,
    'llama.feed_forward_length': 5632,
    'llama.rope.freq_base': 10000,
    'llama.attention.layer_norm_rms_epsilon': 1e-5,
  }

  const config = extractConfig(metadata)
  expect(config.arch).toBe('llama')
  expect(config.dim).toBe(2048)
  expect(config.numLayers).toBe(22)
  expect(config.numHeads).toBe(32)
  expect(config.numKVHeads).toBe(4)
  expect(config.maxSeqLen).toBe(2048)
  expect(config.ffnDim).toBe(5632)
  expect(config.ropeFreqBase).toBe(10000)
})

test('extract config defaults for missing fields', () => {
  const metadata = {
    'general.architecture': 'llama',
    'llama.embedding_length': 768,
    'llama.block_count': 12,
    'llama.attention.head_count': 12,
  }

  const config = extractConfig(metadata)
  expect(config.vocabSize).toBe(32000) // default
  expect(config.maxSeqLen).toBe(2048) // default
  expect(config.ropeFreqBase).toBe(10000) // default
})

// --- Weight mapping ---

test('resolve Llama weight names', () => {
  expect(resolveWeight('token_embd.weight', LLAMA_MAP)).toBe('embedding.tokenWeight')
  expect(resolveWeight('output_norm.weight', LLAMA_MAP)).toBe('lnFGamma')
  expect(resolveWeight('blk.0.attn_q.weight', LLAMA_MAP)).toBe('blocks.0.mha.qProj.weight')
  expect(resolveWeight('blk.15.ffn_down.weight', LLAMA_MAP)).toBe('blocks.15.ffnDown.weight')
  expect(resolveWeight('blk.3.attn_norm.weight', LLAMA_MAP)).toBe('blocks.3.ln1Gamma')
  expect(resolveWeight('unknown_tensor', LLAMA_MAP)).toBe(null)
})

test('resolve GPT-2 weight names', () => {
  expect(resolveWeight('token_embd.weight', GPT2_MAP)).toBe('embedding.tokenWeight')
  expect(resolveWeight('position_embd.weight', GPT2_MAP)).toBe('embedding.posWeight')
  expect(resolveWeight('blk.0.attn_qkv.weight', GPT2_MAP)).toBe('blocks.0.mha.qkvFused.weight')
})

// --- Path resolution ---

test('resolve dotted path into nested object', () => {
  const obj = {
    blocks: [
      { mha: { qProj: { weight: 'found' } } },
    ],
  }
  expect(resolvePath(obj, 'blocks.0.mha.qProj.weight')).toBe('found')
  expect(resolvePath(obj, 'blocks.1.mha')).toBe(null)
})

// --- RoPE ---

test('precompute RoPE tables', () => {
  const rope = precomputeRoPE(8, 4, 10000)
  // cos[0] should be all 1s (pos=0, all angles=0)
  expect(rope.cos.shape).toEqual([4, 4]) // [maxSeqLen, halfDim]
  expect(rope.sin.shape).toEqual([4, 4])
  // At pos=0, angle=0 for all dims → cos=1, sin=0
  for (let i = 0; i < 4; i++) {
    expectClose(rope.cos.data[i], 1.0, 1e-6)
    expectClose(rope.sin.data[i], 0.0, 1e-6)
  }
})

test('apply RoPE preserves shape', () => {
  const rope = precomputeRoPE(4, 8, 10000)
  const x = T.tensor([1, 0, 0, 0, 0, 1, 0, 0], [2, 4])
  const xVar = A.variable(x, { requiresGrad: false })
  const result = applyRoPE(xVar, rope, 0)
  expect(result.data.shape).toEqual([2, 4])
})

test('RoPE at position 0 is identity for [x, 0] pairs', () => {
  // At pos=0, cos=1, sin=0, so (x0, x1) → (x0*1 - x1*0, x0*0 + x1*1) = (x0, x1)
  const rope = precomputeRoPE(4, 4, 10000)
  const x = T.tensor([3.0, 7.0, 3.0, 7.0], [1, 4])
  const xVar = A.variable(x, { requiresGrad: false })
  const result = applyRoPE(xVar, rope, 0)
  const arr = T.toArray(result.data)
  expectClose(arr[0][0], 3.0)
  expectClose(arr[0][1], 7.0)
  expectClose(arr[0][2], 3.0)
  expectClose(arr[0][3], 7.0)
})

// --- RMSNorm ---

test('rmsNorm basic', () => {
  // RMSNorm([1, 1, 1, 1], gamma=1) = [1, 1, 1, 1] / sqrt(1 + eps) ≈ [1, 1, 1, 1]
  const x = T.tensor([1, 1, 1, 1], [1, 4])
  const gamma = T.tensor([1, 1, 1, 1], [4])
  const xVar = A.variable(x, { requiresGrad: false })
  const gVar = A.variable(gamma, { requiresGrad: false })
  const result = rmsNorm(xVar, gVar)
  const arr = T.toArray(result.data)
  // rms = sqrt(mean(1^2 * 4) / 4 + eps) = sqrt(1 + eps) ≈ 1
  for (const v of arr[0]) expectClose(v, 1.0, 0.01)
})

test('rmsNorm with scaling', () => {
  const x = T.tensor([2, 2, 2, 2], [1, 4])
  const gamma = T.tensor([0.5, 0.5, 0.5, 0.5], [4])
  const xVar = A.variable(x, { requiresGrad: false })
  const gVar = A.variable(gamma, { requiresGrad: false })
  const result = rmsNorm(xVar, gVar)
  const arr = T.toArray(result.data)
  // rms = sqrt(mean(4*4)/4 + eps) = sqrt(4 + eps) = 2
  // result = 2 * 0.5 / 2 = 0.5
  for (const v of arr[0]) expectClose(v, 0.5, 0.01)
})

// --- Full GGUF round-trip with f32 tensor ---

test('parse and dequantize f32 tensor from synthetic GGUF', () => {
  const values = [1.5, -2.5, 3.0, 0.0, -1.0, 4.25]
  const data = new Float32Array(values)
  const buf = buildGGUF({
    metadata: {
      'general.architecture': { type: GGUF_TYPE.STRING, value: 'llama' },
    },
    tensors: [{
      name: 'test.weight',
      shape: [2, 3],
      type: GGML_TYPE.F32,
      data: data.buffer,
    }],
  })

  const parsed = parseGGUF(buf)
  expect(parsed.metadata['general.architecture']).toBe('llama')

  const result = dequantizeTensor(parsed, parsed.tensors[0])
  for (let i = 0; i < values.length; i++) {
    expectClose(result[i], values[i])
  }
})

// --- Multiple tensors in one GGUF ---

test('parse GGUF with multiple tensors', () => {
  const data1 = new Float32Array([1, 2, 3, 4])
  const data2 = new Float32Array([5, 6, 7, 8, 9, 10])
  const buf = buildGGUF({
    tensors: [
      { name: 'first', shape: [4], type: GGML_TYPE.F32, data: data1.buffer },
      { name: 'second', shape: [2, 3], type: GGML_TYPE.F32, data: data2.buffer },
    ],
  })

  const parsed = parseGGUF(buf)
  expect(parsed.tensors.length).toBe(2)

  const r1 = dequantizeTensor(parsed, parsed.tensors[0])
  expect(r1.length).toBe(4)
  expectClose(r1[0], 1.0)

  const r2 = dequantizeTensor(parsed, parsed.tensors[1])
  expect(r2.length).toBe(6)
  expectClose(r2[4], 9.0)
})

// --- GGUF version 2 ---

test('parse GGUF version 2', () => {
  const buf = buildGGUF({ version: 2, metadata: {}, tensors: [] })
  const parsed = parseGGUF(buf)
  expect(parsed.version).toBe(2)
})
