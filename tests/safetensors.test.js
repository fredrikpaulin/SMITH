// smith/tests/safetensors.test.js
// Tests for Phase 6: Safetensors parsing, tensor reading, round-trip, GPT-2 weight mapping.
// Note: parseSafetensors/readTensor/listTensors are pure JS (ArrayBuffer/DataView),
// so they can be tested without the Metal device. Tests that require createModel
// (loadGPT2Safetensors, exportSafetensors) need GPU and run on macOS only.

import { test, expect } from 'bun:test'
import smith from '../src/index.js'

// --- Helper: build a minimal safetensors buffer from scratch ---

function buildSafetensorsBuffer(tensors) {
  // tensors: [{ name, shape, dtype, data: Float32Array|Uint8Array }]
  const header = {}
  let dataSize = 0
  const entries = []

  for (const t of tensors) {
    const byteLen = t.data.byteLength
    header[t.name] = {
      dtype: t.dtype || 'F32',
      shape: t.shape,
      data_offsets: [dataSize, dataSize + byteLen],
    }
    entries.push(t)
    dataSize += byteLen
  }

  const headerStr = JSON.stringify(header)
  const headerBytes = new TextEncoder().encode(headerStr)
  const headerLen = headerBytes.length

  const totalSize = 8 + headerLen + dataSize
  const buf = new ArrayBuffer(totalSize)
  const view = new DataView(buf)

  // Write header length as LE u64
  view.setBigUint64(0, BigInt(headerLen), true)

  // Write header JSON
  new Uint8Array(buf, 8, headerLen).set(headerBytes)

  // Write tensor data
  let offset = 8 + headerLen
  for (const t of entries) {
    new Uint8Array(buf, offset, t.data.byteLength).set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength))
    offset += t.data.byteLength
  }

  return buf
}

// --- parseSafetensors ---

test('parseSafetensors reads header and tensor metadata', () => {
  const data = new Float32Array([1, 2, 3, 4, 5, 6])
  const buf = buildSafetensorsBuffer([
    { name: 'weight', shape: [2, 3], dtype: 'F32', data }
  ])

  const parsed = smith.parseSafetensors(buf)
  expect(parsed.tensors.weight).toBeDefined()
  expect(parsed.tensors.weight.shape).toEqual([2, 3])
  expect(parsed.tensors.weight.dtype).toBe('F32')
  expect(parsed.tensors.weight.byteSize).toBe(24) // 6 * 4
})

test('parseSafetensors handles multiple tensors', () => {
  const w1 = new Float32Array([1, 2, 3])
  const w2 = new Float32Array([4, 5, 6, 7])
  const buf = buildSafetensorsBuffer([
    { name: 'layer.weight', shape: [3], dtype: 'F32', data: w1 },
    { name: 'layer.bias', shape: [4], dtype: 'F32', data: w2 },
  ])

  const parsed = smith.parseSafetensors(buf)
  expect(Object.keys(parsed.tensors).length).toBe(2)
  expect(parsed.tensors['layer.weight'].shape).toEqual([3])
  expect(parsed.tensors['layer.bias'].shape).toEqual([4])
})

test('parseSafetensors skips __metadata__', () => {
  // Build buffer with __metadata__ in header
  const data = new Float32Array([1, 2])
  const header = {
    __metadata__: { format: 'pt' },
    'tensor.a': { dtype: 'F32', shape: [2], data_offsets: [0, 8] },
  }
  const headerStr = JSON.stringify(header)
  const headerBytes = new TextEncoder().encode(headerStr)
  const headerLen = headerBytes.length

  const buf = new ArrayBuffer(8 + headerLen + 8)
  const view = new DataView(buf)
  view.setBigUint64(0, BigInt(headerLen), true)
  new Uint8Array(buf, 8, headerLen).set(headerBytes)
  new Float32Array(buf, 8 + headerLen, 2).set(data)

  const parsed = smith.parseSafetensors(buf)
  expect(parsed.tensors['__metadata__']).toBeUndefined()
  expect(parsed.tensors['tensor.a']).toBeDefined()
})

// --- readTensor ---

test('readTensor returns correct data', () => {
  const data = new Float32Array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
  const buf = buildSafetensorsBuffer([
    { name: 'w', shape: [2, 3], dtype: 'F32', data }
  ])

  const parsed = smith.parseSafetensors(buf)
  const t = smith.readTensor(parsed, 'w')

  expect(t.shape).toEqual([2, 3])
  expect(t.dtype).toBe('F32')
  expect(t.data.length).toBe(6)
  expect(t.data[0]).toBeCloseTo(1.5)
  expect(t.data[5]).toBeCloseTo(6.5)
})

test('readTensor throws for missing tensor', () => {
  const data = new Float32Array([1])
  const buf = buildSafetensorsBuffer([
    { name: 'exists', shape: [1], dtype: 'F32', data }
  ])

  const parsed = smith.parseSafetensors(buf)
  expect(() => smith.readTensor(parsed, 'nope')).toThrow('Tensor not found')
})

test('readTensor reads multiple tensors from same buffer at correct offsets', () => {
  const w1 = new Float32Array([10, 20, 30])
  const w2 = new Float32Array([40, 50])
  const buf = buildSafetensorsBuffer([
    { name: 'a', shape: [3], dtype: 'F32', data: w1 },
    { name: 'b', shape: [2], dtype: 'F32', data: w2 },
  ])

  const parsed = smith.parseSafetensors(buf)
  const a = smith.readTensor(parsed, 'a')
  const b = smith.readTensor(parsed, 'b')

  expect(a.data[0]).toBe(10)
  expect(a.data[2]).toBe(30)
  expect(b.data[0]).toBe(40)
  expect(b.data[1]).toBe(50)
})

// --- listTensors ---

test('listTensors enumerates all tensors with metadata', () => {
  const buf = buildSafetensorsBuffer([
    { name: 'x', shape: [4, 4], dtype: 'F32', data: new Float32Array(16) },
    { name: 'y', shape: [2], dtype: 'F32', data: new Float32Array(2) },
  ])

  const parsed = smith.parseSafetensors(buf)
  const list = smith.listTensors(parsed)

  expect(list.length).toBe(2)
  const names = list.map(t => t.name).sort()
  expect(names).toEqual(['x', 'y'])

  const x = list.find(t => t.name === 'x')
  expect(x.shape).toEqual([4, 4])
  expect(x.dtype).toBe('F32')
})

// --- Round-trip: build → parse → read → verify ---

test('round-trip: data survives build → parse → read', () => {
  const original = new Float32Array(100)
  for (let i = 0; i < 100; i++) original[i] = Math.random() * 100 - 50

  const buf = buildSafetensorsBuffer([
    { name: 'test.weight', shape: [10, 10], dtype: 'F32', data: original }
  ])

  const parsed = smith.parseSafetensors(buf)
  const t = smith.readTensor(parsed, 'test.weight')

  expect(t.data.length).toBe(100)
  for (let i = 0; i < 100; i++) {
    expect(t.data[i]).toBeCloseTo(original[i], 5)
  }
})

// --- exportSafetensors + parseSafetensors round-trip (requires GPU) ---

test('exportSafetensors round-trips through parseSafetensors', () => {
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 16 })

  // Fill weights with known values so we can verify
  const params = smith.modelParams(model)
  for (const p of params) {
    for (let i = 0; i < p.data.size; i++) {
      p.data.data[i] = (i % 100) * 0.01
    }
  }

  const buf = smith.exportSafetensors(model)
  expect(buf).toBeInstanceOf(ArrayBuffer)
  expect(buf.byteLength).toBeGreaterThan(0)

  const parsed = smith.parseSafetensors(buf)

  // Should contain GPT-2 named tensors
  expect(parsed.tensors['transformer.wte.weight']).toBeDefined()
  expect(parsed.tensors['transformer.wpe.weight']).toBeDefined()
  expect(parsed.tensors['transformer.h.0.ln_1.weight']).toBeDefined()
  expect(parsed.tensors['transformer.h.0.attn.c_attn.weight']).toBeDefined()
  expect(parsed.tensors['transformer.h.0.attn.c_attn.bias']).toBeDefined()
  expect(parsed.tensors['transformer.h.0.attn.c_proj.weight']).toBeDefined()
  expect(parsed.tensors['transformer.h.0.mlp.c_fc.weight']).toBeDefined()
  expect(parsed.tensors['transformer.h.0.mlp.c_proj.weight']).toBeDefined()
  expect(parsed.tensors['transformer.ln_f.weight']).toBeDefined()

  // Verify shapes
  expect(parsed.tensors['transformer.wte.weight'].shape).toEqual([32, 16])
  expect(parsed.tensors['transformer.wpe.weight'].shape).toEqual([16, 16])
  expect(parsed.tensors['transformer.h.0.attn.c_attn.weight'].shape).toEqual([16, 48]) // [dim, 3*dim]
  expect(parsed.tensors['transformer.h.0.attn.c_attn.bias'].shape).toEqual([48])
})

test('exportSafetensors fuses Q/K/V into c_attn correctly', () => {
  const dim = 16
  const model = smith.createModel({ vocabSize: 32, numLayers: 1, numHeads: 2, dim, maxSeqLen: 16 })

  // Set Q, K, V weights to distinct patterns
  const block = model.blocks[0]
  for (let i = 0; i < dim * dim; i++) {
    block.mha.qProj.weight.data.data[i] = 1.0 // Q = all 1s
    block.mha.kProj.weight.data.data[i] = 2.0 // K = all 2s
    block.mha.vProj.weight.data.data[i] = 3.0 // V = all 3s
  }

  const buf = smith.exportSafetensors(model)
  const parsed = smith.parseSafetensors(buf)
  const cAttn = smith.readTensor(parsed, 'transformer.h.0.attn.c_attn.weight')

  // c_attn is [dim, 3*dim] row-major
  // Row 0: [Q_0_0..Q_0_(dim-1), K_0_0..K_0_(dim-1), V_0_0..V_0_(dim-1)]
  expect(cAttn.data[0]).toBe(1.0)        // Q portion
  expect(cAttn.data[dim]).toBe(2.0)      // K portion
  expect(cAttn.data[2 * dim]).toBe(3.0)  // V portion
})

test('full export → load round-trip preserves weights', () => {
  const config = { vocabSize: 32, numLayers: 1, numHeads: 2, dim: 16, maxSeqLen: 16 }
  const model1 = smith.createModel(config)

  // Fill with non-trivial values
  const params1 = smith.modelParams(model1)
  for (const p of params1) {
    for (let i = 0; i < p.data.size; i++) {
      p.data.data[i] = Math.sin(i * 0.1) * 0.5
    }
  }

  // Export to safetensors
  const buf = smith.exportSafetensors(model1)

  // Create a fresh model and load the weights
  const model2 = smith.createModel(config)
  const parsed = smith.parseSafetensors(buf)
  smith.mapGPT2Weights(parsed, model2)

  // Verify all parameters match
  const params2 = smith.modelParams(model2)
  expect(params1.length).toBe(params2.length)

  for (let p = 0; p < params1.length; p++) {
    const d1 = params1[p].data.data
    const d2 = params2[p].data.data
    expect(d1.length).toBe(d2.length)
    for (let i = 0; i < d1.length; i++) {
      expect(d2[i]).toBeCloseTo(d1[i], 4)
    }
  }
})

// --- GPT-2 weight mapping: c_attn split ---

test('mapGPT2Weights splits fused c_attn into Q/K/V', () => {
  const dim = 16
  const config = { vocabSize: 32, numLayers: 1, numHeads: 2, dim, maxSeqLen: 16 }
  const model = smith.createModel(config)

  // Build a safetensors buffer with known fused c_attn values
  // c_attn weight: [dim, 3*dim], row i = [Q_i, K_i, V_i]
  const cAttnW = new Float32Array(dim * 3 * dim)
  const cAttnB = new Float32Array(3 * dim)
  for (let row = 0; row < dim; row++) {
    for (let col = 0; col < dim; col++) {
      cAttnW[row * 3 * dim + col] = 1.0              // Q region
      cAttnW[row * 3 * dim + dim + col] = 2.0         // K region
      cAttnW[row * 3 * dim + 2 * dim + col] = 3.0     // V region
    }
  }
  for (let i = 0; i < dim; i++) {
    cAttnB[i] = 0.1           // Q bias
    cAttnB[dim + i] = 0.2     // K bias
    cAttnB[2 * dim + i] = 0.3 // V bias
  }

  // Build all required GPT-2 tensors
  const tensors = [
    { name: 'transformer.wte.weight', shape: [32, dim], dtype: 'F32', data: new Float32Array(32 * dim) },
    { name: 'transformer.wpe.weight', shape: [16, dim], dtype: 'F32', data: new Float32Array(16 * dim) },
    { name: 'transformer.h.0.ln_1.weight', shape: [dim], dtype: 'F32', data: new Float32Array(dim).fill(1) },
    { name: 'transformer.h.0.ln_1.bias', shape: [dim], dtype: 'F32', data: new Float32Array(dim) },
    { name: 'transformer.h.0.ln_2.weight', shape: [dim], dtype: 'F32', data: new Float32Array(dim).fill(1) },
    { name: 'transformer.h.0.ln_2.bias', shape: [dim], dtype: 'F32', data: new Float32Array(dim) },
    { name: 'transformer.h.0.attn.c_attn.weight', shape: [dim, 3 * dim], dtype: 'F32', data: cAttnW },
    { name: 'transformer.h.0.attn.c_attn.bias', shape: [3 * dim], dtype: 'F32', data: cAttnB },
    { name: 'transformer.h.0.attn.c_proj.weight', shape: [dim, dim], dtype: 'F32', data: new Float32Array(dim * dim) },
    { name: 'transformer.h.0.attn.c_proj.bias', shape: [dim], dtype: 'F32', data: new Float32Array(dim) },
    { name: 'transformer.h.0.mlp.c_fc.weight', shape: [dim, 4 * dim], dtype: 'F32', data: new Float32Array(dim * 4 * dim) },
    { name: 'transformer.h.0.mlp.c_fc.bias', shape: [4 * dim], dtype: 'F32', data: new Float32Array(4 * dim) },
    { name: 'transformer.h.0.mlp.c_proj.weight', shape: [4 * dim, dim], dtype: 'F32', data: new Float32Array(4 * dim * dim) },
    { name: 'transformer.h.0.mlp.c_proj.bias', shape: [dim], dtype: 'F32', data: new Float32Array(dim) },
    { name: 'transformer.ln_f.weight', shape: [dim], dtype: 'F32', data: new Float32Array(dim).fill(1) },
    { name: 'transformer.ln_f.bias', shape: [dim], dtype: 'F32', data: new Float32Array(dim) },
  ]

  const buf = buildSafetensorsBuffer(tensors)
  const parsed = smith.parseSafetensors(buf)
  smith.mapGPT2Weights(parsed, model)

  // Check Q weights are all 1.0
  const qW = model.blocks[0].mha.qProj.weight.data.data
  for (let i = 0; i < dim * dim; i++) expect(qW[i]).toBe(1.0)

  // Check K weights are all 2.0
  const kW = model.blocks[0].mha.kProj.weight.data.data
  for (let i = 0; i < dim * dim; i++) expect(kW[i]).toBe(2.0)

  // Check V weights are all 3.0
  const vW = model.blocks[0].mha.vProj.weight.data.data
  for (let i = 0; i < dim * dim; i++) expect(vW[i]).toBe(3.0)

  // Check biases
  const qB = model.blocks[0].mha.qProj.bias.data.data
  const kB = model.blocks[0].mha.kProj.bias.data.data
  const vB = model.blocks[0].mha.vProj.bias.data.data
  for (let i = 0; i < dim; i++) {
    expect(qB[i]).toBeCloseTo(0.1)
    expect(kB[i]).toBeCloseTo(0.2)
    expect(vB[i]).toBeCloseTo(0.3)
  }
})

// --- Format edge cases ---

test('parseSafetensors rejects oversized header', () => {
  const buf = new ArrayBuffer(16)
  const view = new DataView(buf)
  view.setBigUint64(0, BigInt(200_000_000), true) // > 100MB limit
  expect(() => smith.parseSafetensors(buf)).toThrow('header exceeds')
})

test('readTensor handles empty tensors gracefully', () => {
  const buf = buildSafetensorsBuffer([
    { name: 'empty', shape: [0], dtype: 'F32', data: new Float32Array(0) }
  ])

  const parsed = smith.parseSafetensors(buf)
  const t = smith.readTensor(parsed, 'empty')
  expect(t.shape).toEqual([0])
  expect(t.data.length).toBe(0)
})
