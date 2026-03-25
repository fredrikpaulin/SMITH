// smith/src/gguf.js
// GGUF file parser — binary format used by llama.cpp, ollama, etc.
// Reads header, metadata key-value pairs, tensor descriptors, and raw data.
// Supports GGUF v2 and v3.

// --- GGUF type IDs ---
const GGUF_TYPE = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3,
  UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7,
  STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11,
  FLOAT64: 12,
}

// --- GGML tensor dtype IDs ---
const GGML_TYPE = {
  F32: 0, F16: 1,
  Q4_0: 2, Q4_1: 3,
  Q5_0: 6, Q5_1: 7,
  Q8_0: 8, Q8_1: 9,
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14,
  IQ2_XXS: 16, IQ2_XS: 17, IQ3_XXS: 18, IQ1_S: 19,
  IQ4_NL: 20, IQ3_S: 21, IQ2_S: 22, IQ4_XS: 23,
  I8: 24, I16: 25, I32: 26, I64: 27,
  F64: 28, IQ1_M: 29, BF16: 30,
}

const GGML_TYPE_NAME = Object.fromEntries(Object.entries(GGML_TYPE).map(([k, v]) => [v, k]))

// Block sizes and bytes per block for quantized types
const GGML_TYPE_INFO = {
  [GGML_TYPE.F32]:  { blockSize: 1, bytesPerBlock: 4 },
  [GGML_TYPE.F16]:  { blockSize: 1, bytesPerBlock: 2 },
  [GGML_TYPE.Q4_0]: { blockSize: 32, bytesPerBlock: 18 },  // 16 nibbles + 2 bytes scale (fp16)
  [GGML_TYPE.Q4_1]: { blockSize: 32, bytesPerBlock: 20 },  // 16 nibbles + 2 scale + 2 min (fp16)
  [GGML_TYPE.Q8_0]: { blockSize: 32, bytesPerBlock: 34 },  // 32 int8 + 2 bytes scale (fp16)
  [GGML_TYPE.Q8_1]: { blockSize: 32, bytesPerBlock: 36 },  // 32 int8 + 2 scale + 2 min (fp16)
  [GGML_TYPE.Q5_0]: { blockSize: 32, bytesPerBlock: 22 },  // 2 fp16 scale + 4 high-bits + 16 nibbles
  [GGML_TYPE.Q4_K]: { blockSize: 256, bytesPerBlock: 144 }, // 2 fp16 d + 2 fp16 dmin + 12 scales + 128 qs
  [GGML_TYPE.Q6_K]: { blockSize: 256, bytesPerBlock: 210 }, // 128 ql + 64 qh + 16 scales + 2 fp16 d
  [GGML_TYPE.BF16]: { blockSize: 1, bytesPerBlock: 2 },
}

// --- Reader: wraps DataView with a cursor ---

function createReader(buffer) {
  const view = new DataView(buffer)
  const bytes = new Uint8Array(buffer)
  let pos = 0

  function u8()  { const v = view.getUint8(pos); pos += 1; return v }
  function u16() { const v = view.getUint16(pos, true); pos += 2; return v }
  function u32() { const v = view.getUint32(pos, true); pos += 4; return v }
  function i32() { const v = view.getInt32(pos, true); pos += 4; return v }
  function f32() { const v = view.getFloat32(pos, true); pos += 4; return v }
  function f64() { const v = view.getFloat64(pos, true); pos += 8; return v }

  // GGUF uses u64 for counts — JS can handle up to 2^53 safely
  function u64() {
    const lo = view.getUint32(pos, true)
    const hi = view.getUint32(pos + 4, true)
    pos += 8
    return lo + hi * 0x100000000
  }

  function i64() {
    const lo = view.getUint32(pos, true)
    const hi = view.getInt32(pos + 4, true)
    pos += 8
    return lo + hi * 0x100000000
  }

  function string() {
    const len = u64()
    const s = new TextDecoder().decode(bytes.subarray(pos, pos + len))
    pos += len
    return s
  }

  function bool() { return u8() !== 0 }

  function skip(n) { pos += n }
  function tell() { return pos }
  function seek(p) { pos = p }

  return { u8, u16, u32, i32, f32, f64, u64, i64, string, bool, skip, tell, seek, view, bytes, buffer }
}

// --- Read a typed value ---

function readValue(r, type) {
  switch (type) {
    case GGUF_TYPE.UINT8:   return r.u8()
    case GGUF_TYPE.INT8:    return r.view.getInt8(r.tell()); // need to advance
    case GGUF_TYPE.UINT16:  return r.u16()
    case GGUF_TYPE.INT16:   { const v = r.view.getInt16(r.tell(), true); r.skip(2); return v }
    case GGUF_TYPE.UINT32:  return r.u32()
    case GGUF_TYPE.INT32:   return r.i32()
    case GGUF_TYPE.FLOAT32: return r.f32()
    case GGUF_TYPE.BOOL:    return r.bool()
    case GGUF_TYPE.STRING:  return r.string()
    case GGUF_TYPE.UINT64:  return r.u64()
    case GGUF_TYPE.INT64:   return r.i64()
    case GGUF_TYPE.FLOAT64: return r.f64()
    case GGUF_TYPE.ARRAY: {
      const elemType = r.u32()
      const len = r.u64()
      const arr = new Array(len)
      for (let i = 0; i < len; i++) arr[i] = readValue(r, elemType)
      return arr
    }
    default: throw new Error(`Unknown GGUF value type: ${type}`)
  }
}

// Fix INT8 reader (needs manual advance)
function readValueFixed(r, type) {
  if (type === GGUF_TYPE.INT8) {
    const v = r.view.getInt8(r.tell())
    r.skip(1)
    return v
  }
  return readValue(r, type)
}

// --- Parse GGUF file ---

function parseGGUF(buffer) {
  const r = createReader(buffer)

  // Magic: "GGUF" = 0x46554747 (little-endian)
  const magic = r.u32()
  if (magic !== 0x46554747) throw new Error(`Not a GGUF file (magic: 0x${magic.toString(16)})`)

  const version = r.u32()
  if (version < 2 || version > 3) throw new Error(`Unsupported GGUF version: ${version}`)

  const tensorCount = r.u64()
  const metadataKVCount = r.u64()

  // Read metadata key-value pairs
  const metadata = {}
  for (let i = 0; i < metadataKVCount; i++) {
    const key = r.string()
    const valueType = r.u32()
    metadata[key] = readValueFixed(r, valueType)
  }

  // Read tensor info
  const tensors = []
  for (let i = 0; i < tensorCount; i++) {
    const name = r.string()
    const nDims = r.u32()
    const shape = new Array(nDims)
    for (let d = 0; d < nDims; d++) shape[d] = r.u64()
    const type = r.u32()
    const offset = r.u64()
    tensors.push({ name, shape, type, offset })
  }

  // Data section starts at next alignment boundary (default 32 bytes)
  const alignment = metadata['general.alignment'] ?? 32
  const dataOffset = Math.ceil(r.tell() / alignment) * alignment

  return { version, metadata, tensors, dataOffset, buffer }
}

// --- Extract tensor data ---

function tensorBytes(info) {
  const typeInfo = GGML_TYPE_INFO[info.type]
  if (!typeInfo) throw new Error(`Unsupported tensor type: ${GGML_TYPE_NAME[info.type] ?? info.type}`)
  const numElements = info.shape.reduce((a, b) => a * b, 1)
  const numBlocks = Math.ceil(numElements / typeInfo.blockSize)
  return numBlocks * typeInfo.bytesPerBlock
}

function readTensorData(parsed, tensorInfo) {
  const byteLen = tensorBytes(tensorInfo)
  const start = parsed.dataOffset + tensorInfo.offset
  return new Uint8Array(parsed.buffer, start, byteLen)
}

// --- Dequantize Q4_0 block to f32 ---
// Q4_0 block: 2 bytes fp16 scale + 16 bytes (32 nibbles)
// val = scale * (nibble - 8)

function dequantQ4_0(blockData, blockOffset) {
  const scaleU16 = blockData[blockOffset] | (blockData[blockOffset + 1] << 8)
  const scale = fromFloat16(scaleU16)
  const values = new Float32Array(32)
  for (let i = 0; i < 32; i++) {
    const byteIdx = blockOffset + 2 + (i >> 1)
    const nibble = (i & 1) ? (blockData[byteIdx] >> 4) : (blockData[byteIdx] & 0x0F)
    values[i] = scale * (nibble - 8)
  }
  return values
}

// --- Dequantize Q4_1 block to f32 ---
// Q4_1 block: 2 bytes fp16 scale + 2 bytes fp16 min + 16 bytes (32 nibbles)
// val = scale * nibble + min

function dequantQ4_1(blockData, blockOffset) {
  const scaleU16 = blockData[blockOffset] | (blockData[blockOffset + 1] << 8)
  const minU16 = blockData[blockOffset + 2] | (blockData[blockOffset + 3] << 8)
  const scale = fromFloat16(scaleU16)
  const min = fromFloat16(minU16)
  const values = new Float32Array(32)
  for (let i = 0; i < 32; i++) {
    const byteIdx = blockOffset + 4 + (i >> 1)
    const nibble = (i & 1) ? (blockData[byteIdx] >> 4) : (blockData[byteIdx] & 0x0F)
    values[i] = scale * nibble + min
  }
  return values
}

// --- Dequantize Q8_0 block to f32 ---
// Q8_0 block: 2 bytes fp16 scale + 32 bytes (32 int8)
// val = scale * int8_val

function dequantQ8_0(blockData, blockOffset) {
  const scaleU16 = blockData[blockOffset] | (blockData[blockOffset + 1] << 8)
  const scale = fromFloat16(scaleU16)
  const values = new Float32Array(32)
  for (let i = 0; i < 32; i++) {
    const v = blockData[blockOffset + 2 + i]
    // Interpret as signed int8
    values[i] = scale * ((v > 127) ? v - 256 : v)
  }
  return values
}

// --- Dequantize Q5_0 block to f32 ---
// Q5_0 block: 2 bytes fp16 scale + 4 bytes high-bits + 16 bytes (32 nibbles)
// Layout: [scale_fp16(2)] [high_bits(4)] [nibbles(16)]
// val = scale * ((nibble | (high_bit << 4)) - 16)

function dequantQ5_0(blockData, blockOffset) {
  const scaleU16 = blockData[blockOffset] | (blockData[blockOffset + 1] << 8)
  const scale = fromFloat16(scaleU16)
  const values = new Float32Array(32)

  // High bits: 4 bytes = 32 bits, one per element
  const hb0 = blockData[blockOffset + 2]
  const hb1 = blockData[blockOffset + 3]
  const hb2 = blockData[blockOffset + 4]
  const hb3 = blockData[blockOffset + 5]
  const highBits = hb0 | (hb1 << 8) | (hb2 << 16) | ((hb3 << 24) >>> 0)

  for (let i = 0; i < 32; i++) {
    const byteIdx = blockOffset + 6 + (i >> 1)
    const nibble = (i & 1) ? (blockData[byteIdx] >> 4) : (blockData[byteIdx] & 0x0F)
    const hBit = (highBits >> i) & 1
    values[i] = scale * ((nibble | (hBit << 4)) - 16)
  }
  return values
}

// --- Dequantize Q4_K block to f32 ---
// Q4_K super-block: 256 elements
// Layout: [d_fp16(2)] [dmin_fp16(2)] [scales(12)] [qs(128)]
// 8 sub-blocks of 32 elements each, with 6-bit scales packed in 12 bytes

function dequantQ4_K(blockData, blockOffset) {
  const dU16 = blockData[blockOffset] | (blockData[blockOffset + 1] << 8)
  const dminU16 = blockData[blockOffset + 2] | (blockData[blockOffset + 3] << 8)
  const d = fromFloat16(dU16)
  const dmin = fromFloat16(dminU16)

  const values = new Float32Array(256)
  const scalesOff = blockOffset + 4
  const qsOff = blockOffset + 16 // 4 + 12

  // Decode the packed 6-bit scales and mins from 12 bytes
  // Lower 4 bits of each scale byte hold scales for sub-blocks 0-7
  // Upper 4 bits hold mins for sub-blocks 0-7
  // Bytes 8-11 hold 2-bit high parts for scales and mins
  const sc = new Float32Array(8)
  const mn = new Float32Array(8)

  for (let i = 0; i < 8; i++) {
    let scVal, mnVal
    if (i < 4) {
      scVal = blockData[scalesOff + i] & 0x3F
      mnVal = blockData[scalesOff + 4 + i] & 0x3F
    } else {
      scVal = (blockData[scalesOff + i - 4] >> 6) | ((blockData[scalesOff + (i - 4) + 8] & 0x0F) << 2)
      mnVal = (blockData[scalesOff + i] >> 6) | ((blockData[scalesOff + (i - 4) + 8] >> 4) << 2)
    }
    sc[i] = d * scVal
    mn[i] = dmin * mnVal
  }

  for (let j = 0; j < 256; j++) {
    const subBlock = j >> 5 // which sub-block (0-7)
    const qByte = blockData[qsOff + (j >> 1)]
    const nibble = (j & 1) ? (qByte >> 4) : (qByte & 0x0F)
    values[j] = sc[subBlock] * nibble - mn[subBlock]
  }

  return values
}

// --- Dequantize Q6_K block to f32 ---
// Q6_K super-block: 256 elements
// Layout: [ql(128)] [qh(64)] [scales(16)] [d_fp16(2)]
// 6-bit quantization: low 4 bits in ql, high 2 bits in qh

function dequantQ6_K(blockData, blockOffset) {
  const qlOff = blockOffset
  const qhOff = blockOffset + 128
  const scOff = blockOffset + 192
  const dU16 = blockData[blockOffset + 208] | (blockData[blockOffset + 209] << 8)
  const d = fromFloat16(dU16)

  const values = new Float32Array(256)

  for (let j = 0; j < 256; j++) {
    // Low 4 bits from ql
    const qlByte = blockData[qlOff + (j >> 1)]
    const ql = (j & 1) ? (qlByte >> 4) : (qlByte & 0x0F)

    // High 2 bits from qh
    const qhIdx = j >> 2  // 4 elements per qh byte
    const qhShift = (j & 3) * 2
    const qh = (blockData[qhOff + qhIdx] >> qhShift) & 0x03

    const q = ql | (qh << 4) // 6-bit value (0-63)

    // Scale: 16 sub-blocks of 16 elements each, scales are int8
    const scIdx = j >> 4
    const sc = blockData[scOff + scIdx]
    const scSigned = sc > 127 ? sc - 256 : sc  // interpret as int8

    values[j] = d * scSigned * (q - 32)
  }

  return values
}

// --- Dequantize a full tensor to Float32Array ---

function dequantizeTensor(parsed, tensorInfo) {
  const data = readTensorData(parsed, tensorInfo)
  const numElements = tensorInfo.shape.reduce((a, b) => a * b, 1)

  if (tensorInfo.type === GGML_TYPE.F32) {
    return new Float32Array(data.buffer, data.byteOffset, numElements)
  }

  if (tensorInfo.type === GGML_TYPE.F16) {
    const u16 = new Uint16Array(data.buffer, data.byteOffset, numElements)
    const out = new Float32Array(numElements)
    for (let i = 0; i < numElements; i++) out[i] = fromFloat16(u16[i])
    return out
  }

  if (tensorInfo.type === GGML_TYPE.BF16) {
    const u16 = new Uint16Array(data.buffer, data.byteOffset, numElements)
    const out = new Float32Array(numElements)
    for (let i = 0; i < numElements; i++) {
      // bf16 → f32: shift left 16 bits
      const buf = new ArrayBuffer(4)
      new Uint16Array(buf)[1] = u16[i]
      out[i] = new Float32Array(buf)[0]
    }
    return out
  }

  const typeInfo = GGML_TYPE_INFO[tensorInfo.type]
  if (!typeInfo) throw new Error(`Cannot dequantize type: ${GGML_TYPE_NAME[tensorInfo.type] ?? tensorInfo.type}`)

  const numBlocks = Math.ceil(numElements / typeInfo.blockSize)
  const out = new Float32Array(numElements)

  let dequantFn
  if (tensorInfo.type === GGML_TYPE.Q4_0) dequantFn = dequantQ4_0
  else if (tensorInfo.type === GGML_TYPE.Q4_1) dequantFn = dequantQ4_1
  else if (tensorInfo.type === GGML_TYPE.Q5_0) dequantFn = dequantQ5_0
  else if (tensorInfo.type === GGML_TYPE.Q8_0) dequantFn = dequantQ8_0
  else if (tensorInfo.type === GGML_TYPE.Q4_K) dequantFn = dequantQ4_K
  else if (tensorInfo.type === GGML_TYPE.Q6_K) dequantFn = dequantQ6_K
  else throw new Error(`Dequantize not implemented for ${GGML_TYPE_NAME[tensorInfo.type]}`)

  for (let b = 0; b < numBlocks; b++) {
    const blockOffset = b * typeInfo.bytesPerBlock
    const values = dequantFn(data, blockOffset)
    const outOffset = b * typeInfo.blockSize
    const count = Math.min(typeInfo.blockSize, numElements - outOffset)
    out.set(values.subarray(0, count), outOffset)
  }

  return out
}

// --- f16 decode (duplicated from dtype.js to avoid circular imports) ---

function fromFloat16(h) {
  const sign = (h >> 15) & 1
  const exp = (h >> 10) & 0x1F
  const frac = h & 0x3FF
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024)
  }
  if (exp === 31) return frac ? NaN : (sign ? -Infinity : Infinity)
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024)
}

// --- Metadata helpers ---

function getArch(metadata) {
  return metadata['general.architecture'] ?? 'unknown'
}

function getMetaValue(metadata, arch, key) {
  // Try arch-prefixed key first, then general
  return metadata[`${arch}.${key}`] ?? metadata[`general.${key}`] ?? undefined
}

function extractConfig(metadata) {
  const arch = getArch(metadata)
  const get = (key) => getMetaValue(metadata, arch, key)

  return {
    arch,
    name: metadata['general.name'] ?? 'unknown',
    vocabSize: get('vocab_size') ?? 32000,
    dim: get('embedding_length') ?? get('hidden_size'),
    numLayers: get('block_count') ?? get('num_hidden_layers'),
    numHeads: get('attention.head_count') ?? get('num_attention_heads'),
    numKVHeads: get('attention.head_count_kv') ?? get('num_key_value_heads'),
    maxSeqLen: get('context_length') ?? 2048,
    ffnDim: get('feed_forward_length') ?? null,
    ropeFreqBase: get('rope.freq_base') ?? 10000,
    ropeScaling: get('rope.scaling.type') ?? null,
    normEps: get('attention.layer_norm_rms_epsilon') ?? get('attention.layer_norm_epsilon') ?? 1e-5,
  }
}

// --- List all tensors in a parsed GGUF ---

function listTensors(parsed) {
  return parsed.tensors.map(t => ({
    name: t.name,
    shape: t.shape,
    type: GGML_TYPE_NAME[t.type] ?? `unknown(${t.type})`,
    bytes: tensorBytes(t),
  }))
}

export {
  parseGGUF, listTensors, readTensorData, dequantizeTensor,
  extractConfig, getArch, getMetaValue,
  GGML_TYPE, GGML_TYPE_NAME, GGML_TYPE_INFO, GGUF_TYPE,
  dequantQ4_0, dequantQ4_1, dequantQ5_0, dequantQ8_0,
  dequantQ4_K, dequantQ6_K,
}
