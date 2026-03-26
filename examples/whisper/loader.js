// examples/whisper/loader.js
// Load Whisper models from whisper.cpp GGML binary format (.bin files).
// Uses ggml_parser.js for the format-level parsing and adds smith-dependent
// weight loading + model construction.
//
// Reference: whisper.cpp/src/whisper.cpp whisper_model_load()

import smith from '../../src/index.js'
import {
  createWhisperModel, WHISPER_CONFIGS,
  createEncoderBlock, createDecoderBlock,
} from './model.js'
import {
  parseWhisperGGML, GGML_FILE_MAGIC, GGML_TYPE, GGML_TYPE_SIZE, GGML_BLOCK_SIZE,
} from './ggml_parser.js'

const { variable, tensor, zeros, ones } = smith

// --- Dequantize helpers ---

function dequantF16(data, offset, count) {
  const out = new Float32Array(count)
  const u16 = new Uint16Array(data.buffer, data.byteOffset + offset, count)
  for (let i = 0; i < count; i++) {
    out[i] = smith.fromFloat16(u16[i])
  }
  return out
}

function dequantQ8_0(data, offset, nElements) {
  const blockSize = 32
  const nBlocks = nElements / blockSize
  const out = new Float32Array(nElements)
  let pos = offset
  for (let b = 0; b < nBlocks; b++) {
    const view = new DataView(data.buffer, data.byteOffset + pos)
    const d = view.getFloat16(0, true)
    pos += 2
    for (let i = 0; i < blockSize; i++) {
      const q = new Int8Array(data.buffer, data.byteOffset + pos, 1)[0]
      out[b * blockSize + i] = q * d
      pos++
    }
  }
  return out
}

// --- Load tensor data as Float32Array ---

function loadTensorF32(parsed, tensorInfo) {
  const data = new Uint8Array(parsed.buffer)
  const { ttype, nElements, dataOffset } = tensorInfo

  if (ttype === GGML_TYPE.F32) {
    return new Float32Array(parsed.buffer, dataOffset, nElements)
  }
  if (ttype === GGML_TYPE.F16) {
    return dequantF16(data, dataOffset, nElements)
  }
  if (ttype === GGML_TYPE.Q8_0) {
    return dequantQ8_0(data, dataOffset, nElements)
  }

  throw new Error(`Unsupported tensor type: ${ttype} for ${tensorInfo.name}`)
}

// --- Weight name mapping ---
// whisper.cpp stores PyTorch-style names like:
//   encoder.blocks.0.attn.query.weight
//   decoder.blocks.1.cross_attn.key.weight

const ENCODER_MAP = {
  'encoder.conv1.weight': 'conv1W',
  'encoder.conv1.bias': 'conv1B',
  'encoder.conv2.weight': 'conv2W',
  'encoder.conv2.bias': 'conv2B',
  'encoder.positional_embedding': 'encoderPE',
  'encoder.ln_post.weight': 'encoderLnW',
  'encoder.ln_post.bias': 'encoderLnB',
}

// Now using Smith's createMultiHeadAttention (qProj/kProj/vProj/outProj with .weight/.bias)
// and createLinear (ffn1/ffn2 with .weight/.bias)
const ENCODER_BLOCK_MAP = {
  'attn_ln.weight': 'attnLnW',
  'attn_ln.bias': 'attnLnB',
  'attn.query.weight': 'mha.qProj.weight',
  'attn.query.bias': 'mha.qProj.bias',
  'attn.key.weight': 'mha.kProj.weight',
  // Note: Whisper's key projection has NO bias in the encoder
  'attn.value.weight': 'mha.vProj.weight',
  'attn.value.bias': 'mha.vProj.bias',
  'attn.out.weight': 'mha.outProj.weight',
  'attn.out.bias': 'mha.outProj.bias',
  'mlp_ln.weight': 'ffnLnW',
  'mlp_ln.bias': 'ffnLnB',
  'mlp.0.weight': 'ffn1.weight',
  'mlp.0.bias': 'ffn1.bias',
  'mlp.2.weight': 'ffn2.weight',
  'mlp.2.bias': 'ffn2.bias',
}

const DECODER_MAP = {
  'decoder.positional_embedding': 'decoderPE',
  'decoder.token_embedding.weight': 'tokenEmbed',
  'decoder.ln.weight': 'decoderLnW',
  'decoder.ln.bias': 'decoderLnB',
}

const DECODER_BLOCK_MAP = {
  'attn_ln.weight': 'selfAttnLnW',
  'attn_ln.bias': 'selfAttnLnB',
  'attn.query.weight': 'selfAttn.qProj.weight',
  'attn.query.bias': 'selfAttn.qProj.bias',
  'attn.key.weight': 'selfAttn.kProj.weight',
  'attn.value.weight': 'selfAttn.vProj.weight',
  'attn.value.bias': 'selfAttn.vProj.bias',
  'attn.out.weight': 'selfAttn.outProj.weight',
  'attn.out.bias': 'selfAttn.outProj.bias',
  'cross_attn_ln.weight': 'crossAttnLnW',
  'cross_attn_ln.bias': 'crossAttnLnB',
  'cross_attn.query.weight': 'crossAttn.qProj.weight',
  'cross_attn.query.bias': 'crossAttn.qProj.bias',
  'cross_attn.key.weight': 'crossAttn.kProj.weight',
  'cross_attn.value.weight': 'crossAttn.vProj.weight',
  'cross_attn.value.bias': 'crossAttn.vProj.bias',
  'cross_attn.out.weight': 'crossAttn.outProj.weight',
  'cross_attn.out.bias': 'crossAttn.outProj.bias',
  'mlp_ln.weight': 'ffnLnW',
  'mlp_ln.bias': 'ffnLnB',
  'mlp.0.weight': 'ffn1.weight',
  'mlp.0.bias': 'ffn1.bias',
  'mlp.2.weight': 'ffn2.weight',
  'mlp.2.bias': 'ffn2.bias',
}

// --- Set a nested property by dot path ---

function setNested(obj, path, value) {
  const parts = path.split('.')
  let target = obj
  for (let i = 0; i < parts.length - 1; i++) {
    target = target[parts[i]]
  }
  target[parts[parts.length - 1]] = value
}

// --- Load a Whisper model from GGML binary ---

async function loadWhisperGGML(path) {
  const file = Bun.file(path)
  const buffer = await file.arrayBuffer()
  const parsed = parseWhisperGGML(buffer)
  const { hparams, vocab, tensors } = parsed

  // Build config
  const config = {
    dim: hparams.nAudioState,
    encoderLayers: hparams.nAudioLayer,
    decoderLayers: hparams.nTextLayer,
    numHeads: hparams.nAudioHead,
    ffnDim: hparams.nAudioState * 4,
    nMels: hparams.nMels,
    vocabSize: hparams.nVocab,
    maxTextCtx: hparams.nTextCtx,
    nAudioCtx: hparams.nAudioCtx,
  }

  // Create model structure
  const model = createWhisperModel(config)

  // Build tensor index
  const tensorMap = {}
  for (const t of tensors) tensorMap[t.name] = t

  let loaded = 0

  // Load encoder global weights
  for (const [ggmlName, modelPath] of Object.entries(ENCODER_MAP)) {
    const t = tensorMap[ggmlName]
    if (!t) continue
    const data = loadTensorF32(parsed, t)
    const shape = t.dims.length === 1 ? [t.dims[0]] : [...t.dims].reverse() // GGML stores row-major reversed
    const v = variable(tensor(Array.from(data), shape), { requiresGrad: false })
    if (modelPath === 'encoderPE') {
      model.encoderPE = v
    } else {
      model[modelPath] = v
    }
    loaded++
  }

  // Load encoder blocks
  for (let i = 0; i < config.encoderLayers; i++) {
    for (const [suffix, modelPath] of Object.entries(ENCODER_BLOCK_MAP)) {
      const ggmlName = `encoder.blocks.${i}.${suffix}`
      const t = tensorMap[ggmlName]
      if (!t) continue
      const data = loadTensorF32(parsed, t)
      const shape = t.dims.length === 1 ? [t.dims[0]] : [...t.dims].reverse()
      const v = variable(tensor(Array.from(data), shape), { requiresGrad: false })
      setNested(model.encoderBlocks[i], modelPath, v)
      loaded++
    }
  }

  // Load decoder global weights
  for (const [ggmlName, modelPath] of Object.entries(DECODER_MAP)) {
    const t = tensorMap[ggmlName]
    if (!t) continue
    const data = loadTensorF32(parsed, t)
    const shape = t.dims.length === 1 ? [t.dims[0]] : [...t.dims].reverse()
    model[modelPath] = variable(tensor(Array.from(data), shape), { requiresGrad: false })
    loaded++
  }

  // Load decoder blocks
  for (let i = 0; i < config.decoderLayers; i++) {
    for (const [suffix, modelPath] of Object.entries(DECODER_BLOCK_MAP)) {
      const ggmlName = `decoder.blocks.${i}.${suffix}`
      const t = tensorMap[ggmlName]
      if (!t) continue
      const data = loadTensorF32(parsed, t)
      const shape = t.dims.length === 1 ? [t.dims[0]] : [...t.dims].reverse()
      const v = variable(tensor(Array.from(data), shape), { requiresGrad: false })
      setNested(model.decoderBlocks[i], modelPath, v)
      loaded++
    }
  }

  console.log(`Loaded ${loaded}/${tensors.length} tensors`)

  return {
    model,
    config,
    vocab,
    melFilters: parsed.melFilters,
    hparams,
  }
}

export { loadWhisperGGML, loadTensorF32, parseWhisperGGML, GGML_FILE_MAGIC, GGML_TYPE }
