// examples/whisper/ggml_parser.js
// Pure JS parser for whisper.cpp GGML binary format.
// No dependencies on smith — can be tested in any environment.

const GGML_FILE_MAGIC = 0x67676d6c // "ggml"

const GGML_TYPE = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3,
  Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14,
}

const GGML_TYPE_SIZE = {
  [GGML_TYPE.F32]: 4,
  [GGML_TYPE.F16]: 2,
  [GGML_TYPE.Q4_0]: 18,
  [GGML_TYPE.Q4_1]: 20,
  [GGML_TYPE.Q5_0]: 22,
  [GGML_TYPE.Q5_1]: 24,
  [GGML_TYPE.Q8_0]: 34,
}

const GGML_BLOCK_SIZE = {
  [GGML_TYPE.F32]: 1,
  [GGML_TYPE.F16]: 1,
  [GGML_TYPE.Q4_0]: 32,
  [GGML_TYPE.Q4_1]: 32,
  [GGML_TYPE.Q5_0]: 32,
  [GGML_TYPE.Q5_1]: 32,
  [GGML_TYPE.Q8_0]: 32,
}

function parseWhisperGGML(buffer) {
  const view = new DataView(buffer)
  let offset = 0

  const read32 = () => { const v = view.getUint32(offset, true); offset += 4; return v }
  const readI32 = () => { const v = view.getInt32(offset, true); offset += 4; return v }
  const readF32 = () => { const v = view.getFloat32(offset, true); offset += 4; return v }

  // Magic
  const magic = read32()
  if (magic !== GGML_FILE_MAGIC) {
    throw new Error(`Not a GGML file: magic 0x${magic.toString(16)} !== 0x${GGML_FILE_MAGIC.toString(16)}`)
  }

  // Hparams
  const hparams = {
    nVocab: readI32(),
    nAudioCtx: readI32(),
    nAudioState: readI32(),
    nAudioHead: readI32(),
    nAudioLayer: readI32(),
    nTextCtx: readI32(),
    nTextState: readI32(),
    nTextHead: readI32(),
    nTextLayer: readI32(),
    nMels: readI32(),
    ftype: readI32(),
  }

  // Mel filters
  const nMelFilters = readI32()
  const nFftFilters = readI32()
  const melFilters = new Float32Array(nMelFilters * nFftFilters)
  for (let i = 0; i < melFilters.length; i++) melFilters[i] = readF32()

  // Vocab
  const vocabSize = readI32()
  const vocab = []
  for (let i = 0; i < vocabSize; i++) {
    const len = read32()
    if (len > 0) {
      const bytes = new Uint8Array(buffer, offset, len)
      vocab.push(new TextDecoder().decode(bytes))
      offset += len
    } else {
      vocab.push('')
    }
  }

  // Tensors
  const tensors = []
  const data = new Uint8Array(buffer)

  while (offset < buffer.byteLength - 12) {
    const nDims = readI32()
    const nameLen = readI32()
    const ttype = readI32()

    if (nDims < 0 || nDims > 4 || nameLen < 0 || nameLen > 256) break

    const dims = []
    let nElements = 1
    for (let i = 0; i < nDims; i++) {
      dims.push(readI32())
      nElements *= dims[dims.length - 1]
    }

    const nameBytes = new Uint8Array(buffer, offset, nameLen)
    const name = new TextDecoder().decode(nameBytes)
    offset += nameLen

    // Align to 32 bytes
    offset = Math.ceil(offset / 32) * 32

    const blockSize = GGML_BLOCK_SIZE[ttype] || 1
    const typeSize = GGML_TYPE_SIZE[ttype] || (ttype === GGML_TYPE.F32 ? 4 : 2)
    const dataSize = ttype === GGML_TYPE.F32
      ? nElements * 4
      : ttype === GGML_TYPE.F16
        ? nElements * 2
        : Math.ceil(nElements / blockSize) * typeSize

    tensors.push({ name, dims, ttype, nElements, dataOffset: offset, dataSize })
    offset += dataSize
  }

  return { hparams, melFilters: { nMel: nMelFilters, nFft: nFftFilters, data: melFilters }, vocab, tensors, buffer }
}

// Load tensor as Float32Array (only f32 for now — dequant functions need smith for f16)
function loadTensorF32Raw(parsed, tensorInfo) {
  const { ttype, nElements, dataOffset } = tensorInfo
  if (ttype === GGML_TYPE.F32) {
    return new Float32Array(parsed.buffer, dataOffset, nElements)
  }
  throw new Error(`Raw F32 loading only supports F32 type, got ${ttype}`)
}

export { parseWhisperGGML, loadTensorF32Raw, GGML_FILE_MAGIC, GGML_TYPE, GGML_TYPE_SIZE, GGML_BLOCK_SIZE }
