import { test, expect } from 'bun:test'
import { parseWhisperGGML, GGML_FILE_MAGIC, GGML_TYPE } from '../ggml_parser.js'

// Build a synthetic minimal GGML binary for testing
function buildTestGGML() {
  // Calculate sizes
  const nMels = 2
  const nFft = 4
  const vocabSize = 3
  const vocabWords = ['hello', 'world', '']

  // Hparams
  const hparamsBytes = 11 * 4 // 11 int32s
  // Mel filters: 2 int32s + nMels*nFft floats
  const melBytes = 8 + nMels * nFft * 4
  // Vocab: 1 int32 + per word: uint32 len + bytes
  let vocabBytes = 4
  for (const w of vocabWords) vocabBytes += 4 + w.length

  // One tensor: "test.weight" shape [2, 3], type f32
  const tensorName = 'test.weight'
  const nElements = 6
  const tensorHeaderBytes = 4 + 4 + 4 + 2 * 4 + tensorName.length // n_dims + name_len + ttype + dims + name
  const alignedOffset = Math.ceil(4 + hparamsBytes + melBytes + vocabBytes + tensorHeaderBytes) // rough
  // We'll compute exact offset during writing

  const totalSize = 4096 // generous buffer
  const buffer = new ArrayBuffer(totalSize)
  const view = new DataView(buffer)
  const u8 = new Uint8Array(buffer)
  let offset = 0

  const write32 = (v) => { view.setUint32(offset, v, true); offset += 4 }
  const writeI32 = (v) => { view.setInt32(offset, v, true); offset += 4 }
  const writeF32 = (v) => { view.setFloat32(offset, v, true); offset += 4 }

  // Magic
  write32(GGML_FILE_MAGIC)

  // Hparams: n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
  //          n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype
  writeI32(vocabSize) // n_vocab
  writeI32(1500)      // n_audio_ctx
  writeI32(64)        // n_audio_state
  writeI32(4)         // n_audio_head
  writeI32(2)         // n_audio_layer
  writeI32(448)       // n_text_ctx
  writeI32(64)        // n_text_state
  writeI32(4)         // n_text_head
  writeI32(2)         // n_text_layer
  writeI32(nMels)     // n_mels
  writeI32(1)         // ftype (f16)

  // Mel filters
  writeI32(nMels)
  writeI32(nFft)
  for (let i = 0; i < nMels * nFft; i++) writeF32(i * 0.1)

  // Vocab
  writeI32(vocabSize)
  for (const w of vocabWords) {
    write32(w.length)
    for (let i = 0; i < w.length; i++) u8[offset++] = w.charCodeAt(i)
  }

  // Tensor: test.weight [2, 3] f32
  writeI32(2) // n_dims
  writeI32(tensorName.length) // name length
  writeI32(GGML_TYPE.F32) // type
  writeI32(3) // dim 0 (cols in GGML = first dim)
  writeI32(2) // dim 1 (rows in GGML = second dim)
  for (let i = 0; i < tensorName.length; i++) u8[offset++] = tensorName.charCodeAt(i)

  // Align to 32 bytes
  offset = Math.ceil(offset / 32) * 32

  // Data: 6 floats
  for (let i = 0; i < 6; i++) writeF32(i + 1)

  return buffer.slice(0, offset) // trim to exact size
}

test('parseWhisperGGML reads hparams', () => {
  const buffer = buildTestGGML()
  const parsed = parseWhisperGGML(buffer)

  expect(parsed.hparams.nVocab).toBe(3)
  expect(parsed.hparams.nAudioCtx).toBe(1500)
  expect(parsed.hparams.nAudioState).toBe(64)
  expect(parsed.hparams.nAudioHead).toBe(4)
  expect(parsed.hparams.nAudioLayer).toBe(2)
  expect(parsed.hparams.nTextCtx).toBe(448)
  expect(parsed.hparams.nTextState).toBe(64)
  expect(parsed.hparams.nTextHead).toBe(4)
  expect(parsed.hparams.nTextLayer).toBe(2)
  expect(parsed.hparams.nMels).toBe(2)
  expect(parsed.hparams.ftype).toBe(1)
})

test('parseWhisperGGML reads mel filters', () => {
  const buffer = buildTestGGML()
  const parsed = parseWhisperGGML(buffer)

  expect(parsed.melFilters.nMel).toBe(2)
  expect(parsed.melFilters.nFft).toBe(4)
  expect(parsed.melFilters.data.length).toBe(8)
  expect(parsed.melFilters.data[0]).toBeCloseTo(0)
  expect(parsed.melFilters.data[1]).toBeCloseTo(0.1)
})

test('parseWhisperGGML reads vocab', () => {
  const buffer = buildTestGGML()
  const parsed = parseWhisperGGML(buffer)

  expect(parsed.vocab.length).toBe(3)
  expect(parsed.vocab[0]).toBe('hello')
  expect(parsed.vocab[1]).toBe('world')
  expect(parsed.vocab[2]).toBe('')
})

test('parseWhisperGGML reads tensors', () => {
  const buffer = buildTestGGML()
  const parsed = parseWhisperGGML(buffer)

  expect(parsed.tensors.length).toBe(1)
  expect(parsed.tensors[0].name).toBe('test.weight')
  expect(parsed.tensors[0].dims).toEqual([3, 2])
  expect(parsed.tensors[0].ttype).toBe(GGML_TYPE.F32)
  expect(parsed.tensors[0].nElements).toBe(6)
})

test('parseWhisperGGML rejects bad magic', () => {
  const buffer = new ArrayBuffer(256)
  const view = new DataView(buffer)
  view.setUint32(0, 0x12345678, true)
  expect(() => parseWhisperGGML(buffer)).toThrow('Not a GGML file')
})
