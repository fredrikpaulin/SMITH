// examples/whisper/tests/chunk.test.js
// Tests for chunked audio processing.
// chunkAudio and stitchTranscriptions are pure JS — no GPU needed.

import { test, expect, describe } from 'bun:test'
import { chunkAudio, stitchTranscriptions, WHISPER_CHUNK_SAMPLES } from '../chunk.js'

const SR = 16000

describe('chunkAudio', () => {
  test('short audio returns single chunk', () => {
    const samples = new Float32Array(10 * SR) // 10 seconds
    const chunks = chunkAudio(samples)
    expect(chunks.length).toBe(1)
    expect(chunks[0].samples).toBe(samples) // same reference, no copy
    expect(chunks[0].offsetSamples).toBe(0)
  })

  test('exactly 30s returns single chunk', () => {
    const samples = new Float32Array(30 * SR)
    const chunks = chunkAudio(samples)
    expect(chunks.length).toBe(1)
  })

  test('31s audio returns 2 chunks', () => {
    const samples = new Float32Array(31 * SR)
    const chunks = chunkAudio(samples, { overlapSamples: 1 * SR })
    expect(chunks.length).toBe(2)
    expect(chunks[0].offsetSamples).toBe(0)
    expect(chunks[0].samples.length).toBe(30 * SR)
    expect(chunks[1].offsetSamples).toBe(29 * SR) // 30s - 1s overlap
  })

  test('60s audio returns 3 chunks with 1s overlap', () => {
    const samples = new Float32Array(60 * SR)
    const chunks = chunkAudio(samples, { overlapSamples: 1 * SR })
    // step = 30s - 1s = 29s
    // chunk 0: [0, 30s), chunk 1: [29s, 59s), chunk 2: [58s, 60s)
    expect(chunks.length).toBe(3)
    expect(chunks[0].offsetSamples).toBe(0)
    expect(chunks[1].offsetSamples).toBe(29 * SR)
    expect(chunks[2].offsetSamples).toBe(58 * SR)
  })

  test('90s audio returns 4 chunks', () => {
    const samples = new Float32Array(90 * SR)
    const chunks = chunkAudio(samples, { overlapSamples: 1 * SR })
    // step = 29s. chunks at: 0, 29, 58, 87
    expect(chunks.length).toBe(4)
  })

  test('overlap samples are shared between chunks', () => {
    const samples = new Float32Array(35 * SR)
    // Fill with distinct values
    for (let i = 0; i < samples.length; i++) samples[i] = i

    const chunks = chunkAudio(samples, { overlapSamples: 2 * SR })
    expect(chunks.length).toBe(2)

    // Last 2s of chunk 0 should match first 2s of chunk 1
    const overlapStart = chunks[1].offsetSamples
    const overlap0 = chunks[0].samples.subarray(overlapStart - chunks[0].offsetSamples)
    const overlap1 = chunks[1].samples.subarray(0, overlap0.length)
    for (let i = 0; i < overlap0.length; i++) {
      expect(overlap0[i]).toBe(overlap1[i])
    }
  })

  test('last chunk may be shorter than 30s', () => {
    const samples = new Float32Array(32 * SR)
    const chunks = chunkAudio(samples, { overlapSamples: 1 * SR })
    expect(chunks.length).toBe(2)
    // Second chunk: from 29s to 32s = 3s
    expect(chunks[1].samples.length).toBe(3 * SR)
  })

  test('custom chunk size', () => {
    const samples = new Float32Array(20 * SR)
    const chunks = chunkAudio(samples, {
      chunkSamples: 10 * SR,
      overlapSamples: 1 * SR,
    })
    // step = 9s. chunks at: 0, 9, 18
    expect(chunks.length).toBe(3)
    expect(chunks[0].samples.length).toBe(10 * SR)
    expect(chunks[1].samples.length).toBe(10 * SR)
    expect(chunks[2].samples.length).toBe(2 * SR) // 20s - 18s
  })

  test('zero-length audio returns single empty chunk', () => {
    const samples = new Float32Array(0)
    const chunks = chunkAudio(samples)
    expect(chunks.length).toBe(1)
    expect(chunks[0].samples.length).toBe(0)
  })
})

describe('stitchTranscriptions', () => {
  // Mock tokenizer that maps token IDs to single characters
  const mockTokenizer = {
    decode(tokens) {
      return tokens
        .filter(t => t < 50257) // skip special tokens
        .map(t => String.fromCharCode(65 + t)) // 0→A, 1→B, etc.
        .join('')
    },
  }

  test('empty input returns empty array', () => {
    const result = stitchTranscriptions([], mockTokenizer)
    expect(result).toEqual([])
  })

  test('single chunk returns its tokens unchanged', () => {
    const chunks = [{ tokens: [0, 1, 2], offsetMs: 0 }]
    const result = stitchTranscriptions(chunks, mockTokenizer)
    expect(result).toEqual([0, 1, 2])
  })

  test('non-overlapping chunks are concatenated', () => {
    const chunks = [
      { tokens: [0, 1, 2], offsetMs: 0 },       // "ABC"
      { tokens: [3, 4, 5], offsetMs: 30000 },    // "DEF"
    ]
    const result = stitchTranscriptions(chunks, mockTokenizer)
    // No overlap text match, so all tokens are kept
    expect(result).toEqual([0, 1, 2, 3, 4, 5])
  })

  test('overlapping chunks deduplicate matching prefix', () => {
    const chunks = [
      { tokens: [0, 1, 2, 3], offsetMs: 0 },       // "ABCD"
      { tokens: [2, 3, 4, 5], offsetMs: 29000 },    // "CDEF" — first 2 tokens overlap
    ]
    const result = stitchTranscriptions(chunks, mockTokenizer)
    const text = mockTokenizer.decode(result)
    // Should produce "ABCDEF" — "CD" deduplicated
    expect(text).toBe('ABCDEF')
  })

  test('three chunks with overlap', () => {
    const chunks = [
      { tokens: [0, 1, 2, 3], offsetMs: 0 },       // "ABCD"
      { tokens: [3, 4, 5, 6], offsetMs: 29000 },    // "DEFG" — D overlaps
      { tokens: [6, 7, 8], offsetMs: 58000 },        // "GHI" — G overlaps
    ]
    const result = stitchTranscriptions(chunks, mockTokenizer)
    const text = mockTokenizer.decode(result)
    expect(text).toBe('ABCDEFGHI')
  })

  test('no overlap match keeps all tokens', () => {
    const chunks = [
      { tokens: [0, 1], offsetMs: 0 },      // "AB"
      { tokens: [10, 11], offsetMs: 29000 }, // "KL"
    ]
    const result = stitchTranscriptions(chunks, mockTokenizer)
    expect(result).toEqual([0, 1, 10, 11])
  })
})

describe('WHISPER_CHUNK_SAMPLES', () => {
  test('is 30 seconds at 16kHz', () => {
    expect(WHISPER_CHUNK_SAMPLES).toBe(480000)
  })
})
