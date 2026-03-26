// examples/autoresearch/tests/data.test.js
// Tests for autoresearch data loading.
// Run: bun test examples/autoresearch/tests/

import { test, expect } from 'bun:test'
import { createDataLoader, evaluateBPB } from '../data.js'

test('createDataLoader yields correct shapes', () => {
  const tokens = new Uint16Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  const loader = createDataLoader(tokens, 4)

  const { input, target } = loader.next()
  expect(input.length).toBe(4)
  expect(target.length).toBe(4)

  // input is tokens[0..3], target is tokens[1..4]
  expect(input).toEqual([0, 1, 2, 3])
  expect(target).toEqual([1, 2, 3, 4])
})

test('createDataLoader advances position', () => {
  const tokens = new Uint16Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
  const loader = createDataLoader(tokens, 3)

  const a = loader.next()
  expect(a.input).toEqual([10, 20, 30])

  const b = loader.next()
  expect(b.input).toEqual([40, 50, 60])
})

test('createDataLoader wraps around', () => {
  const tokens = new Uint16Array([1, 2, 3, 4, 5])
  const loader = createDataLoader(tokens, 3)

  // First: pos 0→3 (tokens[0..2], tokens[1..3])
  loader.next()
  // Second: pos 3, need 4 tokens (3+1), only 2 left → wraps to 0
  const { input } = loader.next()
  expect(input).toEqual([1, 2, 3])  // back to start
})

test('createDataLoader reset', () => {
  const tokens = new Uint16Array([1, 2, 3, 4, 5, 6, 7, 8])
  const loader = createDataLoader(tokens, 3)

  loader.next()
  loader.next()
  loader.reset()

  const { input } = loader.next()
  expect(input).toEqual([1, 2, 3])
})

test('evaluateBPB computes finite result', () => {
  const tokenLosses = [2.0, 3.0, 1.5, 2.5]
  const tokenIds = [0, 1, 2, 3]
  const tokenizer = {
    vocab: [
      new Uint8Array([104, 101]),  // "he" — 2 bytes
      new Uint8Array([108, 108]),  // "ll" — 2 bytes
      new Uint8Array([111]),       // "o"  — 1 byte
      new Uint8Array([32]),        // " "  — 1 byte
    ]
  }

  const bpb = evaluateBPB(tokenLosses, tokenIds, tokenizer)
  expect(isFinite(bpb)).toBe(true)
  expect(bpb).toBeGreaterThan(0)

  // Manual: totalNats = 2+3+1.5+2.5 = 9, totalBytes = 2+2+1+1 = 6
  // bpb = 9 / (ln2 * 6) = 9 / 4.158 ≈ 2.164
  expect(Math.abs(bpb - 9 / (Math.log(2) * 6))).toBeLessThan(0.01)
})

test('evaluateBPB skips tokens with 0 bytes', () => {
  const tokenLosses = [2.0, 999.0, 1.5]
  const tokenIds = [0, 1, 2]
  const tokenizer = {
    vocab: [
      new Uint8Array([65]),   // "A" — 1 byte
      null,                    // special token — 0 bytes, should be skipped
      new Uint8Array([66]),   // "B" — 1 byte
    ]
  }

  const bpb = evaluateBPB(tokenLosses, tokenIds, tokenizer)
  // Should only count tokens 0 and 2: totalNats=3.5, totalBytes=2
  expect(Math.abs(bpb - 3.5 / (Math.log(2) * 2))).toBeLessThan(0.01)
})

test('totalTokens property', () => {
  const tokens = new Uint16Array(100)
  const loader = createDataLoader(tokens, 10)
  expect(loader.totalTokens).toBe(100)
})
