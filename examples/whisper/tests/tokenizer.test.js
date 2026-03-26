import { test, expect } from 'bun:test'
import { createTokenizer, SPECIAL_TOKENS, LANGUAGES, languageToken } from '../tokenizer.js'

test('SPECIAL_TOKENS has expected values', () => {
  expect(SPECIAL_TOKENS.EOT).toBe(50257)
  expect(SPECIAL_TOKENS.SOT).toBe(50258)
  expect(SPECIAL_TOKENS.TRANSCRIBE).toBe(50359)
  expect(SPECIAL_TOKENS.NOT).toBe(50363)
  expect(SPECIAL_TOKENS.BEG).toBe(50364)
})

test('languageToken returns correct tokens', () => {
  expect(languageToken('en')).toBe(50259) // First language
  expect(languageToken('zh')).toBe(50260) // Second language
  expect(languageToken('unknown')).toBe(50259) // Default to en
})

test('LANGUAGES has expected entries', () => {
  expect(LANGUAGES[0]).toBe('en')
  expect(LANGUAGES[1]).toBe('zh')
  expect(LANGUAGES.includes('sv')).toBe(true)
  expect(LANGUAGES.includes('ja')).toBe(true)
})

test('createTokenizer decodes simple tokens', () => {
  // Build a minimal vocab with GPT-2 byte-encoded entries
  const vocab = new Array(51865).fill('')
  // ASCII 'H' = 72, which in GPT-2 BPE maps to 'H' directly (code 72 is in the printable range)
  vocab[39] = 'H'
  vocab[72] = 'ello'

  const tokenizer = createTokenizer(vocab)
  const text = tokenizer.decode([39, 72])
  expect(text).toBe('Hello')
})

test('createTokenizer skips special tokens', () => {
  const vocab = new Array(51865).fill('')
  vocab[100] = 'test'

  const tokenizer = createTokenizer(vocab)
  // Special tokens (≥50257) should be skipped
  const text = tokenizer.decode([50258, 50259, 100, 50257])
  expect(text).toBe('test')
})

test('isTimestamp identifies timestamp tokens', () => {
  const vocab = new Array(51865).fill('')
  const tokenizer = createTokenizer(vocab)
  expect(tokenizer.isTimestamp(50364)).toBe(true) // <|0.00|>
  expect(tokenizer.isTimestamp(50365)).toBe(true) // <|0.02|>
  expect(tokenizer.isTimestamp(50363)).toBe(false) // <|notimestamps|>
  expect(tokenizer.isTimestamp(100)).toBe(false) // Regular token
})

test('timestampToSeconds converts correctly', () => {
  const vocab = new Array(51865).fill('')
  const tokenizer = createTokenizer(vocab)
  expect(tokenizer.timestampToSeconds(50364)).toBeCloseTo(0.0)
  expect(tokenizer.timestampToSeconds(50365)).toBeCloseTo(0.02)
  expect(tokenizer.timestampToSeconds(50414)).toBeCloseTo(1.0) // 50 * 0.02
})
