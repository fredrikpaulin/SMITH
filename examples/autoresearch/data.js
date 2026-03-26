// examples/autoresearch/data.js
// Data loading for autoresearch training.
// Binary format: concatenated uint16 token IDs.
// prepare.js downloads text and tokenizes it to this format.

import { train as trainBPE, encode as bpeEncode } from '../../src/tokenizer.js'

// --- Binary data format ---
// .bin files: flat array of uint16 token IDs, no headers.
// Training reads consecutive chunks of (seqLen+1) tokens:
//   input = tokens[i..i+T], target = tokens[i+1..i+T+1]

async function loadTokens(path) {
  const buf = await Bun.file(path).arrayBuffer()
  return new Uint16Array(buf)
}

// --- Simple data loader ---
// Returns an iterator that yields { input, target } arrays of token IDs.
// Reads sequentially through the token file, wrapping at the end.
// input/target are plain JS arrays of length seqLen.

function createDataLoader(tokens, seqLen) {
  let pos = 0
  const total = tokens.length

  return {
    next() {
      if (pos + seqLen + 1 > total) pos = 0
      const input = []
      const target = []
      for (let i = 0; i < seqLen; i++) {
        input.push(tokens[pos + i])
        target.push(tokens[pos + i + 1])
      }
      pos += seqLen
      return { input, target }
    },
    reset() { pos = 0 },
    get totalTokens() { return total },
  }
}

// --- Tokenizer training from text files ---
// Trains a BPE tokenizer on text, saves tokenizer + tokenized binary data.

async function prepareData(textDir, outDir, vocabSize = 4096) {
  const fs = await import('node:fs')
  const path = await import('node:path')

  // Read all text files
  const files = fs.readdirSync(textDir).filter(f => f.endsWith('.txt')).sort()
  if (files.length === 0) throw new Error(`No .txt files in ${textDir}`)

  let allText = ''
  for (const f of files) {
    allText += await Bun.file(path.join(textDir, f)).text()
  }
  console.log(`Read ${files.length} files, ${allText.length} chars`)

  // Train tokenizer
  console.log(`Training BPE tokenizer (vocab=${vocabSize})...`)
  const tokenizer = trainBPE(allText, vocabSize)
  console.log(`Tokenizer trained: ${tokenizer.vocabSize} tokens, ${tokenizer.merges.length} merges`)

  // Tokenize
  console.log('Tokenizing...')
  const tokens = bpeEncode(allText, tokenizer.merges)
  console.log(`${tokens.length} tokens (${(tokens.length / allText.length).toFixed(2)} tokens/char)`)

  // Save
  fs.mkdirSync(outDir, { recursive: true })

  // Save tokenizer
  const { save } = await import('../../src/tokenizer.js')
  await save(path.join(outDir, 'tokenizer.json'), tokenizer)

  // Split 95/5 train/val
  const splitIdx = Math.floor(tokens.length * 0.95)
  const trainTokens = tokens.slice(0, splitIdx)
  const valTokens = tokens.slice(splitIdx)

  // Save as uint16 binary
  const trainBuf = new Uint16Array(trainTokens)
  const valBuf = new Uint16Array(valTokens)
  await Bun.write(path.join(outDir, 'train.bin'), trainBuf.buffer)
  await Bun.write(path.join(outDir, 'val.bin'), valBuf.buffer)

  console.log(`Saved: train=${trainTokens.length} tokens, val=${valTokens.length} tokens`)
  return { tokenizer, trainTokens: trainTokens.length, valTokens: valTokens.length }
}

// --- BPB evaluation ---
// Bits per byte: sums per-token cross-entropy (nats), sums byte lengths,
// converts nats/byte to bits/byte. Comparable across vocab sizes.

function evaluateBPB(tokenLosses, tokenIds, tokenizer) {
  let totalNats = 0
  let totalBytes = 0
  const { vocab } = tokenizer

  for (let i = 0; i < tokenLosses.length; i++) {
    const tokenBytes = vocab[tokenIds[i]] ? vocab[tokenIds[i]].length : 0
    if (tokenBytes > 0) {
      totalNats += tokenLosses[i]
      totalBytes += tokenBytes
    }
  }

  return totalNats / (Math.log(2) * totalBytes)
}

export {
  loadTokens,
  createDataLoader,
  prepareData,
  evaluateBPB,
}
