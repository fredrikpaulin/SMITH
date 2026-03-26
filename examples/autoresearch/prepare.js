#!/usr/bin/env bun
// examples/autoresearch/prepare.js
// Download a small text dataset and prepare tokenized binary data.
//
// Usage:
//   bun examples/autoresearch/prepare.js [--out data/] [--vocab 4096] [--chars 2000000]
//
// Downloads ~2M chars of text from Project Gutenberg (public domain),
// trains a BPE tokenizer, and saves tokenized train/val splits.

import { parseArgs } from 'node:util'
import { mkdirSync, existsSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const { values: args } = parseArgs({
  options: {
    out: { type: 'string', default: join(import.meta.dir, 'data') },
    vocab: { type: 'string', default: '4096' },
    chars: { type: 'string', default: '2000000' },
  },
})

const OUT_DIR = args.out
const VOCAB_SIZE = parseInt(args.vocab)
const MAX_CHARS = parseInt(args.chars)

// Public domain texts from Project Gutenberg (plain text URLs)
const TEXTS = [
  'https://www.gutenberg.org/cache/epub/1342/pg1342.txt',  // Pride and Prejudice
  'https://www.gutenberg.org/cache/epub/84/pg84.txt',      // Frankenstein
  'https://www.gutenberg.org/cache/epub/1661/pg1661.txt',  // Sherlock Holmes
  'https://www.gutenberg.org/cache/epub/11/pg11.txt',      // Alice in Wonderland
  'https://www.gutenberg.org/cache/epub/2701/pg2701.txt',  // Moby Dick
  'https://www.gutenberg.org/cache/epub/1952/pg1952.txt',  // The Yellow Wallpaper
  'https://www.gutenberg.org/cache/epub/174/pg174.txt',    // Dorian Gray
  'https://www.gutenberg.org/cache/epub/98/pg98.txt',      // A Tale of Two Cities
  'https://www.gutenberg.org/cache/epub/1260/pg1260.txt',  // Jane Eyre
  'https://www.gutenberg.org/cache/epub/16328/pg16328.txt', // Beowulf
]

const textDir = join(OUT_DIR, 'raw')

async function downloadTexts() {
  mkdirSync(textDir, { recursive: true })

  let totalChars = 0
  for (let i = 0; i < TEXTS.length && totalChars < MAX_CHARS; i++) {
    const url = TEXTS[i]
    const filename = `text_${String(i).padStart(2, '0')}.txt`
    const path = join(textDir, filename)

    if (existsSync(path)) {
      const existing = await Bun.file(path).text()
      totalChars += existing.length
      console.log(`  ${filename}: ${existing.length} chars (cached)`)
      continue
    }

    console.log(`  Downloading ${url}...`)
    try {
      const resp = await fetch(url)
      if (!resp.ok) { console.log(`  Failed: ${resp.status}`); continue }
      let text = await resp.text()

      // Strip Gutenberg header/footer (rough)
      const startMarker = '*** START OF'
      const endMarker = '*** END OF'
      const si = text.indexOf(startMarker)
      const ei = text.lastIndexOf(endMarker)
      if (si !== -1) text = text.slice(text.indexOf('\n', si) + 1)
      if (ei !== -1) text = text.slice(0, ei)

      writeFileSync(path, text)
      totalChars += text.length
      console.log(`  ${filename}: ${text.length} chars`)
    } catch (e) {
      console.log(`  Failed: ${e.message}`)
    }
  }

  console.log(`Total: ${totalChars} chars`)
}

async function main() {
  console.log(`Preparing autoresearch data in ${OUT_DIR}`)
  console.log(`  vocab_size=${VOCAB_SIZE}, max_chars=${MAX_CHARS}\n`)

  // Check if already prepared
  if (existsSync(join(OUT_DIR, 'train.bin')) && existsSync(join(OUT_DIR, 'val.bin'))) {
    console.log('Data already prepared. Delete data/ to re-prepare.')
    return
  }

  // Step 1: Download texts
  console.log('Step 1: Downloading texts...')
  await downloadTexts()
  console.log()

  // Step 2: Tokenize
  console.log('Step 2: Training tokenizer and tokenizing...')
  const { prepareData } = await import('./data.js')
  await prepareData(textDir, OUT_DIR, VOCAB_SIZE)
  console.log()

  console.log('Done! Ready to train with:')
  console.log('  bun examples/autoresearch/train.js')
}

main().catch(e => { console.error(e); process.exit(1) })
