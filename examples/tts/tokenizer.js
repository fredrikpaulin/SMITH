// examples/tts/tokenizer.js
// BPE tokenizer for Qwen3-TTS. Loads vocab.json + merges.txt from the model directory.
// Compatible with the Qwen tokenizer format (byte-level BPE with special tokens).

const SPECIAL_TOKENS = {
  '<|im_start|>': 151644,
  '<|im_end|>': 151645,
  '<|tts_bos|>': 151672,
  '<|tts_eos|>': 151673,
  '<|tts_pad|>': 151671,
}

async function loadTokenizer(modelDir) {
  const vocabRaw = await Bun.file(`${modelDir}/vocab.json`).json()
  const mergesRaw = await Bun.file(`${modelDir}/merges.txt`).text()

  // vocab: string → id
  const vocab = vocabRaw

  // merges: array of [a, b] pairs
  const mergeLines = mergesRaw.split('\n').filter(l => l && !l.startsWith('#'))
  const merges = new Map()
  for (let i = 0; i < mergeLines.length; i++) {
    const parts = mergeLines[i].split(' ')
    if (parts.length === 2) {
      merges.set(parts[0] + ' ' + parts[1], i)
    }
  }

  // byte encoder: maps each byte value to a unicode character
  // This is the standard GPT-2 byte-level BPE mapping
  const byteEncoder = buildByteEncoder()
  const byteDecoder = {}
  for (const [k, v] of Object.entries(byteEncoder)) {
    byteDecoder[v] = parseInt(k)
  }

  return {
    vocab,
    merges,
    byteEncoder,
    byteDecoder,
    encode: (text) => bpeEncode(text, vocab, merges, byteEncoder),
    encodeChat: (text) => encodeChatTTS(text, vocab, merges, byteEncoder),
  }
}

// Build the byte-level encoding table (GPT-2 style)
function buildByteEncoder() {
  const bs = []
  // Printable ASCII range
  for (let i = 33; i <= 126; i++) bs.push(i)     // ! through ~
  for (let i = 161; i <= 172; i++) bs.push(i)     // ¡ through ¬
  for (let i = 174; i <= 255; i++) bs.push(i)     // ® through ÿ
  const cs = [...bs]
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n)
      n++
    }
  }
  const encoder = {}
  for (let i = 0; i < bs.length; i++) {
    encoder[bs[i]] = String.fromCharCode(cs[i])
  }
  return encoder
}

// Convert text to byte-level BPE tokens
function bpeEncode(text, vocab, merges, byteEncoder) {
  // Convert to bytes then to byte-level characters
  const encoder = new TextEncoder()
  const bytes = encoder.encode(text)
  let tokens = []
  for (const b of bytes) {
    tokens.push(byteEncoder[b])
  }

  // Apply BPE merges greedily
  tokens = applyBPE(tokens, merges)

  // Map to vocab IDs
  const ids = []
  for (const t of tokens) {
    if (vocab[t] !== undefined) {
      ids.push(vocab[t])
    } else {
      // Fallback: encode each character individually
      for (const ch of t) {
        if (vocab[ch] !== undefined) ids.push(vocab[ch])
      }
    }
  }

  return ids
}

function applyBPE(tokens, merges) {
  if (tokens.length < 2) return tokens

  while (true) {
    // Find the pair with the lowest merge rank
    let bestPair = null
    let bestRank = Infinity
    for (let i = 0; i < tokens.length - 1; i++) {
      const key = tokens[i] + ' ' + tokens[i + 1]
      const rank = merges.get(key)
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank
        bestPair = [tokens[i], tokens[i + 1]]
      }
    }

    if (!bestPair) break

    // Merge all occurrences of this pair
    const merged = bestPair[0] + bestPair[1]
    const newTokens = []
    let i = 0
    while (i < tokens.length) {
      if (i < tokens.length - 1 && tokens[i] === bestPair[0] && tokens[i + 1] === bestPair[1]) {
        newTokens.push(merged)
        i += 2
      } else {
        newTokens.push(tokens[i])
        i++
      }
    }
    tokens = newTokens
    if (tokens.length < 2) break
  }

  return tokens
}

// Encode text for TTS chat format:
// <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
function encodeChatTTS(text, vocab, merges, byteEncoder) {
  const textIds = bpeEncode(text, vocab, merges, byteEncoder)

  // Build the full sequence:
  // <|im_start|> "assistant" \n {text} <|im_end|> \n <|im_start|> "assistant" \n
  const assistantIds = bpeEncode('assistant', vocab, merges, byteEncoder)
  const newlineIds = bpeEncode('\n', vocab, merges, byteEncoder)

  return [
    SPECIAL_TOKENS['<|im_start|>'],
    ...assistantIds,
    ...newlineIds,
    ...textIds,
    SPECIAL_TOKENS['<|im_end|>'],
    ...newlineIds,
    SPECIAL_TOKENS['<|im_start|>'],
    ...assistantIds,
    ...newlineIds,
  ]
}

export { loadTokenizer, SPECIAL_TOKENS }
