// examples/whisper/tokenizer.js
// Whisper token decoder. Handles the GPT-2 BPE vocab stored in GGML files.
// Whisper uses a byte-level BPE vocabulary with special tokens for language,
// task, and timestamps.

// Whisper special token IDs (multilingual model)
const SPECIAL_TOKENS = {
  EOT: 50257,           // <|endoftext|>
  SOT: 50258,           // <|startoftranscript|>
  TRANSLATE: 50358,     // <|translate|>
  TRANSCRIBE: 50359,    // <|transcribe|>
  SOLM: 50360,          // <|startoflm|>
  PREV: 50361,          // <|startofprev|>
  NOSP: 50362,          // <|nospeech|>
  NOT: 50363,           // <|notimestamps|>
  BEG: 50364,           // <|0.00|> (first timestamp)
}

// Language tokens start at 50259 through 50357 (99 languages)
const LANGUAGES = [
  'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca',
  'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms',
  'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la',
  'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn',
  'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw',
  'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be',
  'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn',
  'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha',
  'ba', 'jw', 'su', 'yue',
]

function languageToken(lang) {
  const idx = LANGUAGES.indexOf(lang)
  return idx >= 0 ? 50259 + idx : 50259 // default to en
}

// GPT-2 byte decoder: maps unicode chars back to bytes
function buildByteDecoder() {
  const decoder = {}
  // GPT-2 uses a byte-to-unicode mapping
  // Characters 33-126, 161-172, 174-255 map to themselves
  // Bytes 0-32, 127-160, 173 map to 256+
  const bs = []
  for (let i = 33; i <= 126; i++) bs.push(i)
  for (let i = 161; i <= 172; i++) bs.push(i)
  for (let i = 174; i <= 255; i++) bs.push(i)
  const cs = [...bs]
  let n = 0
  for (let i = 0; i < 256; i++) {
    if (!bs.includes(i)) {
      bs.push(i)
      cs.push(256 + n)
      n++
    }
  }
  for (let i = 0; i < bs.length; i++) {
    decoder[String.fromCodePoint(cs[i])] = bs[i]
  }
  return decoder
}

function createTokenizer(vocab) {
  const byteDecoder = buildByteDecoder()

  function decode(tokens) {
    const pieces = []
    for (const id of tokens) {
      // Skip special tokens
      if (id >= 50257) continue
      if (id < 0 || id >= vocab.length) continue
      pieces.push(vocab[id])
    }

    // GPT-2 BPE tokens are unicode-encoded bytes — decode them
    const text = pieces.join('')
    const bytes = []
    for (const ch of text) {
      if (ch in byteDecoder) {
        bytes.push(byteDecoder[ch])
      } else {
        // Fallback for chars not in the byte decoder
        const code = ch.codePointAt(0)
        if (code < 256) bytes.push(code)
      }
    }
    return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes))
  }

  function isTimestamp(token) {
    return token >= SPECIAL_TOKENS.BEG
  }

  function timestampToSeconds(token) {
    return (token - SPECIAL_TOKENS.BEG) * 0.02
  }

  return { decode, isTimestamp, timestampToSeconds, vocab }
}

export { createTokenizer, SPECIAL_TOKENS, LANGUAGES, languageToken }
