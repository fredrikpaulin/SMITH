// examples/whisper/chunk.js
// Chunked audio processing for Whisper.
// Splits audio longer than 30s into overlapping windows,
// transcribes each chunk independently, and stitches results.
//
// chunkAudio and stitchTranscriptions are pure functions — no GPU dependency.
// transcribeChunk and transcribeChunked lazily import model.js.

const WHISPER_CHUNK_SAMPLES = 30 * 16000 // 30 seconds at 16kHz = 480000

// Split audio into overlapping chunks.
// Returns [{ samples: Float32Array, offsetSamples: number }]
function chunkAudio(samples, opts = {}) {
  const chunkSamples = opts.chunkSamples || WHISPER_CHUNK_SAMPLES
  const overlapSamples = opts.overlapSamples || (1 * 16000) // 1 second default

  if (samples.length <= chunkSamples) {
    return [{ samples, offsetSamples: 0 }]
  }

  const step = chunkSamples - overlapSamples
  const chunks = []
  let pos = 0

  while (pos < samples.length) {
    const end = Math.min(pos + chunkSamples, samples.length)
    chunks.push({
      samples: samples.subarray(pos, end),
      offsetSamples: pos,
    })
    if (end >= samples.length) break
    pos += step
  }

  return chunks
}

// Stitch per-chunk transcriptions together.
// Deduplicates tokens in overlap regions by comparing decoded text.
function stitchTranscriptions(chunkResults, tokenizer, opts = {}) {
  if (chunkResults.length === 0) return []
  if (chunkResults.length === 1) return chunkResults[0].tokens

  const allTokens = []

  for (let i = 0; i < chunkResults.length; i++) {
    const chunk = chunkResults[i]
    if (i === 0) {
      allTokens.push(...chunk.tokens)
      continue
    }

    // For subsequent chunks, try to find overlap with previous text
    const prevText = tokenizer.decode(allTokens)
    const currTokens = chunk.tokens

    // Find the best overlap point: try progressively longer prefixes of
    // the current chunk and check if they match a suffix of the accumulated text
    let bestSkip = 0
    const maxCheck = Math.min(currTokens.length, 20)

    for (let skip = 1; skip <= maxCheck; skip++) {
      const prefix = tokenizer.decode(currTokens.slice(0, skip)).trim()
      if (prefix.length === 0) continue

      const trimPrev = prevText.trimEnd()
      if (trimPrev.endsWith(prefix)) {
        bestSkip = skip
        break
      }

      // Partial word overlap
      const suffixLen = Math.min(prefix.length, trimPrev.length)
      const prevSuffix = trimPrev.slice(-suffixLen)
      if (prevSuffix === prefix.slice(0, suffixLen) && suffixLen >= 3) {
        bestSkip = skip
        break
      }
    }

    allTokens.push(...currTokens.slice(bestSkip))
  }

  return allTokens
}

// Transcribe a single chunk. Returns { tokens, offsetMs }.
// Lazily imports model.js and mel.js to avoid GPU dependency at module load.
async function transcribeChunk(model, chunk, opts = {}) {
  const { melSpectrogram } = await import('./mel.js')
  const { whisperTranscribeCached, whisperTranscribe } = await import('./model.js')
  const { SPECIAL_TOKENS } = await import('./tokenizer.js')

  const { mel } = melSpectrogram(chunk.samples, {
    nMels: opts.nMels || 80,
    sampleRate: 16000,
  })

  const useCached = opts.useCached !== false
  const transcribeFn = useCached ? whisperTranscribeCached : whisperTranscribe

  const tokens = transcribeFn(model, mel, {
    maxTokens: opts.maxTokens || 224,
    temperature: opts.temperature || 0,
    eotToken: opts.eotToken || SPECIAL_TOKENS.EOT,
    sotToken: opts.sotToken || SPECIAL_TOKENS.SOT,
    langToken: opts.langToken,
    transcribeToken: opts.transcribeToken || SPECIAL_TOKENS.TRANSCRIBE,
    noTimestamps: opts.noTimestamps || SPECIAL_TOKENS.NOT,
    onToken: opts.onToken,
  })

  return {
    tokens,
    offsetMs: (chunk.offsetSamples / 16000) * 1000,
  }
}

// Full chunked transcription pipeline.
async function transcribeChunked(model, samples, opts = {}) {
  const sampleRate = 16000
  const overlapSeconds = opts.overlapSeconds || 1

  const chunks = chunkAudio(samples, {
    chunkSamples: WHISPER_CHUNK_SAMPLES,
    overlapSamples: overlapSeconds * sampleRate,
  })

  const results = []
  for (let i = 0; i < chunks.length; i++) {
    if (opts.onChunk) opts.onChunk(i, chunks.length)
    const result = await transcribeChunk(model, chunks[i], opts)
    results.push(result)
  }

  return {
    chunks: results,
    totalChunks: chunks.length,
    durationMs: (samples.length / sampleRate) * 1000,
  }
}

export {
  chunkAudio,
  stitchTranscriptions,
  transcribeChunk,
  transcribeChunked,
  WHISPER_CHUNK_SAMPLES,
}
