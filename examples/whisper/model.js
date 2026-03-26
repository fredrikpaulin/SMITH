// examples/whisper/model.js
// Whisper encoder-decoder transformer architecture for Smith.
// Uses Smith's core ops: conv1d, sinusoidalPE, multiHeadAttention, multiHeadCrossAttention.
//
// Architecture:
//   Encoder: 2x Conv1d → sinusoidal PE → N transformer blocks (self-attention + FFN)
//   Decoder: token embedding + learned PE → N blocks (self-attn + cross-attn + FFN)

import smith from '../../src/index.js'

const { variable, tensor, zeros, ones, conv1d, sinusoidalPE,
  matmul, add, gelu, layernorm, softmax, reshape, transpose, scale,
  noGrad, embedding,
  createLinear, linear, linearParams,
  createMultiHeadAttention, multiHeadAttention, multiHeadCrossAttention, mhaParams,
  createCausalMask,
} = smith

// --- Encoder block ---

function createEncoderBlock(dim, numHeads, ffnDim) {
  return {
    attnLnW: variable(ones([dim]), { requiresGrad: true }),
    attnLnB: variable(zeros([dim]), { requiresGrad: true }),
    mha: createMultiHeadAttention(dim, numHeads),
    ffnLnW: variable(ones([dim]), { requiresGrad: true }),
    ffnLnB: variable(zeros([dim]), { requiresGrad: true }),
    ffn1: createLinear(dim, ffnDim),
    ffn2: createLinear(ffnDim, dim),
    dim, numHeads, ffnDim,
  }
}

function encoderBlock(x, block) {
  const norm1 = layernorm(x, block.attnLnW, block.attnLnB)
  const { output: attnOut } = multiHeadAttention(norm1, block.mha)
  const x2 = add(x, attnOut)

  const norm2 = layernorm(x2, block.ffnLnW, block.ffnLnB)
  const ffnOut = linear(gelu(linear(norm2, block.ffn1)), block.ffn2)
  return add(x2, ffnOut)
}

function encoderBlockParams(block) {
  return [
    block.attnLnW, block.attnLnB,
    ...mhaParams(block.mha),
    block.ffnLnW, block.ffnLnB,
    ...linearParams(block.ffn1),
    ...linearParams(block.ffn2),
  ]
}

// --- Decoder block (self-attention + cross-attention + FFN) ---

function createDecoderBlock(dim, numHeads, ffnDim) {
  return {
    selfAttnLnW: variable(ones([dim]), { requiresGrad: true }),
    selfAttnLnB: variable(zeros([dim]), { requiresGrad: true }),
    selfAttn: createMultiHeadAttention(dim, numHeads),
    crossAttnLnW: variable(ones([dim]), { requiresGrad: true }),
    crossAttnLnB: variable(zeros([dim]), { requiresGrad: true }),
    crossAttn: createMultiHeadAttention(dim, numHeads),
    ffnLnW: variable(ones([dim]), { requiresGrad: true }),
    ffnLnB: variable(zeros([dim]), { requiresGrad: true }),
    ffn1: createLinear(dim, ffnDim),
    ffn2: createLinear(ffnDim, dim),
    dim, numHeads, ffnDim,
  }
}

function decoderBlock(x, block, encoderOut, mask) {
  // Self-attention with causal mask
  const norm1 = layernorm(x, block.selfAttnLnW, block.selfAttnLnB)
  const { output: selfAttnOut } = multiHeadAttention(norm1, block.selfAttn, mask)
  const x2 = add(x, selfAttnOut)

  // Cross-attention to encoder output
  const norm2 = layernorm(x2, block.crossAttnLnW, block.crossAttnLnB)
  const { output: crossAttnOut } = multiHeadCrossAttention(norm2, encoderOut, block.crossAttn)
  const x3 = add(x2, crossAttnOut)

  // FFN
  const norm3 = layernorm(x3, block.ffnLnW, block.ffnLnB)
  const ffnOut = linear(gelu(linear(norm3, block.ffn1)), block.ffn2)
  return add(x3, ffnOut)
}

function decoderBlockParams(block) {
  return [
    block.selfAttnLnW, block.selfAttnLnB,
    ...mhaParams(block.selfAttn),
    block.crossAttnLnW, block.crossAttnLnB,
    ...mhaParams(block.crossAttn),
    block.ffnLnW, block.ffnLnB,
    ...linearParams(block.ffn1),
    ...linearParams(block.ffn2),
  ]
}

// --- Full Whisper model ---

const WHISPER_CONFIGS = {
  tiny:   { dim: 384, encoderLayers: 4,  decoderLayers: 4,  numHeads: 6,  ffnDim: 1536, nMels: 80,  vocabSize: 51865, maxTextCtx: 448 },
  base:   { dim: 512, encoderLayers: 6,  decoderLayers: 6,  numHeads: 8,  ffnDim: 2048, nMels: 80,  vocabSize: 51865, maxTextCtx: 448 },
  small:  { dim: 768, encoderLayers: 12, decoderLayers: 12, numHeads: 12, ffnDim: 3072, nMels: 80,  vocabSize: 51865, maxTextCtx: 448 },
  medium: { dim: 1024, encoderLayers: 24, decoderLayers: 24, numHeads: 16, ffnDim: 4096, nMels: 80,  vocabSize: 51865, maxTextCtx: 448 },
  large:  { dim: 1280, encoderLayers: 32, decoderLayers: 32, numHeads: 20, ffnDim: 5120, nMels: 128, vocabSize: 51865, maxTextCtx: 448 },
}

function createWhisperModel(config) {
  const { dim, encoderLayers, decoderLayers, numHeads, ffnDim, nMels, vocabSize, maxTextCtx } = config

  return {
    config,
    // Encoder conv layers
    conv1W: variable(tensor(Array.from({ length: dim * nMels * 3 }, () => (Math.random() * 2 - 1) * 0.02), [dim, nMels, 3]), { requiresGrad: true }),
    conv1B: variable(zeros([dim]), { requiresGrad: true }),
    conv2W: variable(tensor(Array.from({ length: dim * dim * 3 }, () => (Math.random() * 2 - 1) * 0.02), [dim, dim, 3]), { requiresGrad: true }),
    conv2B: variable(zeros([dim]), { requiresGrad: true }),
    // Encoder positional embedding (sinusoidal, computed on first use)
    encoderPE: null,
    encoderBlocks: Array.from({ length: encoderLayers }, () => createEncoderBlock(dim, numHeads, ffnDim)),
    encoderLnW: variable(ones([dim]), { requiresGrad: true }),
    encoderLnB: variable(zeros([dim]), { requiresGrad: true }),
    // Decoder embeddings
    tokenEmbed: variable(tensor(Array.from({ length: vocabSize * dim }, () => (Math.random() * 2 - 1) * 0.02), [vocabSize, dim]), { requiresGrad: true }),
    decoderPE: variable(tensor(Array.from({ length: maxTextCtx * dim }, () => (Math.random() * 2 - 1) * 0.02), [maxTextCtx, dim]), { requiresGrad: true }),
    decoderBlocks: Array.from({ length: decoderLayers }, () => createDecoderBlock(dim, numHeads, ffnDim)),
    decoderLnW: variable(ones([dim]), { requiresGrad: true }),
    decoderLnB: variable(zeros([dim]), { requiresGrad: true }),
  }
}

function whisperEncode(model, melInput) {
  const { config } = model
  const numFrames = melInput.length / config.nMels

  const mel = variable(tensor(Array.from(melInput), [config.nMels, numFrames]), { requiresGrad: false })

  // Conv1: [nMels, numFrames] → [dim, numFrames], kernel=3, stride=1, padding=1
  let x = gelu(conv1d(mel, model.conv1W, model.conv1B, { stride: 1, padding: 1 }))

  // Conv2: [dim, numFrames] → [dim, numFrames/2], kernel=3, stride=2, padding=1
  x = gelu(conv1d(x, model.conv2W, model.conv2B, { stride: 2, padding: 1 }))

  // [dim, audioCtx] → [audioCtx, dim] for transformer
  const audioCtx = x.data.shape[1]
  x = transpose(x, [1, 0])

  // Add sinusoidal positional embedding (from Smith core)
  if (!model.encoderPE || model.encoderPE.data.shape[0] !== audioCtx) {
    model.encoderPE = variable(sinusoidalPE(audioCtx, config.dim), { requiresGrad: false })
  }
  x = add(x, model.encoderPE)

  for (const block of model.encoderBlocks) {
    x = encoderBlock(x, block)
  }

  return layernorm(x, model.encoderLnW, model.encoderLnB)
}

function whisperDecode(model, encoderOut, tokens) {
  const { config } = model
  const seqLen = tokens.length

  let x = embedding(tokens, model.tokenEmbed)

  // Slice learned decoder PE to seqLen
  const peData = model.decoderPE.data.data.slice(0, seqLen * config.dim)
  const pe = variable(tensor(Array.from(peData), [seqLen, config.dim]), { requiresGrad: false })
  x = add(x, pe)

  const mask = createCausalMask(seqLen)

  for (const block of model.decoderBlocks) {
    x = decoderBlock(x, block, encoderOut, mask)
  }

  x = layernorm(x, model.decoderLnW, model.decoderLnB)

  // Weight-tied output projection
  return matmul(x, transpose(model.tokenEmbed, [1, 0]))
}

// --- Greedy decoding ---

function whisperTranscribe(model, melInput, opts = {}) {
  const maxTokens = opts.maxTokens || 224
  const temperature = opts.temperature || 0
  const eotToken = opts.eotToken || 50257
  const sotToken = opts.sotToken || 50258
  const langToken = opts.langToken || 50259
  const transcribeToken = opts.transcribeToken || 50359
  const noTimestamps = opts.noTimestamps || 50363

  return noGrad(() => {
    const encoderOut = whisperEncode(model, melInput)
    const tokens = [sotToken, langToken, transcribeToken, noTimestamps]
    const result = []

    for (let step = 0; step < maxTokens; step++) {
      const logits = whisperDecode(model, encoderOut, tokens)
      const lastLogits = logits.data.data.slice(
        (tokens.length - 1) * model.config.vocabSize,
        tokens.length * model.config.vocabSize
      )

      let nextToken
      if (temperature === 0) {
        nextToken = 0
        let maxVal = lastLogits[0]
        for (let i = 1; i < lastLogits.length; i++) {
          if (lastLogits[i] > maxVal) { maxVal = lastLogits[i]; nextToken = i }
        }
      } else {
        const scaled = new Float32Array(lastLogits.length)
        let maxVal = -Infinity
        for (let i = 0; i < scaled.length; i++) {
          scaled[i] = lastLogits[i] / temperature
          if (scaled[i] > maxVal) maxVal = scaled[i]
        }
        let sum = 0
        for (let i = 0; i < scaled.length; i++) {
          scaled[i] = Math.exp(scaled[i] - maxVal)
          sum += scaled[i]
        }
        for (let i = 0; i < scaled.length; i++) scaled[i] /= sum

        let r = Math.random()
        nextToken = scaled.length - 1
        for (let i = 0; i < scaled.length; i++) {
          r -= scaled[i]
          if (r <= 0) { nextToken = i; break }
        }
      }

      if (nextToken === eotToken) break
      result.push(nextToken)
      tokens.push(nextToken)

      if (opts.onToken) {
        const stop = opts.onToken(nextToken, step)
        if (stop) break
      }
    }

    return result
  })
}

function whisperParams(model) {
  const params = [model.conv1W, model.conv1B, model.conv2W, model.conv2B]
  params.push(model.encoderLnW, model.encoderLnB)
  for (const block of model.encoderBlocks) params.push(...encoderBlockParams(block))
  params.push(model.tokenEmbed, model.decoderPE)
  params.push(model.decoderLnW, model.decoderLnB)
  for (const block of model.decoderBlocks) params.push(...decoderBlockParams(block))
  return params
}

export {
  WHISPER_CONFIGS,
  createWhisperModel,
  whisperEncode, whisperDecode, whisperTranscribe,
  whisperParams,
  createEncoderBlock, encoderBlock, encoderBlockParams,
  createDecoderBlock, decoderBlock, decoderBlockParams,
}
