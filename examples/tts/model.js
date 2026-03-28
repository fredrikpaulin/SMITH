// examples/tts/model.js
// Qwen3-TTS weight loader. Parses safetensors, maps weights to model structure.
// Handles both the main model (talker + code predictor) and speech tokenizer.

import smith from '../../src/index.js'
import { writeSync } from 'node:fs'

const { parseSafetensors, readTensor, listTensors } = smith

let _loadCount = 0

// Unbuffered write to stdout — bypasses process.stdout buffering
// so progress updates render mid-loop without yielding to the event loop.
function out(text) { writeSync(1, text) }

// Load a tensor from safetensors, convert to f32 GPU tensor.
// Large BF16 tensors (>50M elements) are stored as lazy — raw BF16 data
// with per-row conversion on demand. This avoids allocating and writing
// 1.24GB of Metal memory for embeddings that are 99.9% unused.
function loadWeight(parsed, name) {
  _loadCount++
  const raw = readTensor(parsed, name)
  const elems = raw.shape.reduce((a, b) => a * b, 1)
  const mb = (elems * 4 / 1e6).toFixed(0)
  out(`\r  [${_loadCount}] ${name} [${raw.shape}] ${mb}MB ${raw.dtype}   `)

  if (raw.dtype === 'BF16' && elems > 50_000_000) {
    const u16 = new Uint16Array(raw.data.buffer, raw.data.byteOffset, elems)
    out(`(lazy — ${mb}MB deferred)\n`)
    return { shape: raw.shape.slice(), bf16: u16, lazy: true }
  }

  const t = smith.zeros(raw.shape)

  if (raw.dtype === 'BF16') {
    const u16 = new Uint16Array(raw.data.buffer, raw.data.byteOffset, elems)
    const u32 = new Uint32Array(t.data.buffer, t.data.byteOffset, elems)
    for (let i = 0; i < elems; i++) u32[i] = u16[i] << 16
  } else if (raw.dtype === 'F16') {
    for (let i = 0; i < elems; i++) t.data[i] = smith.fromFloat16(raw.data[i])
  } else if (raw.dtype === 'F32') {
    const src = raw.data instanceof Float32Array
      ? raw.data
      : new Float32Array(raw.data.buffer, raw.data.byteOffset, elems)
    t.data.set(src)
  } else {
    throw new Error(`Unsupported dtype: ${raw.dtype} for ${name}`)
  }
  return t
}

// Try to load a weight, return null if not found
function tryLoad(parsed, name) {
  try {
    return loadWeight(parsed, name)
  } catch {
    return null
  }
}

// Load the Talker model (28-layer Llama-style transformer)
function loadTalker(parsed, config) {
  const c = config.talker_config
  const prefix = 'talker.model.'

  // Embeddings
  const codecEmbedding = loadWeight(parsed, `${prefix}codec_embedding.weight`)
  const textEmbedding = loadWeight(parsed, `${prefix}text_embedding.weight`)

  // Text projection MLP (text_hidden_size → hidden_size)
  const textProjection = {
    fc1: {
      weight: loadWeight(parsed, 'talker.text_projection.linear_fc1.weight'),
      bias: loadWeight(parsed, 'talker.text_projection.linear_fc1.bias'),
    },
    fc2: {
      weight: loadWeight(parsed, 'talker.text_projection.linear_fc2.weight'),
      bias: loadWeight(parsed, 'talker.text_projection.linear_fc2.bias'),
    },
  }

  // Transformer blocks
  const blocks = []
  for (let i = 0; i < c.num_hidden_layers; i++) {
    const lp = `${prefix}layers.${i}.`
    blocks.push({
      inputLayernorm: loadWeight(parsed, `${lp}input_layernorm.weight`),
      postAttnLayernorm: loadWeight(parsed, `${lp}post_attention_layernorm.weight`),
      qProj: loadWeight(parsed, `${lp}self_attn.q_proj.weight`),
      kProj: loadWeight(parsed, `${lp}self_attn.k_proj.weight`),
      vProj: loadWeight(parsed, `${lp}self_attn.v_proj.weight`),
      oProj: loadWeight(parsed, `${lp}self_attn.o_proj.weight`),
      qNorm: loadWeight(parsed, `${lp}self_attn.q_norm.weight`),
      kNorm: loadWeight(parsed, `${lp}self_attn.k_norm.weight`),
      gateProj: loadWeight(parsed, `${lp}mlp.gate_proj.weight`),
      upProj: loadWeight(parsed, `${lp}mlp.up_proj.weight`),
      downProj: loadWeight(parsed, `${lp}mlp.down_proj.weight`),
    })
  }

  // Final norm + output head
  const norm = loadWeight(parsed, `${prefix}norm.weight`)
  const codecHead = loadWeight(parsed, 'talker.codec_head.weight')

  return {
    codecEmbedding,
    textEmbedding,
    textProjection,
    blocks,
    norm,
    codecHead,
    config: {
      dim: c.hidden_size,
      numHeads: c.num_attention_heads,
      numKVHeads: c.num_key_value_heads,
      headDim: c.head_dim,
      numLayers: c.num_hidden_layers,
      intermediateDim: c.intermediate_size,
      vocabSize: c.vocab_size,
      textVocabSize: c.text_vocab_size,
      normEps: c.rms_norm_eps,
      ropeTheta: c.rope_theta,
      numCodeGroups: c.num_code_groups,
      codecBosId: c.codec_bos_id,
      codecEosId: c.codec_eos_token_id,
      codecPadId: c.codec_pad_id,
      codecThinkId: c.codec_think_id,
      codecNothinkId: c.codec_nothink_id,
      codecThinkBosId: c.codec_think_bos_id,
      codecThinkEosId: c.codec_think_eos_id,
    },
  }
}

// Load the Code Predictor (5-layer transformer)
function loadCodePredictor(parsed, config) {
  const c = config.talker_config.code_predictor_config
  const prefix = 'talker.code_predictor.model.'

  // 15 codec embeddings (for groups 1-15)
  const codecEmbeddings = []
  for (let i = 0; i < 15; i++) {
    codecEmbeddings.push(loadWeight(parsed, `${prefix}codec_embedding.${i}.weight`))
  }

  // Projection from talker hidden (2048) → predictor hidden (1024)
  const projection = loadWeight(parsed, 'talker.code_predictor.small_to_mtp_projection.weight')

  // Transformer blocks
  const blocks = []
  for (let i = 0; i < c.num_hidden_layers; i++) {
    const lp = `${prefix}layers.${i}.`
    blocks.push({
      inputLayernorm: loadWeight(parsed, `${lp}input_layernorm.weight`),
      postAttnLayernorm: loadWeight(parsed, `${lp}post_attention_layernorm.weight`),
      qProj: loadWeight(parsed, `${lp}self_attn.q_proj.weight`),
      kProj: loadWeight(parsed, `${lp}self_attn.k_proj.weight`),
      vProj: loadWeight(parsed, `${lp}self_attn.v_proj.weight`),
      oProj: loadWeight(parsed, `${lp}self_attn.o_proj.weight`),
      qNorm: loadWeight(parsed, `${lp}self_attn.q_norm.weight`),
      kNorm: loadWeight(parsed, `${lp}self_attn.k_norm.weight`),
      gateProj: loadWeight(parsed, `${lp}mlp.gate_proj.weight`),
      upProj: loadWeight(parsed, `${lp}mlp.up_proj.weight`),
      downProj: loadWeight(parsed, `${lp}mlp.down_proj.weight`),
    })
  }

  const norm = loadWeight(parsed, `${prefix}norm.weight`)

  // 15 separate lm_heads (one per code group)
  const lmHeads = []
  for (let i = 0; i < 15; i++) {
    lmHeads.push(loadWeight(parsed, `talker.code_predictor.lm_head.${i}.weight`))
  }

  return {
    codecEmbeddings,
    projection,
    projectionBias: loadWeight(parsed, 'talker.code_predictor.small_to_mtp_projection.bias'),
    blocks,
    norm,
    lmHeads,
    config: {
      dim: c.hidden_size,
      numHeads: c.num_attention_heads,
      numKVHeads: c.num_key_value_heads,
      headDim: c.head_dim,
      numLayers: c.num_hidden_layers,
      intermediateDim: c.intermediate_size,
      vocabSize: c.vocab_size,
      normEps: c.rms_norm_eps,
      ropeTheta: c.rope_theta,
    },
  }
}

// Load speech tokenizer decoder weights
function loadSpeechDecoder(parsed, config) {
  const dc = config.decoder_config
  const prefix = 'decoder.'

  // Quantizer codebooks: split RVQ (1 semantic + 15 acoustic)
  // First quantizer (semantic) — Conv1d projections with kernel=1
  const rvqFirst = {
    inputProj: tryLoad(parsed, `${prefix}quantizer.rvq_first.input_proj.weight`),
    outputProj: tryLoad(parsed, `${prefix}quantizer.rvq_first.output_proj.weight`),
    layers: [{
      clusterUsage: loadWeight(parsed, `${prefix}quantizer.rvq_first.vq.layers.0._codebook.cluster_usage`),
      embeddingSum: loadWeight(parsed, `${prefix}quantizer.rvq_first.vq.layers.0._codebook.embedding_sum`),
    }],
  }

  // Rest quantizers (acoustic, 15 layers) — shared input/output projections
  const rvqRest = {
    inputProj: tryLoad(parsed, `${prefix}quantizer.rvq_rest.input_proj.weight`),
    outputProj: tryLoad(parsed, `${prefix}quantizer.rvq_rest.output_proj.weight`),
    layers: [],
  }
  for (let i = 0; i < 15; i++) {
    rvqRest.layers.push({
      clusterUsage: loadWeight(parsed, `${prefix}quantizer.rvq_rest.vq.layers.${i}._codebook.cluster_usage`),
      embeddingSum: loadWeight(parsed, `${prefix}quantizer.rvq_rest.vq.layers.${i}._codebook.embedding_sum`),
    })
  }

  // Pre-conv: codebook_dim → latent_dim
  const preConv = {
    weight: loadWeight(parsed, `${prefix}pre_conv.conv.weight`),
    bias: loadWeight(parsed, `${prefix}pre_conv.conv.bias`),
  }

  // Pre-transformer input/output projections (latentDim ↔ hiddenDim)
  const transformerInputProj = {
    weight: loadWeight(parsed, `${prefix}pre_transformer.input_proj.weight`),
    bias: loadWeight(parsed, `${prefix}pre_transformer.input_proj.bias`),
  }
  const transformerOutputProj = {
    weight: loadWeight(parsed, `${prefix}pre_transformer.output_proj.weight`),
    bias: loadWeight(parsed, `${prefix}pre_transformer.output_proj.bias`),
  }

  // Pre-transformer (8 attention layers)
  const transformerLayers = []
  for (let i = 0; i < dc.num_hidden_layers; i++) {
    const lp = `${prefix}pre_transformer.layers.${i}.`
    transformerLayers.push({
      inputLayernorm: loadWeight(parsed, `${lp}input_layernorm.weight`),
      postAttnLayernorm: loadWeight(parsed, `${lp}post_attention_layernorm.weight`),
      qProj: loadWeight(parsed, `${lp}self_attn.q_proj.weight`),
      kProj: loadWeight(parsed, `${lp}self_attn.k_proj.weight`),
      vProj: loadWeight(parsed, `${lp}self_attn.v_proj.weight`),
      oProj: loadWeight(parsed, `${lp}self_attn.o_proj.weight`),
      gateProj: loadWeight(parsed, `${lp}mlp.gate_proj.weight`),
      upProj: loadWeight(parsed, `${lp}mlp.up_proj.weight`),
      downProj: loadWeight(parsed, `${lp}mlp.down_proj.weight`),
      layerScaleAttn: tryLoad(parsed, `${lp}self_attn_layer_scale.scale`),
      layerScaleMlp: tryLoad(parsed, `${lp}mlp_layer_scale.scale`),
    })
  }
  const transformerNorm = loadWeight(parsed, `${prefix}pre_transformer.norm.weight`)

  // Upsample blocks: [2, 2] upsampling_ratios with ConvNeXt
  const upsampleBlocks = []
  for (let i = 0; i < dc.upsampling_ratios.length; i++) {
    upsampleBlocks.push({
      transConv: {
        weight: loadWeight(parsed, `${prefix}upsample.${i}.0.conv.weight`),
        bias: tryLoad(parsed, `${prefix}upsample.${i}.0.conv.bias`),
      },
      convNeXt: {
        dwConv: {
          weight: loadWeight(parsed, `${prefix}upsample.${i}.1.dwconv.conv.weight`),
          bias: tryLoad(parsed, `${prefix}upsample.${i}.1.dwconv.conv.bias`),
        },
        norm: loadWeight(parsed, `${prefix}upsample.${i}.1.norm.weight`),
        normBias: loadWeight(parsed, `${prefix}upsample.${i}.1.norm.bias`),
        pwconv1: {
          weight: loadWeight(parsed, `${prefix}upsample.${i}.1.pwconv1.weight`),
          bias: loadWeight(parsed, `${prefix}upsample.${i}.1.pwconv1.bias`),
        },
        pwconv2: {
          weight: loadWeight(parsed, `${prefix}upsample.${i}.1.pwconv2.weight`),
          bias: loadWeight(parsed, `${prefix}upsample.${i}.1.pwconv2.bias`),
        },
        gamma: loadWeight(parsed, `${prefix}upsample.${i}.1.gamma`),
      },
    })
  }

  // Decoder: conv → [DecoderBlock × 4] → snake → conv
  const decoderConvIn = {
    weight: loadWeight(parsed, `${prefix}decoder.0.conv.weight`),
    bias: loadWeight(parsed, `${prefix}decoder.0.conv.bias`),
  }

  // 4 DecoderBlocks (one per upsample rate)
  const decoderBlocks = []
  for (let i = 0; i < dc.upsample_rates.length; i++) {
    const bp = `${prefix}decoder.${i + 1}.block.`
    const block = {
      snakeAlpha: loadWeight(parsed, `${bp}0.alpha`),
      snakeBeta: loadWeight(parsed, `${bp}0.beta`),
      transConv: {
        weight: loadWeight(parsed, `${bp}1.conv.weight`),
        bias: tryLoad(parsed, `${bp}1.conv.bias`),
      },
      residualUnits: [],
    }
    for (let j = 0; j < 3; j++) {
      const rp = `${bp}${j + 2}.`
      block.residualUnits.push({
        snake1Alpha: loadWeight(parsed, `${rp}act1.alpha`),
        snake1Beta: loadWeight(parsed, `${rp}act1.beta`),
        conv1: {
          weight: loadWeight(parsed, `${rp}conv1.conv.weight`),
          bias: loadWeight(parsed, `${rp}conv1.conv.bias`),
        },
        snake2Alpha: loadWeight(parsed, `${rp}act2.alpha`),
        snake2Beta: loadWeight(parsed, `${rp}act2.beta`),
        conv2: {
          weight: loadWeight(parsed, `${rp}conv2.conv.weight`),
          bias: loadWeight(parsed, `${rp}conv2.conv.bias`),
        },
      })
    }
    decoderBlocks.push(block)
  }

  // Final: SnakeBeta + Conv1d → 1 channel output
  const numBlocks = dc.upsample_rates.length
  const decoderFinalSnake = {
    alpha: loadWeight(parsed, `${prefix}decoder.${numBlocks + 1}.alpha`),
    beta: loadWeight(parsed, `${prefix}decoder.${numBlocks + 1}.beta`),
  }
  const decoderConvOut = {
    weight: loadWeight(parsed, `${prefix}decoder.${numBlocks + 2}.conv.weight`),
    bias: loadWeight(parsed, `${prefix}decoder.${numBlocks + 2}.conv.bias`),
  }

  return {
    rvqFirst,
    rvqRest,
    preConv,
    transformerInputProj,
    transformerOutputProj,
    transformerLayers,
    transformerNorm,
    upsampleBlocks,
    decoderConvIn,
    decoderBlocks,
    decoderFinalSnake,
    decoderConvOut,
    config: {
      latentDim: dc.latent_dim,
      decoderDim: dc.decoder_dim,
      codebookDim: dc.codebook_dim,
      codebookSize: dc.codebook_size,
      numQuantizers: dc.num_quantizers,
      numLayers: dc.num_hidden_layers,
      numHeads: dc.num_attention_heads,
      numKVHeads: dc.num_key_value_heads,
      headDim: dc.head_dim,
      hiddenDim: dc.hidden_size,
      intermediateDim: dc.intermediate_size,
      normEps: dc.rms_norm_eps,
      ropeTheta: dc.rope_theta,
      slidingWindow: dc.sliding_window,
      upsampleRates: dc.upsample_rates,
      upsamplingRatios: dc.upsampling_ratios,
      sampleRate: 24000,
    },
  }
}

// Load the complete model from a directory
async function loadModel(modelDir) {
  console.log('Loading config...')
  const config = await Bun.file(`${modelDir}/config.json`).json()
  const tokenizerConfig = await Bun.file(`${modelDir}/speech_tokenizer/config.json`).json()

  console.log('Loading main model weights...')
  const mainBuffer = await Bun.file(`${modelDir}/model.safetensors`).arrayBuffer()
  const mainParsed = parseSafetensors(mainBuffer)

  console.log(`  ${Object.keys(mainParsed.tensors).length} tensors`)

  const talker = loadTalker(mainParsed, config)
  console.log(`\n  Talker: ${talker.config.numLayers} layers, dim=${talker.config.dim}`)

  const predictor = loadCodePredictor(mainParsed, config)
  console.log(`\n  Code Predictor: ${predictor.config.numLayers} layers, dim=${predictor.config.dim}`)

  _loadCount = 0
  console.log('\nLoading speech tokenizer weights...')
  const tokBuffer = await Bun.file(`${modelDir}/speech_tokenizer/model.safetensors`).arrayBuffer()
  const tokParsed = parseSafetensors(tokBuffer)

  console.log(`  ${Object.keys(tokParsed.tensors).length} tensors`)

  const decoder = loadSpeechDecoder(tokParsed, tokenizerConfig)
  console.log(`\n  Decoder: ${decoder.config.numLayers} transformer layers, upsample ${decoder.config.upsampleRates.join('×')}`)

  return { talker, predictor, decoder, config }
}

export { loadModel, loadWeight, tryLoad }
