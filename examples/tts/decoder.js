// examples/tts/decoder.js
// Qwen3-TTS Speech Tokenizer Decoder.
// Converts 16-group codec codes → 24kHz PCM audio waveform.
// Pipeline: dequantize → pre-conv → transformer → upsample → vocoder

import smith from '../../src/index.js'
import { rmsnormForward } from '../../src/ops/rmsnorm.js'
import { swigluForward } from '../../src/ops/swiglu.js'
import { precomputeRoPE } from '../../src/ops/rope.js'
import { conv1dForward } from '../../src/ops/conv1d.js'
import { convTranspose1dForward } from '../../src/ops/conv1d_transpose.js'
import { matmul2d } from '../../src/ops/matmul.js'
import {
  linearNoBias,
  addTensors,
  reshapeToHeads,
  reshapeFromHeads,
  repeatKV,
  ropePerHead,
} from './talker.js'

// Linear with bias: input [seqLen, inDim] * weight [outDim, inDim]^T + bias [outDim] → [seqLen, outDim]
function linearBiasRowwise(input, weight, bias) {
  const seqLen = input.shape[0]
  const inDim = weight.shape[1]
  const outDim = weight.shape[0]
  const out = smith.zeros([seqLen, outDim])
  for (let s = 0; s < seqLen; s++) {
    for (let o = 0; o < outDim; o++) {
      let acc = bias ? bias.data[o] : 0
      for (let i = 0; i < inDim; i++) {
        acc += input.data[s * inDim + i] * weight.data[o * inDim + i]
      }
      out.data[s * outDim + o] = acc
    }
  }
  return out
}

// SnakeBeta activation: x + sin²(αx) / β
// α and β are stored as log-space parameters (exp(alpha), exp(beta))
function snakeBeta(input, alphaParam, betaParam) {
  const [channels, length] = input.shape
  const output = smith.zeros(input.shape)
  for (let c = 0; c < channels; c++) {
    const a = Math.exp(alphaParam.data[c])
    const b = Math.exp(betaParam.data[c]) + 1e-9
    for (let t = 0; t < length; t++) {
      const idx = c * length + t
      const x = input.data[idx]
      const s = Math.sin(a * x)
      output.data[idx] = x + (s * s) / b
    }
  }
  return output
}

// Causal Conv1d: left-only padding, supports dilation.
// For dilation=1, uses GPU im2col via conv1dForward then trims right.
// For dilation>1, builds dilated im2col on CPU then uses GPU GEMM.
function causalConv1d(input, weight, bias, dilation = 1, stride = 1) {
  const [cOut, cIn, kernelSize] = weight.shape
  const [, length] = input.shape
  const effectiveKernel = (kernelSize - 1) * dilation + 1
  const targetLen = Math.floor((length - 1) / stride) + 1

  if (dilation === 1) {
    const padding = effectiveKernel - stride
    const full = conv1dForward(input, weight, bias, stride, padding).output
    const [channels, fullLen] = full.shape
    if (fullLen <= targetLen) return full
    const trimmed = smith.zeros([channels, targetLen])
    for (let c = 0; c < channels; c++) {
      trimmed.data.set(full.data.subarray(c * fullLen, c * fullLen + targetLen), c * targetLen)
    }
    return trimmed
  }

  // Dilated path: CPU im2col with dilation + GPU GEMM
  const causalPad = effectiveKernel - 1
  const outLen = length + causalPad // full output before trimming

  // Build im2col columns with dilation support
  const colRows = cIn * kernelSize
  const cols = smith.zeros([colRows, outLen])
  for (let t = 0; t < outLen; t++) {
    for (let c = 0; c < cIn; c++) {
      const rowBase = c * kernelSize
      for (let k = 0; k < kernelSize; k++) {
        const inPos = t + k * dilation - causalPad
        if (inPos >= 0 && inPos < length) {
          cols.data[(rowBase + k) * outLen + t] = input.data[c * length + inPos]
        }
      }
    }
  }

  // GEMM: [C_out, C_in*K] × [C_in*K, outLen] → [C_out, outLen]
  const wFlat = smith.zeros([cOut, colRows])
  wFlat.data.set(weight.data)
  const full = matmul2d(wFlat, cols)

  if (bias) {
    for (let oc = 0; oc < cOut; oc++) {
      const b = bias.data[oc]
      const off = oc * outLen
      for (let t = 0; t < outLen; t++) full.data[off + t] += b
    }
  }

  // Trim right for causal (keep first targetLen samples)
  if (outLen > targetLen) {
    const trimmed = smith.zeros([cOut, targetLen])
    for (let c = 0; c < cOut; c++) {
      trimmed.data.set(full.data.subarray(c * outLen, c * outLen + targetLen), c * targetLen)
    }
    return trimmed
  }
  return full
}

// Causal transposed conv1d: GPU kernel + trim right padding
function causalTransConv1d(input, weight, bias, stride) {
  const kernelSize = weight.shape[2]
  const pad = kernelSize - stride
  const out = convTranspose1dForward(input, weight, bias, stride, 0)
  if (pad > 0) {
    const [channels, outLen] = out.shape
    const trimmedLen = outLen - pad
    const trimmed = smith.zeros([channels, trimmedLen])
    for (let c = 0; c < channels; c++) {
      trimmed.data.set(
        out.data.subarray(c * outLen, c * outLen + trimmedLen),
        c * trimmedLen
      )
    }
    return trimmed
  }
  return out
}

// Linear (for pointwise convolutions in ConvNeXt): input [channels, length] treated as [length, channels]
function linearTransposed(input, weight, bias) {
  // input: [C, T] → treat as [T, C], apply linear [C, outC], get [T, outC] → [outC, T]
  const [inC, T] = input.shape
  const outC = weight.shape[0]

  const result = smith.zeros([outC, T])
  for (let t = 0; t < T; t++) {
    for (let oc = 0; oc < outC; oc++) {
      let acc = bias ? bias.data[oc] : 0
      for (let ic = 0; ic < inC; ic++) {
        acc += input.data[ic * T + t] * weight.data[oc * inC + ic]
      }
      result.data[oc * T + t] = acc
    }
  }
  return result
}

// Layer normalization (not RMS — full LayerNorm with bias, applied per-timestep)
function layernormPerTimestep(input, weight, bias, eps = 1e-6) {
  // input: [C, T] → for each t, normalize over C
  const [C, T] = input.shape
  const output = smith.zeros(input.shape)

  for (let t = 0; t < T; t++) {
    let mean = 0
    for (let c = 0; c < C; c++) mean += input.data[c * T + t]
    mean /= C

    let variance = 0
    for (let c = 0; c < C; c++) {
      const d = input.data[c * T + t] - mean
      variance += d * d
    }
    variance /= C

    const invStd = 1 / Math.sqrt(variance + eps)
    for (let c = 0; c < C; c++) {
      const normalized = (input.data[c * T + t] - mean) * invStd
      output.data[c * T + t] = normalized * weight.data[c] + bias.data[c]
    }
  }
  return output
}

// GELU activation (element-wise)
function gelu(input) {
  const output = smith.zeros(input.shape)
  const SQRT_2_PI = 0.7978845608
  const COEFF = 0.044715
  for (let i = 0; i < input.data.length; i++) {
    const x = input.data[i]
    const inner = SQRT_2_PI * (x + COEFF * x * x * x)
    output.data[i] = 0.5 * x * (1 + Math.tanh(inner))
  }
  return output
}

// ---- Dequantization ----

// Dequantize codes using the EuclideanCodebook (cluster_usage + embedding_sum)
function dequantizeCodebook(codes, clusterUsage, embeddingSum) {
  // codes: array of code IDs, length T
  // clusterUsage: [codebookSize], embeddingSum: [codebookSize, dim]
  const codebookSize = clusterUsage.shape[0]
  const dim = embeddingSum.shape[1]
  const T = codes.length

  // Compute effective embeddings: embedding_sum / cluster_usage
  const embedding = smith.zeros([codebookSize, dim])
  for (let i = 0; i < codebookSize; i++) {
    const usage = Math.max(clusterUsage.data[i], 1e-5)
    for (let d = 0; d < dim; d++) {
      embedding.data[i * dim + d] = embeddingSum.data[i * dim + d] / usage
    }
  }

  // Lookup: [T, dim]
  const quantized = smith.zeros([T, dim])
  for (let t = 0; t < T; t++) {
    const id = codes[t]
    if (id >= 0 && id < codebookSize) {
      quantized.data.set(embedding.data.subarray(id * dim, (id + 1) * dim), t * dim)
    }
  }

  return quantized // [T, dim]
}

// Apply Conv1d with kernel=1 as a linear projection: weight [outDim, inDim, 1]
// input [T, inDim] → output [T, outDim]
function conv1x1Project(input, weight) {
  const T = input.shape[0]
  const inDim = weight.shape[1]
  const outDim = weight.shape[0]
  const out = smith.zeros([T, outDim])
  for (let t = 0; t < T; t++) {
    for (let o = 0; o < outDim; o++) {
      let acc = 0
      for (let i = 0; i < inDim; i++) {
        acc += input.data[t * inDim + i] * weight.data[o * inDim + i]
      }
      out.data[t * outDim + o] = acc
    }
  }
  return out
}

// Full SplitRVQ decode: codes [T, 16] → quantized [codebookDim, T]
function dequantize(codes, decoder) {
  const T = codes.length
  const numQ = codes[0].length // 16
  const { rvqFirst, rvqRest } = decoder
  const codebookDim = decoder.config.codebookDim // 512

  // First quantizer (semantic, group 0)
  // Codebook gives [T, 256], output_proj maps [256 → 512]
  const firstCodes = codes.map(c => c[0])
  let firstQ = dequantizeCodebook(
    firstCodes,
    rvqFirst.layers[0].clusterUsage,
    rvqFirst.layers[0].embeddingSum,
  )
  // Apply output projection: Conv1d [512, 256, 1] as linear
  let quantized = rvqFirst.outputProj ? conv1x1Project(firstQ, rvqFirst.outputProj) : firstQ

  // Rest quantizers (acoustic, groups 1-15) — shared output projection
  for (let q = 0; q < 15; q++) {
    const layerCodes = codes.map(c => c[q + 1])
    let layerQ = dequantizeCodebook(
      layerCodes,
      rvqRest.layers[q].clusterUsage,
      rvqRest.layers[q].embeddingSum,
    )
    if (rvqRest.outputProj) {
      layerQ = conv1x1Project(layerQ, rvqRest.outputProj)
    }

    // Sum into quantized
    for (let i = 0; i < quantized.data.length; i++) {
      quantized.data[i] += layerQ.data[i]
    }
  }

  // Transpose: [T, codebookDim] → [codebookDim, T]
  const outDim = quantized.shape[1]
  const transposed = smith.zeros([outDim, T])
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < outDim; d++) {
      transposed.data[d * T + t] = quantized.data[t * outDim + d]
    }
  }
  return transposed
}

// ---- Transformer ----

function decoderTransformerBlock(x, block, seqLen, config, rope) {
  const { numHeads, numKVHeads, headDim, normEps } = config

  const norm1 = rmsnormForward(x, block.inputLayernorm, normEps)

  const Q = linearNoBias(norm1, block.qProj)
  const K = linearNoBias(norm1, block.kProj)
  const V = linearNoBias(norm1, block.vProj)

  // Reshape first, then RoPE per-head (Q/K are [seqLen, numHeads*headDim], tables are [maxSeq, headDim/2])
  const Qh = reshapeToHeads(Q, seqLen, numHeads, headDim)
  const Kh = reshapeToHeads(K, seqLen, numKVHeads, headDim)
  const Vh = reshapeToHeads(V, seqLen, numKVHeads, headDim)

  ropePerHead(Qh, numHeads, seqLen, headDim, rope, 0)
  ropePerHead(Kh, numKVHeads, seqLen, headDim, rope, 0)

  const fullK = repeatKV(Kh, numHeads, numKVHeads)
  const fullV = repeatKV(Vh, numHeads, numKVHeads)

  const qV = smith.variable(Qh, { requiresGrad: false })
  const kV = smith.variable(fullK, { requiresGrad: false })
  const vV = smith.variable(fullV, { requiresGrad: false })
  const attnResult = smith.flashAttention(qV, kV, vV, true)

  const attnDim = numHeads * headDim
  const attnOut = reshapeFromHeads(attnResult.data, seqLen, numHeads, headDim, attnDim)
  let projected = linearNoBias(attnOut, block.oProj)

  if (block.layerScaleAttn) {
    for (let i = 0; i < projected.data.length; i++) {
      projected.data[i] *= block.layerScaleAttn.data[i % block.layerScaleAttn.data.length]
    }
  }

  const x2 = addTensors(x, projected)

  const norm2 = rmsnormForward(x2, block.postAttnLayernorm, normEps)
  const gate = linearNoBias(norm2, block.gateProj)
  const up = linearNoBias(norm2, block.upProj)
  const fused = swigluForward(gate, up)
  let ffnOut = linearNoBias(fused, block.downProj)

  if (block.layerScaleMlp) {
    for (let i = 0; i < ffnOut.data.length; i++) {
      ffnOut.data[i] *= block.layerScaleMlp.data[i % block.layerScaleMlp.data.length]
    }
  }

  return addTensors(x2, ffnOut)
}

// Depthwise causal conv1d: weight [C, 1, K], each channel convolved independently
// Uses PyTorch cross-correlation convention: weight[k] applied to input[t + k*d - padding]
// With causal padding = (K-1)*d, this becomes: weight[k] → input[t - (K-1-k)*d]
function depthwiseCausalConv1d(input, weight, bias, dilation = 1) {
  const [channels, length] = input.shape
  const kernelSize = weight.shape[2]
  const outLen = length // causal: same length output

  const output = smith.zeros([channels, outLen])
  for (let c = 0; c < channels; c++) {
    for (let t = 0; t < outLen; t++) {
      let acc = bias ? bias.data[c] : 0
      for (let k = 0; k < kernelSize; k++) {
        const inPos = t - (kernelSize - 1 - k) * dilation
        if (inPos >= 0 && inPos < length) {
          acc += input.data[c * length + inPos] * weight.data[c * kernelSize + k]
        }
      }
      output.data[c * outLen + t] = acc
    }
  }
  return output
}

// ConvNeXt block: depthwise conv → layernorm → pw1 → gelu → pw2 → gamma scale → residual
function convNeXtBlock(input, block) {
  const residual = input
  let x = depthwiseCausalConv1d(input, block.dwConv.weight, block.dwConv.bias)

  // LayerNorm per timestep (transpose to [T, C], normalize, transpose back)
  x = layernormPerTimestep(x, block.norm, block.normBias)

  // Pointwise convolutions (implemented as linear on transposed data)
  x = linearTransposed(x, block.pwconv1.weight, block.pwconv1.bias)
  x = gelu(x)
  x = linearTransposed(x, block.pwconv2.weight, block.pwconv2.bias)

  // Gamma scale
  const [C, T] = x.shape
  for (let c = 0; c < C; c++) {
    const g = block.gamma.data[c]
    for (let t = 0; t < T; t++) {
      x.data[c * T + t] *= g
    }
  }

  return addTensors(residual, x)
}

// Residual unit: snake → conv(dilation) → snake → conv(1×1) → residual
function residualUnit(input, unit) {
  const residual = input
  let x = snakeBeta(input, unit.snake1Alpha, unit.snake1Beta)
  x = causalConv1d(x, unit.conv1.weight, unit.conv1.bias, getDilation(unit))
  x = snakeBeta(x, unit.snake2Alpha, unit.snake2Beta)
  x = causalConv1d(x, unit.conv2.weight, unit.conv2.bias)
  return addTensors(residual, x)
}

function getDilation(unit) {
  // Infer dilation from kernel shape and weight shape
  // The conv1 has dilation, conv2 is always 1×1
  return 1 // Will be set per-call
}

// Decoder block: snake → transposed conv (upsample) → 3 residual units
function decoderBlock(input, block, upsampleRate) {
  let x = snakeBeta(input, block.snakeAlpha, block.snakeBeta)
  x = causalTransConv1d(x, block.transConv.weight, block.transConv.bias, upsampleRate)

  const dilations = [1, 3, 9]
  for (let i = 0; i < 3; i++) {
    const unit = block.residualUnits[i]
    const residual = x
    x = snakeBeta(x, unit.snake1Alpha, unit.snake1Beta)
    x = causalConv1d(x, unit.conv1.weight, unit.conv1.bias, dilations[i])
    x = snakeBeta(x, unit.snake2Alpha, unit.snake2Beta)
    x = causalConv1d(x, unit.conv2.weight, unit.conv2.bias)
    x = addTensors(residual, x)
  }
  return x
}

// Prepare decoder (precompute RoPE)
function prepareDecoder(decoder) {
  decoder.rope = precomputeRoPE(decoder.config.headDim, 8000, decoder.config.ropeTheta)
  return decoder
}

// Full decode pipeline: codes [T, 16] → PCM waveform
function decode(codes, decoder) {
  const T = codes.length
  const config = decoder.config
  console.log(`  Dequantizing ${T} frames...`)

  // 1. Dequantize: codes → [codebookDim, T]
  let hidden = dequantize(codes, decoder)
  console.log(`  Dequantized: [${hidden.shape}]`)

  // 2. Pre-conv: [codebookDim, T] → [latentDim, T]
  hidden = causalConv1d(hidden, decoder.preConv.weight, decoder.preConv.bias)
  console.log(`  Pre-conv: [${hidden.shape}]`)

  // 3. Transpose for transformer: [latentDim, T] → [T, latentDim]
  const latentDim = hidden.shape[0]
  const seqLen = hidden.shape[1]
  const transposed = smith.zeros([seqLen, latentDim])
  for (let t = 0; t < seqLen; t++) {
    for (let d = 0; d < latentDim; d++) {
      transposed.data[t * latentDim + d] = hidden.data[d * seqLen + t]
    }
  }

  // 3b. Input projection: latentDim (1024) → hiddenDim (512)
  let x = linearBiasRowwise(transposed, decoder.transformerInputProj.weight, decoder.transformerInputProj.bias)
  console.log(`  Transformer input proj: [${x.shape}]`)

  // 4. Transformer (8 layers)
  const hiddenDim = x.shape[1]
  for (let i = 0; i < decoder.transformerLayers.length; i++) {
    x = decoderTransformerBlock(x, decoder.transformerLayers[i], seqLen, config, decoder.rope)
  }
  // Final norm
  x = rmsnormForward(x, decoder.transformerNorm, config.normEps)
  console.log(`  Transformer: [${x.shape}]`)

  // 4b. Output projection: hiddenDim (512) → latentDim (1024)
  x = linearBiasRowwise(x, decoder.transformerOutputProj.weight, decoder.transformerOutputProj.bias)

  // 5. Transpose back: [T, latentDim] → [latentDim, T]
  const postTrans = smith.zeros([latentDim, seqLen])
  for (let t = 0; t < seqLen; t++) {
    for (let d = 0; d < latentDim; d++) {
      postTrans.data[d * seqLen + t] = x.data[t * latentDim + d]
    }
  }
  hidden = postTrans

  // 6. Upsample blocks: [2, 2] with ConvNeXt
  for (let i = 0; i < decoder.upsampleBlocks.length; i++) {
    const ratio = config.upsamplingRatios[i]
    hidden = causalTransConv1d(hidden, decoder.upsampleBlocks[i].transConv.weight,
      decoder.upsampleBlocks[i].transConv.bias, ratio)
    hidden = convNeXtBlock(hidden, decoder.upsampleBlocks[i].convNeXt)
    console.log(`  Upsample ${ratio}×: [${hidden.shape}]`)
  }

  // 7. Decoder conv in: [latentDim, T'] → [decoderDim, T']
  hidden = causalConv1d(hidden, decoder.decoderConvIn.weight, decoder.decoderConvIn.bias)
  console.log(`  Decoder conv in: [${hidden.shape}]`)

  // 8. Decoder blocks: 4 blocks with upsample rates [8, 5, 4, 3]
  for (let i = 0; i < decoder.decoderBlocks.length; i++) {
    hidden = decoderBlock(hidden, decoder.decoderBlocks[i], config.upsampleRates[i])
    console.log(`  Decoder block ${i} (${config.upsampleRates[i]}×): [${hidden.shape}]`)
  }

  // 9. Final snake + conv → [1, samples]
  hidden = snakeBeta(hidden, decoder.decoderFinalSnake.alpha, decoder.decoderFinalSnake.beta)
  hidden = causalConv1d(hidden, decoder.decoderConvOut.weight, decoder.decoderConvOut.bias)
  console.log(`  Output: [${hidden.shape}]`)

  // 10. Clamp to [-1, 1] and extract as Float32Array
  const numSamples = hidden.shape[1]
  const pcm = new Float32Array(numSamples)
  for (let i = 0; i < numSamples; i++) {
    pcm[i] = Math.max(-1, Math.min(1, hidden.data[i]))
  }

  return pcm
}

export { prepareDecoder, decode }
