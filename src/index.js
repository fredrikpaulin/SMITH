// smith/src/index.js
// Public API entry point. Single flat namespace.

import * as device from './device.js'
import * as T from './tensor.js'
import * as autograd from './autograd.js'
import * as optim from './optim.js'
import * as nn from './nn.js'
import * as model from './model.js'
import { forwardCached, forwardFlash } from './model.js'
import * as tokenizer from './tokenizer.js'
import * as gen from './generate.js'
import * as ckpt from './checkpoint.js'
import { quantizeQ4, matmulQ4, matmulQ8 } from './ops/quantize.js'
import { loadGGUF, createGGUFCache, resetCache as resetGGUFCache, generateGGUF } from './gguf_loader.js'
import { parseGGUF, listTensors as listGGUFTensors, extractConfig as extractGGUFConfig } from './gguf.js'
import { poolStats, poolDrain } from './pool.js'
import { dtypeBytes, toFloat16, fromFloat16, float32ToFloat16, float16ToFloat32 } from './dtype.js'
import { f16Mode, defaultDtype, createLossScaler } from './f16mode.js'
import { cast } from './ops/cast.js'
import { transformWeights as winogradTransformWeights, canUseWinograd } from './ops/conv2d_winograd.js'
import { shouldUseIm2col } from './ops/conv2d_im2col.js'
import { poolOutputSize } from './ops/pool2d.js'
import {
  parseSafetensors, readTensor, listTensors,
  loadSafetensors, loadGPT2Safetensors,
  exportSafetensors, saveSafetensors,
  mapGPT2Weights,
} from './safetensors.js'

// Re-export tensor creation
const { tensor, zeros, ones, full, rand, randn, scalar, toArray, toString: tensorToString } = T

// Re-export autograd
const {
  variable, param, backward, zeroGrad, noGrad,
  add, sub, mul, matmul, scale, neg,
  relu, gelu, softmax, layernorm, crossEntropy,
  flashAttention,
  sum, reshape, transpose,
  embedding, addGrad,
  conv2d, maxPool2d, avgPool2d, batchnorm,
  createBatchNorm, convOutputSize,
  rope, rmsNorm, swiglu, precomputeRoPE,
} = autograd

// Re-export optimizer
const {
  createAdamW, adamwStep,
  createSchedule, getLr,
  clipGradNorm,
} = optim

// Re-export nn
const {
  createLinear, linear, linearParams,
  createCausalMask,
  createMultiHeadAttention, multiHeadAttention, multiHeadAttentionFlash, multiHeadAttentionCached,
  createTransformerBlock, transformerBlock, transformerBlockFlash, transformerBlockCached, blockParams,
  countParams,
} = nn

// Re-export model
const {
  CONFIGS, createModel, forward, modelParams, modelInfo,
} = model

const { generate, generateCached, topKPredictions } = gen
const { saveCheckpoint, loadCheckpoint } = ckpt

// Device info
function info() {
  return {
    device: device.deviceName(),
    maxThreadgroupMemory: device.maxThreadgroupMemory(),
    maxThreadsPerThreadgroup: device.maxThreadsPerThreadgroup(),
    pool: poolStats(),
  }
}

const smith = {
  // Tensor creation
  tensor, zeros, ones, full, rand, randn, scalar,
  toArray, toString: tensorToString,

  // Autograd
  variable, param, backward, zeroGrad, noGrad,

  // Ops (on variables)
  add, sub, mul, matmul, scale, neg,
  relu, gelu, softmax, layernorm, crossEntropy,
  flashAttention,
  sum, reshape, transpose,
  embedding, addGrad,

  // Neural network
  createLinear, linear, linearParams,
  createCausalMask,
  createMultiHeadAttention, multiHeadAttention, multiHeadAttentionFlash, multiHeadAttentionCached,
  createTransformerBlock, transformerBlock, transformerBlockFlash, transformerBlockCached, blockParams,
  countParams,

  // Model
  CONFIGS, createModel, forward, forwardFlash, forwardCached, modelParams, modelInfo,

  // Tokenizer
  tokenizer,

  // Generation + Checkpoint
  generate, generateCached, topKPredictions,
  saveCheckpoint, loadCheckpoint,

  // Safetensors
  parseSafetensors, readTensor, listTensors,
  loadSafetensors, loadGPT2Safetensors,
  exportSafetensors, saveSafetensors,
  mapGPT2Weights,

  // Quantization
  quantizeQ4, matmulQ4, matmulQ8,

  // GGUF
  loadGGUF, parseGGUF, listGGUFTensors, extractGGUFConfig,
  createGGUFCache, resetGGUFCache, generateGGUF,

  // Optimizer
  createAdamW, adamwStep,
  createSchedule, getLr,
  clipGradNorm,

  // Device
  info,

  // Memory
  poolStats, poolDrain,

  // Dtype utilities
  dtypeBytes, toFloat16, fromFloat16, float32ToFloat16, float16ToFloat32,

  // Mixed precision
  f16Mode, defaultDtype, cast, createLossScaler,

  // Conv / Pool / BatchNorm / Winograd
  conv2d, maxPool2d, avgPool2d, batchnorm,
  createBatchNorm, convOutputSize, poolOutputSize,
  winogradTransformWeights, canUseWinograd, shouldUseIm2col,

  // RoPE / RMSNorm / SwiGLU
  rope, rmsNorm, swiglu, precomputeRoPE,
}

export default smith
export {
  tensor, zeros, ones, full, rand, randn, scalar, toArray,
  variable, param, backward, zeroGrad, noGrad,
  add, sub, mul, matmul, scale, neg,
  relu, gelu, softmax, layernorm, crossEntropy,
  flashAttention,
  sum, reshape, transpose,
  embedding, addGrad,
  createLinear, linear, linearParams,
  createCausalMask,
  createMultiHeadAttention, multiHeadAttention, multiHeadAttentionFlash, multiHeadAttentionCached,
  createTransformerBlock, transformerBlock, transformerBlockFlash, transformerBlockCached, blockParams,
  countParams,
  CONFIGS, createModel, forward, forwardFlash, forwardCached, modelParams, modelInfo,
  tokenizer,
  generate, generateCached, topKPredictions,
  saveCheckpoint, loadCheckpoint,
  parseSafetensors, readTensor, listTensors,
  loadSafetensors, loadGPT2Safetensors,
  exportSafetensors, saveSafetensors,
  mapGPT2Weights,
  quantizeQ4, matmulQ4,
  createAdamW, adamwStep,
  createSchedule, getLr,
  clipGradNorm,
  info, poolStats, poolDrain,
  f16Mode, defaultDtype, cast, createLossScaler,
  loadGGUF, parseGGUF, listGGUFTensors, extractGGUFConfig,
  createGGUFCache, resetGGUFCache, generateGGUF,
  matmulQ8,
  conv2d, maxPool2d, avgPool2d, batchnorm,
  createBatchNorm, convOutputSize, poolOutputSize,
  winogradTransformWeights, canUseWinograd, shouldUseIm2col,
  rope, rmsNorm, swiglu, precomputeRoPE,
}
