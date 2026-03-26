// smith/src/index.js
// Public API entry point. Single flat namespace.

import * as device from './device.js'
import * as T from './tensor.js'
import * as autograd from './autograd.js'
import * as optim from './optim.js'
import { createMuonAdamW, muonAdamWStep } from './muon.js'
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
import { gpuFFT, gpuIFFT, gpuBatchFFT } from './ops/fft.js'
import { gpuArgmax, gpuSample } from './ops/sampling.js'
import { gather as gpuGatherOp, scatterAdd as gpuScatterAdd, scatter as gpuScatterOp } from './ops/gather.js'
import { tanh as gpuTanhOp } from './ops/tanh.js'
import { sigmoid as gpuSigmoidOp } from './ops/sigmoid.js'
import { reluSquared as gpuReluSquaredOp } from './ops/relusquared.js'
import { gpuMelSpectrogram } from './ops/mel.js'
import { dtypeBytes, toFloat16, fromFloat16, float32ToFloat16, float16ToFloat32 } from './dtype.js'
import { f16Mode, defaultDtype, createLossScaler } from './f16mode.js'
import { cast } from './ops/cast.js'
import { transformWeights as winogradTransformWeights, canUseWinograd } from './ops/conv2d_winograd.js'
import { shouldUseIm2col } from './ops/conv2d_im2col.js'
import { poolOutputSize } from './ops/pool2d.js'
import {
  createResNet, forwardResNet, resnetParams,
  mapResNetWeights, loadResNet, RESNET_CONFIGS,
} from './resnet.js'
import {
  createCLIP, forwardVision, forwardText,
  clipSimilarity, l2Normalize, clipParams,
  mapCLIPWeights, loadCLIP, CLIP_CONFIGS,
} from './clip.js'
import {
  resizeBilinear, centerCrop, normalize as normalizeImage,
  rgbaToChw, rgbToChw,
  preprocessResNet, preprocessCLIP,
  loadPPM, IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD,
} from './vision.js'
import {
  parseSafetensors, readTensor, listTensors,
  loadSafetensors, loadGPT2Safetensors,
  exportSafetensors, saveSafetensors,
  mapGPT2Weights,
} from './safetensors.js'
import {
  enableProfiling, disableProfiling, isProfilingEnabled,
  report as profileReport, resetProfile,
  profile, benchmark,
  memorySnapshot,
} from './profile.js'
import {
  listModels, getModel, modelPath, modelPaths,
  fetchModel, fetchUrl, registerModel, removeModel,
  modelsDir, reloadRegistry, hashFile,
} from './models.js'
import {
  dispose, retain, isDisposed,
  using, usingAsync, withNoAlloc,
  activeScopeDepth,
} from './lifecycle.js'

// Re-export tensor creation
const { tensor, zeros, ones, full, rand, randn, scalar, toArray, toString: tensorToString } = T

// Re-export autograd
const {
  variable, param, backward, zeroGrad, noGrad,
  add, sub, mul, div, matmul, scale, neg,
  relu, gelu, tanh, sigmoid, reluSquared,
  softmax, layernorm, crossEntropy,
  flashAttention,
  sum, reshape, transpose,
  embedding, addGrad,
  gather, scatter,
  fft,
  conv1d, conv1dOutputSize,
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
  multiHeadCrossAttention, multiHeadCrossAttentionCached,
  createTransformerBlock, transformerBlock, transformerBlockFlash, transformerBlockCached, blockParams,
  countParams,
  sinusoidalPE,
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
  add, sub, mul, div, matmul, scale, neg,
  relu, gelu, tanh, sigmoid, reluSquared,
  softmax, layernorm, crossEntropy,
  flashAttention,
  sum, reshape, transpose,
  embedding, addGrad,
  gather, scatter,

  // Gather/Scatter (raw GPU ops)
  gpuGatherOp, gpuScatterAdd, gpuScatterOp,

  // Neural network
  createLinear, linear, linearParams,
  createCausalMask,
  createMultiHeadAttention, multiHeadAttention, multiHeadAttentionFlash, multiHeadAttentionCached,
  multiHeadCrossAttention, multiHeadCrossAttentionCached,
  createTransformerBlock, transformerBlock, transformerBlockFlash, transformerBlockCached, blockParams,
  countParams,
  sinusoidalPE,

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
  createMuonAdamW, muonAdamWStep,
  createSchedule, getLr,
  clipGradNorm,

  // Device
  info,

  // Memory
  poolStats, poolDrain,

  // Lifecycle
  dispose, retain, isDisposed,
  using, usingAsync, withNoAlloc,
  activeScopeDepth,

  // Dtype utilities
  dtypeBytes, toFloat16, fromFloat16, float32ToFloat16, float16ToFloat32,

  // Mixed precision
  f16Mode, defaultDtype, cast, createLossScaler,

  // Conv / Pool / BatchNorm / Winograd
  conv1d, conv1dOutputSize,
  conv2d, maxPool2d, avgPool2d, batchnorm,
  createBatchNorm, convOutputSize, poolOutputSize,
  winogradTransformWeights, canUseWinograd, shouldUseIm2col,

  // FFT / Mel spectrogram
  fft, gpuFFT, gpuIFFT, gpuBatchFFT,
  gpuMelSpectrogram,

  // GPU sampling
  gpuArgmax, gpuSample,

  // RoPE / RMSNorm / SwiGLU
  rope, rmsNorm, swiglu, precomputeRoPE,

  // Vision models
  createResNet, forwardResNet, resnetParams, mapResNetWeights, loadResNet, RESNET_CONFIGS,
  createCLIP, forwardVision, forwardText, clipSimilarity, l2Normalize, clipParams, mapCLIPWeights, loadCLIP, CLIP_CONFIGS,

  // Image preprocessing
  resizeBilinear, centerCrop, normalizeImage,
  rgbaToChw, rgbToChw, preprocessResNet, preprocessCLIP,
  loadPPM, IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD,

  // Profiling
  enableProfiling, disableProfiling, isProfilingEnabled,
  profileReport, resetProfile,
  profile, benchmark,
  memorySnapshot,

  // Models
  listModels, getModel, modelPath, modelPaths,
  fetchModel, fetchUrl, registerModel, removeModel,
  modelsDir, reloadRegistry, hashFile,
}

export default smith
export {
  tensor, zeros, ones, full, rand, randn, scalar, toArray,
  variable, param, backward, zeroGrad, noGrad,
  add, sub, mul, div, matmul, scale, neg,
  relu, gelu, tanh, sigmoid, reluSquared,
  softmax, layernorm, crossEntropy,
  flashAttention,
  sum, reshape, transpose,
  embedding, addGrad,
  gather, scatter,
  gpuGatherOp, gpuScatterAdd, gpuScatterOp,
  createLinear, linear, linearParams,
  createCausalMask,
  createMultiHeadAttention, multiHeadAttention, multiHeadAttentionFlash, multiHeadAttentionCached,
  multiHeadCrossAttention, multiHeadCrossAttentionCached,
  createTransformerBlock, transformerBlock, transformerBlockFlash, transformerBlockCached, blockParams,
  countParams,
  sinusoidalPE,
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
  createMuonAdamW, muonAdamWStep,
  createSchedule, getLr,
  clipGradNorm,
  info, poolStats, poolDrain,
  dispose, retain, isDisposed,
  using, usingAsync, withNoAlloc,
  activeScopeDepth,
  f16Mode, defaultDtype, cast, createLossScaler,
  loadGGUF, parseGGUF, listGGUFTensors, extractGGUFConfig,
  createGGUFCache, resetGGUFCache, generateGGUF,
  matmulQ8,
  conv1d, conv1dOutputSize,
  conv2d, maxPool2d, avgPool2d, batchnorm,
  createBatchNorm, convOutputSize, poolOutputSize,
  winogradTransformWeights, canUseWinograd, shouldUseIm2col,
  rope, rmsNorm, swiglu, precomputeRoPE,
  fft, gpuFFT, gpuIFFT, gpuBatchFFT,
  gpuMelSpectrogram,
  gpuArgmax, gpuSample,
  createResNet, forwardResNet, resnetParams, mapResNetWeights, loadResNet, RESNET_CONFIGS,
  createCLIP, forwardVision, forwardText, clipSimilarity, l2Normalize, clipParams, mapCLIPWeights, loadCLIP, CLIP_CONFIGS,
  resizeBilinear, centerCrop, normalizeImage,
  rgbaToChw, rgbToChw, preprocessResNet, preprocessCLIP,
  loadPPM, IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD,
  enableProfiling, disableProfiling, isProfilingEnabled,
  profileReport, resetProfile,
  profile, benchmark,
  memorySnapshot,
  listModels, getModel, modelPath, modelPaths,
  fetchModel, fetchUrl, registerModel, removeModel,
  modelsDir, reloadRegistry, hashFile,
}
