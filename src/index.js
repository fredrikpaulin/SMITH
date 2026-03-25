// smith/src/index.js
// Public API entry point. Single flat namespace.

import * as device from './device.js'
import * as T from './tensor.js'
import * as autograd from './autograd.js'
import * as optim from './optim.js'
import * as nn from './nn.js'
import * as model from './model.js'
import * as tokenizer from './tokenizer.js'
import * as gen from './generate.js'
import * as ckpt from './checkpoint.js'
import { quantizeQ4, matmulQ4 } from './ops/quantize.js'
import { poolStats, poolDrain } from './pool.js'
import { dtypeBytes, toFloat16, fromFloat16, float32ToFloat16, float16ToFloat32 } from './dtype.js'

// Re-export tensor creation
const { tensor, zeros, ones, full, rand, randn, scalar, toArray, toString: tensorToString } = T

// Re-export autograd
const {
  variable, param, backward, zeroGrad, noGrad,
  add, sub, mul, matmul, scale, neg,
  relu, gelu, softmax, layernorm, crossEntropy,
  sum, reshape, transpose,
  embedding, addGrad,
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
  createMultiHeadAttention, multiHeadAttention,
  createTransformerBlock, transformerBlock, blockParams,
  countParams,
} = nn

// Re-export model
const {
  CONFIGS, createModel, forward, modelParams, modelInfo,
} = model

const { generate, topKPredictions } = gen
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
  sum, reshape, transpose,
  embedding, addGrad,

  // Neural network
  createLinear, linear, linearParams,
  createCausalMask,
  createMultiHeadAttention, multiHeadAttention,
  createTransformerBlock, transformerBlock, blockParams,
  countParams,

  // Model
  CONFIGS, createModel, forward, modelParams, modelInfo,

  // Tokenizer
  tokenizer,

  // Generation + Checkpoint
  generate, topKPredictions,
  saveCheckpoint, loadCheckpoint,

  // Quantization
  quantizeQ4, matmulQ4,

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
}

export default smith
export {
  tensor, zeros, ones, full, rand, randn, scalar, toArray,
  variable, param, backward, zeroGrad, noGrad,
  add, sub, mul, matmul, scale, neg,
  relu, gelu, softmax, layernorm, crossEntropy,
  sum, reshape, transpose,
  embedding, addGrad,
  createLinear, linear, linearParams,
  createCausalMask,
  createMultiHeadAttention, multiHeadAttention,
  createTransformerBlock, transformerBlock, blockParams,
  countParams,
  CONFIGS, createModel, forward, modelParams, modelInfo,
  tokenizer,
  generate, topKPredictions,
  saveCheckpoint, loadCheckpoint,
  quantizeQ4, matmulQ4,
  createAdamW, adamwStep,
  createSchedule, getLr,
  clipGradNorm,
  info, poolStats, poolDrain,
}
