// smith/src/ops/batchnorm.js
// Batch normalization for NCHW layout.
// Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
// Tracks running mean/var for inference mode.

import * as T from '../tensor.js'
import { run } from '../dispatch.js'

function batchnormParams(batch, channels, spatial, eps, momentum) {
  const buf = new ArrayBuffer(20)
  const u = new Uint32Array(buf, 0, 3)
  const f = new Float32Array(buf, 12, 2)
  u[0] = batch
  u[1] = channels
  u[2] = spatial
  f[0] = eps
  f[1] = momentum
  return new Uint8Array(buf)
}

// Create batchnorm layer state
function createBatchNorm(channels, opts = {}) {
  const { eps = 1e-5, momentum = 0.1 } = opts
  return {
    gamma: T.ones([channels]),
    beta: T.zeros([channels]),
    runningMean: T.zeros([channels]),
    runningVar: T.ones([channels]),
    channels,
    eps,
    momentum,
  }
}

// Forward (training mode) — computes batch stats, updates running stats
function batchnormForward(input, layer) {
  const [batch, channels, H, W] = input.shape
  const spatial = H * W
  const out = T.create(input.shape, input.dtype)
  const savedMean = T.create([channels], input.dtype)
  const savedInvStd = T.create([channels], input.dtype)

  const params = batchnormParams(batch, channels, spatial, layer.eps, layer.momentum)

  run('batchnorm_forward', [
    { buffer: input.buffer, index: 0 },
    { buffer: layer.gamma.buffer, index: 1 },
    { buffer: layer.beta.buffer, index: 2 },
    { buffer: out.buffer, index: 3 },
    { buffer: savedMean.buffer, index: 4 },
    { buffer: savedInvStd.buffer, index: 5 },
    { buffer: layer.runningMean.buffer, index: 6 },
    { buffer: layer.runningVar.buffer, index: 7 },
  ], { x: channels }, { x: Math.min(channels, 256) },
  { data: params, index: 8 })

  return { out, savedMean, savedInvStd }
}

// Forward (inference mode) — uses running stats
function batchnormInference(input, layer) {
  const [batch, channels, H, W] = input.shape
  const spatial = H * W
  const out = T.create(input.shape, input.dtype)

  const params = batchnormParams(batch, channels, spatial, layer.eps, layer.momentum)

  run('batchnorm_forward_inference', [
    { buffer: input.buffer, index: 0 },
    { buffer: layer.gamma.buffer, index: 1 },
    { buffer: layer.beta.buffer, index: 2 },
    { buffer: out.buffer, index: 3 },
    { buffer: layer.runningMean.buffer, index: 4 },
    { buffer: layer.runningVar.buffer, index: 5 },
  ], { x: channels }, { x: Math.min(channels, 256) },
  { data: params, index: 6 })

  return out
}

// Backward — computes gradInput, gradGamma, gradBeta
function batchnormBackward(gradOutput, input, savedMean, savedInvStd, layer) {
  const [batch, channels, H, W] = input.shape
  const spatial = H * W

  const gradInput = T.create(input.shape, input.dtype)
  const gradGamma = T.zeros([channels], input.dtype)
  const gradBeta = T.zeros([channels], input.dtype)

  const params = batchnormParams(batch, channels, spatial, layer.eps, layer.momentum)

  run('batchnorm_backward', [
    { buffer: gradOutput.buffer, index: 0 },
    { buffer: input.buffer, index: 1 },
    { buffer: savedMean.buffer, index: 2 },
    { buffer: savedInvStd.buffer, index: 3 },
    { buffer: layer.gamma.buffer, index: 4 },
    { buffer: gradInput.buffer, index: 5 },
    { buffer: gradGamma.buffer, index: 6 },
    { buffer: gradBeta.buffer, index: 7 },
  ], { x: channels }, { x: Math.min(channels, 256) },
  { data: params, index: 8 })

  return { gradInput, gradGamma, gradBeta }
}

export {
  createBatchNorm, batchnormForward, batchnormInference, batchnormBackward,
  batchnormParams,
}
