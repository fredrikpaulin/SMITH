// smith/shaders/batchnorm.metal
// Batch normalization for NCHW layout.
// Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
// Backward: gradients w.r.t. input, gamma, beta

#include <metal_stdlib>
using namespace metal;

struct BatchNormParams {
  uint batch;
  uint channels;
  uint spatial;   // H * W
  float eps;
  float momentum; // for running stats update
};

// --- Forward: compute mean and variance per channel, then normalize ---
// Two-pass: pass 1 computes mean+var, pass 2 normalizes
// For simplicity, single kernel with internal loop

kernel void batchnorm_forward(
  device const float *input      [[buffer(0)]],
  device const float *gamma      [[buffer(1)]],
  device const float *beta       [[buffer(2)]],
  device float *output           [[buffer(3)]],
  device float *savedMean        [[buffer(4)]],
  device float *savedInvStd      [[buffer(5)]],
  device float *runningMean      [[buffer(6)]],
  device float *runningVar       [[buffer(7)]],
  constant BatchNormParams &p    [[buffer(8)]],
  uint gid                       [[thread_position_in_grid]]
) {
  uint c = gid;
  if (c >= p.channels) return;

  uint count = p.batch * p.spatial;

  // Compute mean
  float mean = 0.0f;
  for (uint n = 0; n < p.batch; n++) {
    for (uint s = 0; s < p.spatial; s++) {
      uint idx = n * p.channels * p.spatial + c * p.spatial + s;
      mean += input[idx];
    }
  }
  mean /= float(count);

  // Compute variance
  float var = 0.0f;
  for (uint n = 0; n < p.batch; n++) {
    for (uint s = 0; s < p.spatial; s++) {
      uint idx = n * p.channels * p.spatial + c * p.spatial + s;
      float diff = input[idx] - mean;
      var += diff * diff;
    }
  }
  var /= float(count);

  float invStd = 1.0f / sqrt(var + p.eps);

  // Save for backward
  savedMean[c] = mean;
  savedInvStd[c] = invStd;

  // Update running stats
  runningMean[c] = (1.0f - p.momentum) * runningMean[c] + p.momentum * mean;
  runningVar[c]  = (1.0f - p.momentum) * runningVar[c]  + p.momentum * var;

  // Normalize
  float g = gamma[c];
  float b = beta[c];
  for (uint n = 0; n < p.batch; n++) {
    for (uint s = 0; s < p.spatial; s++) {
      uint idx = n * p.channels * p.spatial + c * p.spatial + s;
      output[idx] = g * (input[idx] - mean) * invStd + b;
    }
  }
}

// --- Forward inference (use running stats, no saved stats) ---

kernel void batchnorm_forward_inference(
  device const float *input      [[buffer(0)]],
  device const float *gamma      [[buffer(1)]],
  device const float *beta       [[buffer(2)]],
  device float *output           [[buffer(3)]],
  device const float *runningMean [[buffer(4)]],
  device const float *runningVar  [[buffer(5)]],
  constant BatchNormParams &p    [[buffer(6)]],
  uint gid                       [[thread_position_in_grid]]
) {
  uint c = gid;
  if (c >= p.channels) return;

  float mean = runningMean[c];
  float invStd = 1.0f / sqrt(runningVar[c] + p.eps);
  float g = gamma[c];
  float b = beta[c];

  for (uint n = 0; n < p.batch; n++) {
    for (uint s = 0; s < p.spatial; s++) {
      uint idx = n * p.channels * p.spatial + c * p.spatial + s;
      output[idx] = g * (input[idx] - mean) * invStd + b;
    }
  }
}

// --- Backward ---
// Computes gradInput, gradGamma, gradBeta

kernel void batchnorm_backward(
  device const float *gradOutput  [[buffer(0)]],
  device const float *input       [[buffer(1)]],
  device const float *savedMean   [[buffer(2)]],
  device const float *savedInvStd [[buffer(3)]],
  device const float *gamma       [[buffer(4)]],
  device float *gradInput         [[buffer(5)]],
  device float *gradGamma         [[buffer(6)]],
  device float *gradBeta          [[buffer(7)]],
  constant BatchNormParams &p     [[buffer(8)]],
  uint gid                        [[thread_position_in_grid]]
) {
  uint c = gid;
  if (c >= p.channels) return;

  uint count = p.batch * p.spatial;
  float mean = savedMean[c];
  float invStd = savedInvStd[c];
  float g = gamma[c];

  // Accumulate gradGamma and gradBeta
  float dGamma = 0.0f;
  float dBeta = 0.0f;
  for (uint n = 0; n < p.batch; n++) {
    for (uint s = 0; s < p.spatial; s++) {
      uint idx = n * p.channels * p.spatial + c * p.spatial + s;
      float xhat = (input[idx] - mean) * invStd;
      dGamma += gradOutput[idx] * xhat;
      dBeta += gradOutput[idx];
    }
  }
  gradGamma[c] = dGamma;
  gradBeta[c] = dBeta;

  // Compute gradInput
  // dX = invStd * gamma * (dY - mean(dY) - xhat * mean(dY * xhat)) / 1
  // Simplified: dX[i] = gamma * invStd * (dY[i] - dBeta/N - xhat[i] * dGamma/N)
  float invN = 1.0f / float(count);
  for (uint n = 0; n < p.batch; n++) {
    for (uint s = 0; s < p.spatial; s++) {
      uint idx = n * p.channels * p.spatial + c * p.spatial + s;
      float xhat = (input[idx] - mean) * invStd;
      gradInput[idx] = g * invStd * (gradOutput[idx] - dBeta * invN - xhat * dGamma * invN);
    }
  }
}
