// smith/shaders/layernorm.metal
// Layer normalization: normalize over the last dimension, scale by gamma + beta.
// Each threadgroup handles one row.
// Stores normalized values (xhat) for use in backward pass.

#include <metal_stdlib>
using namespace metal;

struct LayerNormParams {
  uint rows;  // product of all dims except last
  uint cols;  // last dimension (normalization axis)
  float eps;
};

// Forward: out = gamma * (x - mean) / sqrt(var + eps) + beta
// Also writes xhat (normalized, pre-scale) for backward.
kernel void layernorm_forward(
  device const float *input  [[buffer(0)]],
  device const float *gamma  [[buffer(1)]],
  device const float *beta   [[buffer(2)]],
  device float *output       [[buffer(3)]],
  device float *xhat_out     [[buffer(4)]],
  constant LayerNormParams &p [[buffer(5)]],
  uint row                   [[threadgroup_position_in_grid]],
  uint tid                   [[thread_index_in_threadgroup]],
  uint tpg                   [[threads_per_threadgroup]]
) {
  if (row >= p.rows) return;

  device const float *x = input + row * p.cols;
  device float *y = output + row * p.cols;
  device float *xh = xhat_out + row * p.cols;

  threadgroup float shared[256];

  // 1. Compute mean
  float local_sum = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    local_sum += x[i];
  }
  shared[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = shared[0] / float(p.cols);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Compute variance
  float local_var = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    float d = x[i] - mean;
    local_var += d * d;
  }
  shared[tid] = local_var;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float variance = shared[0] / float(p.cols);
  float inv_std = rsqrt(variance + p.eps);

  // 3. Normalize, scale, shift
  for (uint i = tid; i < p.cols; i += tpg) {
    float xhat_val = (x[i] - mean) * inv_std;
    xh[i] = xhat_val;
    y[i] = gamma[i] * xhat_val + beta[i];
  }
}

// Backward: compute dL/dx, dL/dgamma, dL/dbeta
// dL/dgamma = sum(grad * xhat, over rows)
// dL/dbeta = sum(grad, over rows)
// dL/dx = (1/std) * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
//   where dxhat = grad * gamma
kernel void layernorm_backward(
  device const float *grad_out  [[buffer(0)]],
  device const float *xhat      [[buffer(1)]],
  device const float *gamma      [[buffer(2)]],
  device const float *input      [[buffer(3)]],
  device float *grad_input       [[buffer(4)]],
  device float *grad_gamma       [[buffer(5)]],
  device float *grad_beta        [[buffer(6)]],
  constant LayerNormParams &p    [[buffer(7)]],
  uint row                       [[threadgroup_position_in_grid]],
  uint tid                       [[thread_index_in_threadgroup]],
  uint tpg                       [[threads_per_threadgroup]]
) {
  if (row >= p.rows) return;

  device const float *dy = grad_out + row * p.cols;
  device const float *xh = xhat + row * p.cols;
  device float *dx = grad_input + row * p.cols;

  threadgroup float shared[256];
  threadgroup float shared2[256];

  float D = float(p.cols);

  // Accumulate grad_gamma and grad_beta (atomically across rows)
  // For simplicity, we do this per-row and rely on the JS side to handle multi-row accumulation
  // Actually, we'll let JS accumulate across rows. Per-row backward just computes dx.

  // Compute dxhat = grad * gamma
  // Need mean(dxhat) and mean(dxhat * xhat) for dx

  // 1. mean(dxhat) and mean(dxhat * xhat)
  float local_dxhat_sum = 0.0f;
  float local_dxhat_xhat_sum = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    float dxh = dy[i] * gamma[i];
    local_dxhat_sum += dxh;
    local_dxhat_xhat_sum += dxh * xh[i];
  }
  shared[tid] = local_dxhat_sum;
  shared2[tid] = local_dxhat_xhat_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
      shared2[tid] += shared2[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float dxhat_mean = shared[0] / D;
  float dxhat_xhat_mean = shared2[0] / D;

  // 2. Recompute inv_std from input
  float local_sum = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    local_sum += input[i + row * p.cols];
  }
  shared[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = shared[0] / D;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float local_var = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    float d = input[i + row * p.cols] - mean;
    local_var += d * d;
  }
  shared[tid] = local_var;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_std = rsqrt(shared[0] / D + p.eps);

  // 3. dx = inv_std * (dxhat - dxhat_mean - xhat * dxhat_xhat_mean)
  for (uint i = tid; i < p.cols; i += tpg) {
    float dxh = dy[i] * gamma[i];
    dx[i] = inv_std * (dxh - dxhat_mean - xh[i] * dxhat_xhat_mean);
  }
}
