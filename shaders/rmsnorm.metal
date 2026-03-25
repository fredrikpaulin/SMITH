// smith/shaders/rmsnorm.metal
// RMS Normalization (Llama-style): out = x * gamma / sqrt(mean(x²) + eps)
// No mean subtraction, no beta. One threadgroup per row.

#include <metal_stdlib>
using namespace metal;

struct RMSNormParams {
  uint rows;  // product of all dims except last
  uint cols;  // last dimension (normalization axis)
  float eps;
};

// Forward: out = gamma * x / sqrt(mean(x²) + eps)
kernel void rmsnorm_forward(
  device const float *input   [[buffer(0)]],
  device const float *gamma   [[buffer(1)]],
  device float *output        [[buffer(2)]],
  constant RMSNormParams &p   [[buffer(3)]],
  uint row                    [[threadgroup_position_in_grid]],
  uint tid                    [[thread_index_in_threadgroup]],
  uint tpg                    [[threads_per_threadgroup]]
) {
  if (row >= p.rows) return;

  device const float *x = input + row * p.cols;
  device float *y = output + row * p.cols;

  threadgroup float shared[256];

  // 1. Compute sum of squares
  float local_ss = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    float v = x[i];
    local_ss += v * v;
  }
  shared[tid] = local_ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float rms = rsqrt(shared[0] / float(p.cols) + p.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Normalize and scale
  for (uint i = tid; i < p.cols; i += tpg) {
    y[i] = x[i] * rms * gamma[i];
  }
}

// f16 variant — half I/O, f32 accumulator for sum of squares
kernel void rmsnorm_forward_f16(
  device const half *input    [[buffer(0)]],
  device const half *gamma    [[buffer(1)]],
  device half *output         [[buffer(2)]],
  constant RMSNormParams &p   [[buffer(3)]],
  uint row                    [[threadgroup_position_in_grid]],
  uint tid                    [[thread_index_in_threadgroup]],
  uint tpg                    [[threads_per_threadgroup]]
) {
  if (row >= p.rows) return;

  device const half *x = input + row * p.cols;
  device half *y = output + row * p.cols;

  threadgroup float shared[256];

  float local_ss = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    float v = float(x[i]);
    local_ss += v * v;
  }
  shared[tid] = local_ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float rms = rsqrt(shared[0] / float(p.cols) + p.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < p.cols; i += tpg) {
    y[i] = half(float(x[i]) * rms * float(gamma[i]));
  }
}

// Backward: given gradOutput and saved input/rms
// gradInput[i] = gamma[i] * rms * (gradOut[i] - x[i] * rms² * dot(gradOut * x * gamma) / cols)
// gradGamma[i] = sum_rows(gradOut[i] * x[i] * rms)  — accumulated on CPU
kernel void rmsnorm_backward(
  device const float *gradOut [[buffer(0)]],
  device const float *input   [[buffer(1)]],
  device const float *gamma   [[buffer(2)]],
  device float *gradIn        [[buffer(3)]],
  constant RMSNormParams &p   [[buffer(4)]],
  uint row                    [[threadgroup_position_in_grid]],
  uint tid                    [[thread_index_in_threadgroup]],
  uint tpg                    [[threads_per_threadgroup]]
) {
  if (row >= p.rows) return;

  device const float *x = input + row * p.cols;
  device const float *go = gradOut + row * p.cols;
  device float *gi = gradIn + row * p.cols;

  threadgroup float shared[256];

  // 1. Compute RMS (same as forward)
  float local_ss = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    float v = x[i];
    local_ss += v * v;
  }
  shared[tid] = local_ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float meanSq = shared[0] / float(p.cols);
  float rms = rsqrt(meanSq + p.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Compute dot(gradOut * gamma * x) for this row
  float local_dot = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    local_dot += go[i] * gamma[i] * x[i];
  }
  shared[tid] = local_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float dotVal = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. Compute gradInput
  float rms3 = rms * rms * rms;
  for (uint i = tid; i < p.cols; i += tpg) {
    gi[i] = gamma[i] * rms * go[i] - x[i] * rms3 * dotVal / float(p.cols);
  }
}

kernel void rmsnorm_backward_f16(
  device const half *gradOut  [[buffer(0)]],
  device const half *input    [[buffer(1)]],
  device const half *gamma    [[buffer(2)]],
  device half *gradIn         [[buffer(3)]],
  constant RMSNormParams &p   [[buffer(4)]],
  uint row                    [[threadgroup_position_in_grid]],
  uint tid                    [[thread_index_in_threadgroup]],
  uint tpg                    [[threads_per_threadgroup]]
) {
  if (row >= p.rows) return;

  device const half *x = input + row * p.cols;
  device const half *go = gradOut + row * p.cols;
  device half *gi = gradIn + row * p.cols;

  threadgroup float shared[256];

  float local_ss = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    float v = float(x[i]);
    local_ss += v * v;
  }
  shared[tid] = local_ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float meanSq = shared[0] / float(p.cols);
  float rms = rsqrt(meanSq + p.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float local_dot = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    local_dot += float(go[i]) * float(gamma[i]) * float(x[i]);
  }
  shared[tid] = local_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float dotVal = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float rms3 = rms * rms * rms;
  for (uint i = tid; i < p.cols; i += tpg) {
    gi[i] = half(float(gamma[i]) * rms * float(go[i]) - float(x[i]) * rms3 * dotVal / float(p.cols));
  }
}
