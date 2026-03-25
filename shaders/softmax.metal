// smith/shaders/softmax.metal
// Numerically stable softmax along the last dimension.
// Each threadgroup handles one row (one [seqLen] or [vocab] slice).
// Uses shared memory for max-reduction and sum-reduction.

#include <metal_stdlib>
using namespace metal;

struct SoftmaxParams {
  uint rows;   // product of all dims except last
  uint cols;   // last dimension size
};

// Forward: out[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
kernel void softmax_forward(
  device const float *input [[buffer(0)]],
  device float *output      [[buffer(1)]],
  constant SoftmaxParams &p [[buffer(2)]],
  uint row                  [[threadgroup_position_in_grid]],
  uint tid                  [[thread_index_in_threadgroup]],
  uint tpg                  [[threads_per_threadgroup]]
) {
  if (row >= p.rows) return;

  device const float *x = input + row * p.cols;
  device float *y = output + row * p.cols;

  threadgroup float shared[256];

  // 1. Find max across row (parallel reduction)
  float local_max = -INFINITY;
  for (uint i = tid; i < p.cols; i += tpg) {
    local_max = max(local_max, x[i]);
  }
  shared[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Compute exp(x - max) and sum
  float local_sum = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    float e = exp(x[i] - row_max);
    y[i] = e;
    local_sum += e;
  }
  shared[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = tpg / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_sum = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. Normalize
  float inv_sum = 1.0f / row_sum;
  for (uint i = tid; i < p.cols; i += tpg) {
    y[i] *= inv_sum;
  }
}

// f16 variant — half I/O, f32 reductions for numerical stability
kernel void softmax_forward_f16(
  device const half *input [[buffer(0)]],
  device half *output      [[buffer(1)]],
  constant SoftmaxParams &p [[buffer(2)]],
  uint row                  [[threadgroup_position_in_grid]],
  uint tid                  [[thread_index_in_threadgroup]],
  uint tpg                  [[threads_per_threadgroup]]
) {
  if (row >= p.rows) return;
  device const half *x = input + row * p.cols;
  device half *y = output + row * p.cols;
  threadgroup float shared[256];

  float local_max = -INFINITY;
  for (uint i = tid; i < p.cols; i += tpg) local_max = max(local_max, float(x[i]));
  shared[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) { if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
  float row_max = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float local_sum = 0.0f;
  for (uint i = tid; i < p.cols; i += tpg) {
    float e = exp(float(x[i]) - row_max);
    y[i] = half(e);
    local_sum += e;
  }
  shared[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tpg / 2; s > 0; s >>= 1) { if (tid < s) shared[tid] += shared[tid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  float inv_sum = 1.0f / shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < p.cols; i += tpg) y[i] = half(float(y[i]) * inv_sum);
}
