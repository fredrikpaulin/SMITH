// smith/shaders/reduce.metal
// Parallel reduction operations: sum, max, min.
// Two patterns: full reduction (to scalar) and axis reduction (along one dimension).

#include <metal_stdlib>
using namespace metal;

// --- Full sum reduction ---
// Uses threadgroup shared memory for parallel reduction.
// Call with grid = (size), group = (threadgroup_size).
// Output buffer should have (num_threadgroups) elements.
// For large inputs, run a second pass to reduce the partial sums.

kernel void reduce_sum(
    device const float* input   [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& size          [[buffer(2)]],
    threadgroup float* shared    [[threadgroup(0)]],
    uint tid                     [[thread_position_in_grid]],
    uint lid                     [[thread_index_in_threadgroup]],
    uint group_id                [[threadgroup_position_in_grid]],
    uint group_size              [[threads_per_threadgroup]])
{
    // Each thread loads one element (or 0 if out of bounds)
    float val = tid < size ? input[tid] : 0.0f;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[group_id] = shared[0];
    }
}

kernel void reduce_max(
    device const float* input   [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& size          [[buffer(2)]],
    threadgroup float* shared    [[threadgroup(0)]],
    uint tid                     [[thread_position_in_grid]],
    uint lid                     [[thread_index_in_threadgroup]],
    uint group_id                [[threadgroup_position_in_grid]],
    uint group_size              [[threads_per_threadgroup]])
{
    float val = tid < size ? input[tid] : -INFINITY;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[group_id] = shared[0];
    }
}

// --- Axis reduction ---
// Reduce along a specific axis of an N-dimensional tensor.
// The input is treated as [outer, axis_size, inner] where:
//   outer = product of dims before axis
//   inner = product of dims after axis
// Grid: (outer * inner), one thread per output element.

struct AxisReduceParams {
    uint outer;
    uint axis_size;
    uint inner;
};

kernel void reduce_sum_axis(
    device const float* input       [[buffer(0)]],
    device float* output             [[buffer(1)]],
    constant AxisReduceParams& p     [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]])
{
    uint out_size = p.outer * p.inner;
    if (tid >= out_size) return;

    uint outer_idx = tid / p.inner;
    uint inner_idx = tid % p.inner;

    float sum = 0.0f;
    uint base = outer_idx * p.axis_size * p.inner + inner_idx;
    for (uint i = 0; i < p.axis_size; i++) {
        sum += input[base + i * p.inner];
    }
    output[tid] = sum;
}

kernel void reduce_max_axis(
    device const float* input       [[buffer(0)]],
    device float* output             [[buffer(1)]],
    constant AxisReduceParams& p     [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]])
{
    uint out_size = p.outer * p.inner;
    if (tid >= out_size) return;

    uint outer_idx = tid / p.inner;
    uint inner_idx = tid % p.inner;

    float m = -INFINITY;
    uint base = outer_idx * p.axis_size * p.inner + inner_idx;
    for (uint i = 0; i < p.axis_size; i++) {
        m = max(m, input[base + i * p.inner]);
    }
    output[tid] = m;
}
