// smith/shaders/elementwise.metal
// Element-wise operations: add, sub, mul, div, scale, fill.
// Each kernel operates on flat buffers with a 1D grid of threads.

#include <metal_stdlib>
using namespace metal;

// --- Binary ops ---

kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint tid               [[thread_position_in_grid]])
{
    out[tid] = a[tid] + b[tid];
}

kernel void elementwise_sub(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint tid               [[thread_position_in_grid]])
{
    out[tid] = a[tid] - b[tid];
}

kernel void elementwise_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint tid               [[thread_position_in_grid]])
{
    out[tid] = a[tid] * b[tid];
}

kernel void elementwise_div(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint tid               [[thread_position_in_grid]])
{
    out[tid] = a[tid] / b[tid];
}

// --- Unary / scalar ops ---

// out = a * scalar
struct ScaleParams {
    float value;
};

kernel void elementwise_scale(
    device const float* a   [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant ScaleParams& p  [[buffer(2)]],
    uint tid                 [[thread_position_in_grid]])
{
    out[tid] = a[tid] * p.value;
}

// out = scalar (fill)
kernel void elementwise_fill(
    device float* out        [[buffer(0)]],
    constant ScaleParams& p  [[buffer(1)]],
    uint tid                 [[thread_position_in_grid]])
{
    out[tid] = p.value;
}

// out = -a
kernel void elementwise_neg(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint tid               [[thread_position_in_grid]])
{
    out[tid] = -a[tid];
}

// --- Fused: add + relu (saves one kernel launch for FFN residual + activation) ---

kernel void elementwise_add_relu(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint tid               [[thread_position_in_grid]])
{
    float v = a[tid] + b[tid];
    out[tid] = v > 0.0f ? v : 0.0f;
}

// --- Broadcasting support ---
// For broadcasted ops, we pass shape/stride info as params and use
// a flattened index to compute source indices.

struct BroadcastParams {
    uint ndim;
    uint out_shape[8];   // max 8 dimensions
    uint a_strides[8];
    uint b_strides[8];
    uint out_size;
};

// Convert flat index to strided offset for a broadcast source
static uint broadcast_offset(uint flat, constant uint* out_shape, constant uint* src_strides, uint ndim) {
    uint offset = 0;
    uint remaining = flat;
    for (uint d = 0; d < ndim; d++) {
        uint dim_stride = 1;
        for (uint k = d + 1; k < ndim; k++) dim_stride *= out_shape[k];
        uint idx = remaining / dim_stride;
        remaining %= dim_stride;
        offset += idx * src_strides[d];
    }
    return offset;
}

kernel void broadcast_add(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant BroadcastParams& p  [[buffer(3)]],
    uint tid                     [[thread_position_in_grid]])
{
    if (tid >= p.out_size) return;
    uint ai = broadcast_offset(tid, p.out_shape, p.a_strides, p.ndim);
    uint bi = broadcast_offset(tid, p.out_shape, p.b_strides, p.ndim);
    out[tid] = a[ai] + b[bi];
}

kernel void broadcast_mul(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant BroadcastParams& p  [[buffer(3)]],
    uint tid                     [[thread_position_in_grid]])
{
    if (tid >= p.out_size) return;
    uint ai = broadcast_offset(tid, p.out_shape, p.a_strides, p.ndim);
    uint bi = broadcast_offset(tid, p.out_shape, p.b_strides, p.ndim);
    out[tid] = a[ai] * b[bi];
}

kernel void broadcast_sub(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant BroadcastParams& p  [[buffer(3)]],
    uint tid                     [[thread_position_in_grid]])
{
    if (tid >= p.out_size) return;
    uint ai = broadcast_offset(tid, p.out_shape, p.a_strides, p.ndim);
    uint bi = broadcast_offset(tid, p.out_shape, p.b_strides, p.ndim);
    out[tid] = a[ai] - b[bi];
}

kernel void broadcast_div(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant BroadcastParams& p  [[buffer(3)]],
    uint tid                     [[thread_position_in_grid]])
{
    if (tid >= p.out_size) return;
    uint ai = broadcast_offset(tid, p.out_shape, p.a_strides, p.ndim);
    uint bi = broadcast_offset(tid, p.out_shape, p.b_strides, p.ndim);
    out[tid] = a[ai] / b[bi];
}
