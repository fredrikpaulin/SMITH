// smith/shaders/gather_scatter.metal
// Gather (indexed read) and scatter-add (indexed write with atomic accumulation).
// Used for embedding lookup backward, beam search, advanced indexing.
//
// Layout: input is treated as [outer, dimSize, inner] where:
//   outer = product of dims before the gather axis
//   dimSize = size of the gather axis
//   inner = product of dims after the gather axis
// Indices: [outer, indexLen, inner] (broadcast outer/inner with input)
// Output:  [outer, indexLen, inner]

#include <metal_stdlib>
using namespace metal;

struct GatherParams {
    uint outer;       // product of dims before axis
    uint dimSize;     // size of the axis being indexed
    uint inner;       // product of dims after axis
    uint indexLen;    // number of indices (output size along axis)
    uint totalOut;    // outer * indexLen * inner
};

// gather_forward: output[o][i][k] = input[o][indices[i]][k]
// Grid: totalOut threads, one per output element.
kernel void gather_forward(
    device const float* input       [[buffer(0)]],
    device const uint* indices      [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant GatherParams& p        [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]])
{
    if (tid >= p.totalOut) return;

    // Decompose tid into (outerIdx, indexIdx, innerIdx)
    uint innerIdx = tid % p.inner;
    uint temp = tid / p.inner;
    uint indexIdx = temp % p.indexLen;
    uint outerIdx = temp / p.indexLen;

    uint srcDim = indices[indexIdx];  // which element along the axis to read
    if (srcDim >= p.dimSize) return;  // bounds check

    uint srcOffset = outerIdx * p.dimSize * p.inner + srcDim * p.inner + innerIdx;
    output[tid] = input[srcOffset];
}

// scatter_add: dst[o][indices[i]][k] += src[o][i][k]
// Atomic add handles duplicate indices (multiple sources map to same destination).
// Grid: totalSrc threads, one per source element.
kernel void scatter_add(
    device const float* src         [[buffer(0)]],
    device const uint* indices      [[buffer(1)]],
    device atomic_float* dst        [[buffer(2)]],
    constant GatherParams& p        [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]])
{
    if (tid >= p.totalOut) return;

    uint innerIdx = tid % p.inner;
    uint temp = tid / p.inner;
    uint indexIdx = temp % p.indexLen;
    uint outerIdx = temp / p.indexLen;

    uint dstDim = indices[indexIdx];
    if (dstDim >= p.dimSize) return;

    uint dstOffset = outerIdx * p.dimSize * p.inner + dstDim * p.inner + innerIdx;
    atomic_fetch_add_explicit(&dst[dstOffset], src[tid], memory_order_relaxed);
}

// scatter_forward: output[o][indices[i]][k] = src[o][i][k]
// Non-atomic write — last write wins for duplicate indices.
// Use scatter_add when duplicate indices need summing.
kernel void scatter_forward(
    device const float* input       [[buffer(0)]],  // base tensor to copy
    device const float* src         [[buffer(1)]],  // values to scatter
    device const uint* indices      [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant GatherParams& p        [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]])
{
    // First: copy input to output (identity)
    // This kernel assumes output is pre-filled with input.
    // Only scatter the src values.
    if (tid >= p.totalOut) return;

    uint innerIdx = tid % p.inner;
    uint temp = tid / p.inner;
    uint indexIdx = temp % p.indexLen;
    uint outerIdx = temp / p.indexLen;

    uint dstDim = indices[indexIdx];
    if (dstDim >= p.dimSize) return;

    uint dstOffset = outerIdx * p.dimSize * p.inner + dstDim * p.inner + innerIdx;
    output[dstOffset] = src[tid];
}
