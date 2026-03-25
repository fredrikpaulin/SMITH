// smith/shaders/rope.metal
// Rotary Position Embeddings (RoPE) for transformer Q/K vectors.
// Rotates pairs: (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
// where x0 is from the first half and x1 from the second half of each row.

#include <metal_stdlib>
using namespace metal;

struct RoPEParams {
  uint seqLen;    // number of sequence positions
  uint dim;       // full dimension (headDim)
  uint halfDim;   // dim / 2
  uint startPos;  // position offset for KV cache
};

// Forward: apply RoPE to input [seqLen, dim]
// cos/sin tables: [maxSeqLen, halfDim] — precomputed per-position frequencies
kernel void rope_forward(
  device const float *input   [[buffer(0)]],
  device const float *cosTab  [[buffer(1)]],
  device const float *sinTab  [[buffer(2)]],
  device float *output        [[buffer(3)]],
  constant RoPEParams &p      [[buffer(4)]],
  uint2 gid                   [[thread_position_in_grid]]
) {
  uint s = gid.y;  // sequence position
  uint i = gid.x;  // pair index within halfDim
  if (s >= p.seqLen || i >= p.halfDim) return;

  uint pos = p.startPos + s;
  float x0 = input[s * p.dim + i];
  float x1 = input[s * p.dim + p.halfDim + i];
  float c  = cosTab[pos * p.halfDim + i];
  float sn = sinTab[pos * p.halfDim + i];

  output[s * p.dim + i]              = x0 * c - x1 * sn;
  output[s * p.dim + p.halfDim + i]  = x0 * sn + x1 * c;
}

// f16 variant — half I/O, cos/sin tables stay f32 for precision
kernel void rope_forward_f16(
  device const half *input    [[buffer(0)]],
  device const float *cosTab  [[buffer(1)]],
  device const float *sinTab  [[buffer(2)]],
  device half *output         [[buffer(3)]],
  constant RoPEParams &p      [[buffer(4)]],
  uint2 gid                   [[thread_position_in_grid]]
) {
  uint s = gid.y;
  uint i = gid.x;
  if (s >= p.seqLen || i >= p.halfDim) return;

  uint pos = p.startPos + s;
  float x0 = float(input[s * p.dim + i]);
  float x1 = float(input[s * p.dim + p.halfDim + i]);
  float c  = cosTab[pos * p.halfDim + i];
  float sn = sinTab[pos * p.halfDim + i];

  output[s * p.dim + i]              = half(x0 * c - x1 * sn);
  output[s * p.dim + p.halfDim + i]  = half(x0 * sn + x1 * c);
}

// Backward: RoPE is its own inverse with negated sin
// d/dx0 of (x0*c - x1*s) = c, d/dx1 of (x0*c - x1*s) = -s
// d/dx0 of (x0*s + x1*c) = s, d/dx1 of (x0*s + x1*c) = c
// So grad_x0 = grad_out0 * c + grad_out1 * s
//    grad_x1 = -grad_out0 * s + grad_out1 * c
kernel void rope_backward(
  device const float *gradOut [[buffer(0)]],
  device const float *cosTab  [[buffer(1)]],
  device const float *sinTab  [[buffer(2)]],
  device float *gradIn        [[buffer(3)]],
  constant RoPEParams &p      [[buffer(4)]],
  uint2 gid                   [[thread_position_in_grid]]
) {
  uint s = gid.y;
  uint i = gid.x;
  if (s >= p.seqLen || i >= p.halfDim) return;

  uint pos = p.startPos + s;
  float go0 = gradOut[s * p.dim + i];
  float go1 = gradOut[s * p.dim + p.halfDim + i];
  float c   = cosTab[pos * p.halfDim + i];
  float sn  = sinTab[pos * p.halfDim + i];

  gradIn[s * p.dim + i]              = go0 * c + go1 * sn;
  gradIn[s * p.dim + p.halfDim + i]  = -go0 * sn + go1 * c;
}

kernel void rope_backward_f16(
  device const half *gradOut  [[buffer(0)]],
  device const float *cosTab  [[buffer(1)]],
  device const float *sinTab  [[buffer(2)]],
  device half *gradIn         [[buffer(3)]],
  constant RoPEParams &p      [[buffer(4)]],
  uint2 gid                   [[thread_position_in_grid]]
) {
  uint s = gid.y;
  uint i = gid.x;
  if (s >= p.seqLen || i >= p.halfDim) return;

  uint pos = p.startPos + s;
  float go0 = float(gradOut[s * p.dim + i]);
  float go1 = float(gradOut[s * p.dim + p.halfDim + i]);
  float c   = cosTab[pos * p.halfDim + i];
  float sn  = sinTab[pos * p.halfDim + i];

  gradIn[s * p.dim + i]              = half(go0 * c + go1 * sn);
  gradIn[s * p.dim + p.halfDim + i]  = half(-go0 * sn + go1 * c);
}
