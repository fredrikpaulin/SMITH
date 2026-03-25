// smith/shaders/matmul_q4.metal
// Quantized matmul: C[M,N] = A[M,K](f32) @ B[K,N](q4)
// B is stored as 4-bit quantized with per-group scales.
// Each group of 32 elements shares one f32 scale and one f32 zero-point.
// B layout in memory:
//   - For each group of 32 values: 16 bytes (32 x 4-bit nibbles) + 4 bytes scale + 4 bytes zero
//   - Total per group: 24 bytes
// This gives ~4.6 bits/weight effective (including scale overhead).

#include <metal_stdlib>
using namespace metal;

struct Q4MatmulParams {
  uint M;       // rows of A and C
  uint N;       // cols of B and C
  uint K;       // cols of A / rows of B
  uint groups;  // K / 32 (number of quantization groups per column)
};

// Dequantize one group of 32 values from packed q4
// nibbles: 16 bytes = 32 x 4-bit values
// scale, zero: per-group dequant params
// val = scale * (nibble - zero)
static float dequant_q4(device const uchar *nibbles, uint idx, float scale, float zero) {
  uchar byte = nibbles[idx / 2];
  uchar nibble = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
  return scale * (float(nibble) - zero);
}

// Bytes per group: 16 (nibbles) + 4 (scale) + 4 (zero) = 24
constant uint GROUP_SIZE = 32;
constant uint GROUP_BYTES = 24;

kernel void matmul_q4(
  device const float *A       [[buffer(0)]],
  device const uchar *B_quant [[buffer(1)]],
  device float *C             [[buffer(2)]],
  constant Q4MatmulParams &p  [[buffer(3)]],
  uint2 gid                   [[thread_position_in_grid]]
) {
  uint row = gid.y;
  uint col = gid.x;
  if (row >= p.M || col >= p.N) return;

  float acc = 0.0f;

  for (uint g = 0; g < p.groups; g++) {
    // Locate this group's data for column `col`
    // B is stored column-major in groups: group index = g * N + col
    uint group_idx = g * p.N + col;
    device const uchar *group_ptr = B_quant + group_idx * GROUP_BYTES;

    // Read scale and zero from the end of the group
    float scale = *reinterpret_cast<device const float *>(group_ptr + 16);
    float zero  = *reinterpret_cast<device const float *>(group_ptr + 20);

    uint k_start = g * GROUP_SIZE;
    for (uint k = 0; k < GROUP_SIZE && (k_start + k) < p.K; k++) {
      float b_val = dequant_q4(group_ptr, k, scale, zero);
      acc += A[row * p.K + k_start + k] * b_val;
    }
  }

  C[row * p.N + col] = acc;
}
