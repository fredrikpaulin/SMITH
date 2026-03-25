// smith/shaders/matmul_q8.metal
// Quantized matmul: C[M,N] = A[M,K](f32) @ B[K,N](q8_0)
// B is stored as 8-bit quantized with per-group scales (GGML Q8_0 format).
// Each block of 32 elements: 2 bytes fp16 scale + 32 bytes int8 values = 34 bytes
// Dequant: val = scale * int8_val

#include <metal_stdlib>
using namespace metal;

struct Q8MatmulParams {
  uint M;
  uint N;
  uint K;
  uint groups; // ceil(K / 32)
};

constant uint Q8_BLOCK_SIZE = 32;
constant uint Q8_BLOCK_BYTES = 34; // 2 (fp16 scale) + 32 (int8 values)

kernel void matmul_q8(
  device const float *A       [[buffer(0)]],
  device const uchar *B_quant [[buffer(1)]],
  device float *C             [[buffer(2)]],
  constant Q8MatmulParams &p  [[buffer(3)]],
  uint2 gid                   [[thread_position_in_grid]]
) {
  uint row = gid.y;
  uint col = gid.x;
  if (row >= p.M || col >= p.N) return;

  float acc = 0.0f;

  for (uint g = 0; g < p.groups; g++) {
    // B is stored row-major in blocks: block index = (g * 32 + k) * N + col
    // But GGUF stores weight tensors contiguously per-row, so for a [K, N] weight:
    // The blocks are laid out per-row: for row k_group, the block covers 32 elements
    // We need column-major block layout for efficient matmul: block(g, col)
    // For GGUF Q8_0: blocks are stored sequentially per tensor row
    // Weight shape [K, N] → K/32 groups, each group has N blocks of 34 bytes
    uint block_idx = g * p.N + col;
    device const uchar *block_ptr = B_quant + block_idx * Q8_BLOCK_BYTES;

    // Read fp16 scale
    half scale_h = *reinterpret_cast<device const half *>(block_ptr);
    float scale = float(scale_h);

    uint k_start = g * Q8_BLOCK_SIZE;
    device const char *vals = reinterpret_cast<device const char *>(block_ptr + 2);

    for (uint k = 0; k < Q8_BLOCK_SIZE && (k_start + k) < p.K; k++) {
      float b_val = scale * float(vals[k]);
      acc += A[row * p.K + k_start + k] * b_val;
    }
  }

  C[row * p.N + col] = acc;
}
