// smith/shaders/fft.metal
// Radix-2 Cooley-Tukey FFT in threadgroup shared memory.
// Each threadgroup processes one FFT of length N (power of 2, max 4096).
// Input/output: interleaved complex [re0, im0, re1, im1, ...] in a single buffer.
// For batch FFT, each threadgroup handles one independent FFT (gid.y = batch index).

#include <metal_stdlib>
using namespace metal;

struct FFTParams {
  uint n;         // FFT length (power of 2)
  uint logN;      // log2(n)
  uint batch;     // number of independent FFTs
  uint inverse;   // 0 = forward, 1 = inverse
};

// Bit-reversal of 'val' for 'bits' bit positions
static uint bitReverse(uint val, uint bits) {
  uint result = 0;
  for (uint i = 0; i < bits; i++) {
    result = (result << 1) | (val & 1);
    val >>= 1;
  }
  return result;
}

// In-threadgroup radix-2 FFT.
// gid.x = thread index within one FFT (handles two elements per butterfly)
// gid.y = batch index
// Each threadgroup needs n * 2 floats of shared memory (n complex values).

kernel void fft_radix2(
  device const float *input  [[buffer(0)]],
  device float *output       [[buffer(1)]],
  constant FFTParams &p      [[buffer(2)]],
  uint2 gid                  [[thread_position_in_grid]],
  uint2 lid                  [[thread_position_in_threadgroup]],
  uint2 tgid                 [[threadgroup_position_in_grid]]
) {
  // Shared memory: interleaved complex, size = n * 2 floats
  // Max: 4096 complex = 32KB of float = fine for Metal's threadgroup limit
  threadgroup float shared[8192]; // 4096 complex numbers max

  uint tid = lid.x;
  uint batchIdx = tgid.y;
  if (batchIdx >= p.batch) return;

  uint n = p.n;
  uint logN = p.logN;

  // Base offset in input/output buffers for this batch
  uint batchOff = batchIdx * n * 2;

  // Load input into shared memory with bit-reversal permutation
  // Each of n threads loads one complex element
  if (tid < n) {
    uint rev = bitReverse(tid, logN);
    shared[rev * 2]     = input[batchOff + tid * 2];
    shared[rev * 2 + 1] = input[batchOff + tid * 2 + 1];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Butterfly stages
  float sign = (p.inverse != 0) ? 1.0f : -1.0f;

  for (uint s = 1; s <= logN; s++) {
    uint m = 1u << s;           // butterfly group size
    uint halfM = m >> 1;

    // Each thread handles one butterfly
    if (tid < n / 2) {
      uint group = tid / halfM;
      uint j = tid % halfM;
      uint idx0 = group * m + j;
      uint idx1 = idx0 + halfM;

      // Twiddle factor: e^(sign * 2πi * j / m)
      float angle = sign * 2.0f * M_PI_F * float(j) / float(m);
      float wr = cos(angle);
      float wi = sin(angle);

      float ar = shared[idx0 * 2];
      float ai = shared[idx0 * 2 + 1];
      float br = shared[idx1 * 2];
      float bi = shared[idx1 * 2 + 1];

      // Butterfly: a' = a + w*b, b' = a - w*b
      float tr = wr * br - wi * bi;
      float ti = wr * bi + wi * br;

      shared[idx0 * 2]     = ar + tr;
      shared[idx0 * 2 + 1] = ai + ti;
      shared[idx1 * 2]     = ar - tr;
      shared[idx1 * 2 + 1] = ai - ti;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write output (with optional 1/N normalization for inverse)
  if (tid < n) {
    float scale = (p.inverse != 0) ? (1.0f / float(n)) : 1.0f;
    output[batchOff + tid * 2]     = shared[tid * 2]     * scale;
    output[batchOff + tid * 2 + 1] = shared[tid * 2 + 1] * scale;
  }
}
