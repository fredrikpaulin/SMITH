// smith/shaders/conv1d.metal
// im2col / col2im for 1D convolution.
// Input layout: [C_in, length] (no batch dimension — single sequence)

#include <metal_stdlib>
using namespace metal;

struct Conv1dParams {
  uint cIn;
  uint length;
  uint outLen;
  uint kernelSize;
  uint stride;
  uint padding;
};

// --- im2col 1D forward ---
// Transform input [C_in, length] into column matrix [C_in * kernelSize, outLen]
// gid.x = output time step (t), gid.y = patch element (c * kernelSize + k)

kernel void im2col_1d_forward(
  device const float *input  [[buffer(0)]],
  device float *cols         [[buffer(1)]],
  constant Conv1dParams &p   [[buffer(2)]],
  uint2 gid                  [[thread_position_in_grid]]
) {
  uint t = gid.x;          // output time step
  uint patchEl = gid.y;    // c * kernelSize + k

  uint colRows = p.cIn * p.kernelSize;
  if (t >= p.outLen || patchEl >= colRows) return;

  uint c = patchEl / p.kernelSize;
  uint k = patchEl % p.kernelSize;

  int pos = int(t * p.stride + k) - int(p.padding);

  float val = 0.0f;
  if (pos >= 0 && pos < int(p.length)) {
    val = input[c * p.length + uint(pos)];
  }

  // cols layout: [colRows, outLen]
  cols[patchEl * p.outLen + t] = val;
}

// --- col2im 1D backward ---
// Scatter-add column matrix [C_in * kernelSize, outLen] back to input gradient [C_in, length]
// Each thread handles one (c, pos) and accumulates contributions from all kernel
// offsets k and output positions t that map to this input position.
// gid.x = pos (input position), gid.y = c (channel)

kernel void col2im_1d_backward(
  device const float *cols    [[buffer(0)]],
  device float *gradInput     [[buffer(1)]],
  constant Conv1dParams &p    [[buffer(2)]],
  uint2 gid                   [[thread_position_in_grid]]
) {
  uint pos = gid.x;
  uint c = gid.y;

  if (pos >= p.length || c >= p.cIn) return;

  float acc = 0.0f;

  for (uint k = 0; k < p.kernelSize; k++) {
    // Find the output position t such that t * stride + k - padding == pos
    int t_num = int(pos) + int(p.padding) - int(k);
    if (t_num >= 0 && (t_num % int(p.stride)) == 0) {
      uint t = uint(t_num) / p.stride;
      if (t < p.outLen) {
        uint patchEl = c * p.kernelSize + k;
        acc += cols[patchEl * p.outLen + t];
      }
    }
  }

  gradInput[c * p.length + pos] = acc;
}
