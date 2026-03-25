// smith/shaders/conv2d_winograd.metal
// Winograd F(2x2, 3x3) convolution: forward and backward (input gradient).
// Computes 2x2 output tiles from 4x4 input tiles using 16 multiplications
// instead of 36 — a 2.25x reduction in arithmetic intensity.
// Only valid for 3x3 kernels, stride 1, dilation 1.
// Input layout: NCHW, Weight layout: [outC, inC/groups, 3, 3]

#include <metal_stdlib>
using namespace metal;

struct WinogradParams {
  uint batch;
  uint inC;
  uint inH;
  uint inW;
  uint outC;
  uint outH;   // = inH - 2 (no padding) or inH (pad=1)
  uint outW;
  uint tileH;  // ceil(outH / 2)
  uint tileW;  // ceil(outW / 2)
  uint padH;
  uint padW;
  uint groups;
};

// --- Forward: Winograd F(2x2, 3x3) ---
// Each thread computes one output tile (2x2) for one (n, oc) pair.
// gid.x = tileCol, gid.y = tileRow, gid.z = n * outC + oc

kernel void conv2d_winograd_forward(
  device const float *input       [[buffer(0)]],  // NCHW
  device const float *transWeight [[buffer(1)]],  // pre-transformed: [outC, inC, 4, 4]
  device const float *bias        [[buffer(2)]],
  device float *output            [[buffer(3)]],  // NCHW
  constant WinogradParams &p      [[buffer(4)]],
  uint3 gid                       [[thread_position_in_grid]]
) {
  uint tc = gid.x;  // tile column
  uint tr = gid.y;  // tile row
  uint n_oc = gid.z;
  uint oc = n_oc % p.outC;
  uint n  = n_oc / p.outC;

  if (tc >= p.tileW || tr >= p.tileH || n >= p.batch) return;

  uint groupSize = p.inC / p.groups;
  uint group = oc / (p.outC / p.groups);
  uint icStart = group * groupSize;

  // Accumulate transformed domain products across input channels
  float m[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

  int tileOriginH = int(tr * 2) - int(p.padH);
  int tileOriginW = int(tc * 2) - int(p.padW);

  for (uint ic = 0; ic < groupSize; ic++) {
    uint icGlobal = icStart + ic;

    // Load 4x4 input tile with boundary checks
    float d[16];
    for (uint r = 0; r < 4; r++) {
      for (uint c = 0; c < 4; c++) {
        int ih = tileOriginH + int(r);
        int iw = tileOriginW + int(c);
        if (ih >= 0 && ih < int(p.inH) && iw >= 0 && iw < int(p.inW)) {
          d[r * 4 + c] = input[n * p.inC * p.inH * p.inW + icGlobal * p.inH * p.inW + uint(ih) * p.inW + uint(iw)];
        } else {
          d[r * 4 + c] = 0.0f;
        }
      }
    }

    // B^T * d * B  (input transform)
    // B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
    float temp[16];
    // B^T * d (rows)
    for (uint c = 0; c < 4; c++) {
      temp[0 * 4 + c] = d[0 * 4 + c] - d[2 * 4 + c];
      temp[1 * 4 + c] = d[1 * 4 + c] + d[2 * 4 + c];
      temp[2 * 4 + c] = -d[1 * 4 + c] + d[2 * 4 + c];
      temp[3 * 4 + c] = d[1 * 4 + c] - d[3 * 4 + c];
    }
    // (B^T * d) * B (cols), B = transpose of B^T
    float V[16];
    for (uint r = 0; r < 4; r++) {
      V[r * 4 + 0] = temp[r * 4 + 0] - temp[r * 4 + 2];
      V[r * 4 + 1] = temp[r * 4 + 1] + temp[r * 4 + 2];
      V[r * 4 + 2] = -temp[r * 4 + 1] + temp[r * 4 + 2];
      V[r * 4 + 3] = temp[r * 4 + 1] - temp[r * 4 + 3];
    }

    // Load pre-transformed weight: U[oc, ic, 4, 4]
    uint wBase = oc * groupSize * 16 + ic * 16;

    // Element-wise multiply and accumulate
    for (uint i = 0; i < 16; i++) {
      m[i] += V[i] * transWeight[wBase + i];
    }
  }

  // A^T * m * A  (output transform)
  // A^T = [[1,1,1,0],[0,1,-1,-1]]
  float temp2[8];
  // A^T * m (rows): 2x4 * 4x4 = 2x4
  for (uint c = 0; c < 4; c++) {
    temp2[0 * 4 + c] = m[0 * 4 + c] + m[1 * 4 + c] + m[2 * 4 + c];
    temp2[1 * 4 + c] = m[1 * 4 + c] - m[2 * 4 + c] - m[3 * 4 + c];
  }
  // (A^T * m) * A (cols): 2x4 * 4x2 = 2x2
  float o[4];
  o[0] = temp2[0] + temp2[1] + temp2[2];
  o[1] = temp2[1] - temp2[2] - temp2[3];
  o[2] = temp2[4] + temp2[5] + temp2[6];
  o[3] = temp2[5] - temp2[6] - temp2[7];

  // Add bias
  float b = bias[oc];

  // Write 2x2 output tile
  uint outBase = n * p.outC * p.outH * p.outW + oc * p.outH * p.outW;
  uint oh0 = tr * 2;
  uint ow0 = tc * 2;

  if (oh0 < p.outH && ow0 < p.outW)
    output[outBase + oh0 * p.outW + ow0] = o[0] + b;
  if (oh0 < p.outH && ow0 + 1 < p.outW)
    output[outBase + oh0 * p.outW + ow0 + 1] = o[1] + b;
  if (oh0 + 1 < p.outH && ow0 < p.outW)
    output[outBase + (oh0 + 1) * p.outW + ow0] = o[2] + b;
  if (oh0 + 1 < p.outH && ow0 + 1 < p.outW)
    output[outBase + (oh0 + 1) * p.outW + ow0 + 1] = o[3] + b;
}

// Forward without bias
kernel void conv2d_winograd_forward_no_bias(
  device const float *input       [[buffer(0)]],
  device const float *transWeight [[buffer(1)]],
  device float *output            [[buffer(2)]],
  constant WinogradParams &p      [[buffer(3)]],
  uint3 gid                       [[thread_position_in_grid]]
) {
  uint tc = gid.x;
  uint tr = gid.y;
  uint n_oc = gid.z;
  uint oc = n_oc % p.outC;
  uint n  = n_oc / p.outC;

  if (tc >= p.tileW || tr >= p.tileH || n >= p.batch) return;

  uint groupSize = p.inC / p.groups;
  uint group = oc / (p.outC / p.groups);
  uint icStart = group * groupSize;

  float m[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

  int tileOriginH = int(tr * 2) - int(p.padH);
  int tileOriginW = int(tc * 2) - int(p.padW);

  for (uint ic = 0; ic < groupSize; ic++) {
    uint icGlobal = icStart + ic;

    float d[16];
    for (uint r = 0; r < 4; r++) {
      for (uint c = 0; c < 4; c++) {
        int ih = tileOriginH + int(r);
        int iw = tileOriginW + int(c);
        if (ih >= 0 && ih < int(p.inH) && iw >= 0 && iw < int(p.inW)) {
          d[r * 4 + c] = input[n * p.inC * p.inH * p.inW + icGlobal * p.inH * p.inW + uint(ih) * p.inW + uint(iw)];
        } else {
          d[r * 4 + c] = 0.0f;
        }
      }
    }

    float temp[16];
    for (uint c = 0; c < 4; c++) {
      temp[0 * 4 + c] = d[0 * 4 + c] - d[2 * 4 + c];
      temp[1 * 4 + c] = d[1 * 4 + c] + d[2 * 4 + c];
      temp[2 * 4 + c] = -d[1 * 4 + c] + d[2 * 4 + c];
      temp[3 * 4 + c] = d[1 * 4 + c] - d[3 * 4 + c];
    }
    float V[16];
    for (uint r = 0; r < 4; r++) {
      V[r * 4 + 0] = temp[r * 4 + 0] - temp[r * 4 + 2];
      V[r * 4 + 1] = temp[r * 4 + 1] + temp[r * 4 + 2];
      V[r * 4 + 2] = -temp[r * 4 + 1] + temp[r * 4 + 2];
      V[r * 4 + 3] = temp[r * 4 + 1] - temp[r * 4 + 3];
    }

    uint wBase = oc * groupSize * 16 + ic * 16;
    for (uint i = 0; i < 16; i++) {
      m[i] += V[i] * transWeight[wBase + i];
    }
  }

  float temp2[8];
  for (uint c = 0; c < 4; c++) {
    temp2[0 * 4 + c] = m[0 * 4 + c] + m[1 * 4 + c] + m[2 * 4 + c];
    temp2[1 * 4 + c] = m[1 * 4 + c] - m[2 * 4 + c] - m[3 * 4 + c];
  }
  float o[4];
  o[0] = temp2[0] + temp2[1] + temp2[2];
  o[1] = temp2[1] - temp2[2] - temp2[3];
  o[2] = temp2[4] + temp2[5] + temp2[6];
  o[3] = temp2[5] - temp2[6] - temp2[7];

  uint outBase = n * p.outC * p.outH * p.outW + oc * p.outH * p.outW;
  uint oh0 = tr * 2;
  uint ow0 = tc * 2;

  if (oh0 < p.outH && ow0 < p.outW)
    output[outBase + oh0 * p.outW + ow0] = o[0];
  if (oh0 < p.outH && ow0 + 1 < p.outW)
    output[outBase + oh0 * p.outW + ow0 + 1] = o[1];
  if (oh0 + 1 < p.outH && ow0 < p.outW)
    output[outBase + (oh0 + 1) * p.outW + ow0] = o[2];
  if (oh0 + 1 < p.outH && ow0 + 1 < p.outW)
    output[outBase + (oh0 + 1) * p.outW + ow0 + 1] = o[3];
}

// --- Backward: Winograd gradient w.r.t. input ---
// Uses the Winograd transform in reverse:
// gradInput tile (4x4) from gradOutput tile (2x2) with transposed transformed weight.
// One thread per input tile per (n, ic) pair.
// gid.x = tileCol, gid.y = tileRow, gid.z = n * inC + ic

kernel void conv2d_winograd_backward_input(
  device const float *gradOutput   [[buffer(0)]],  // NCHW [batch, outC, outH, outW]
  device const float *transWeight  [[buffer(1)]],  // pre-transformed: [outC, inC/groups, 4, 4]
  device float *gradInput          [[buffer(2)]],  // NCHW [batch, inC, inH, inW]
  constant WinogradParams &p       [[buffer(3)]],
  uint3 gid                        [[thread_position_in_grid]]
) {
  uint tc = gid.x;
  uint tr = gid.y;
  uint n_ic = gid.z;
  uint ic = n_ic % p.inC;
  uint n  = n_ic / p.inC;

  if (tc >= p.tileW || tr >= p.tileH || n >= p.batch) return;

  uint groupSize = p.inC / p.groups;
  uint group = ic / groupSize;
  uint ocStart = group * (p.outC / p.groups);
  uint ocEnd = ocStart + (p.outC / p.groups);
  uint icInGroup = ic - group * groupSize;

  // Accumulate in transform domain across output channels
  float m[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

  uint oh0 = tr * 2;
  uint ow0 = tc * 2;

  for (uint oc = ocStart; oc < ocEnd; oc++) {
    // Load 2x2 gradOutput tile, zero-pad to build the 4x4 representation
    // For backward, we need A * gradOutput * A^T to get 4x4 transform
    float go[4] = {0, 0, 0, 0};
    uint goBase = n * p.outC * p.outH * p.outW + oc * p.outH * p.outW;
    if (oh0 < p.outH && ow0 < p.outW) go[0] = gradOutput[goBase + oh0 * p.outW + ow0];
    if (oh0 < p.outH && ow0 + 1 < p.outW) go[1] = gradOutput[goBase + oh0 * p.outW + ow0 + 1];
    if (oh0 + 1 < p.outH && ow0 < p.outW) go[2] = gradOutput[goBase + (oh0 + 1) * p.outW + ow0];
    if (oh0 + 1 < p.outH && ow0 + 1 < p.outW) go[3] = gradOutput[goBase + (oh0 + 1) * p.outW + ow0 + 1];

    // A * go * A^T  (2x2 → 4x4)
    // A = [[1,0],[1,1],[1,-1],[0,-1]]
    // First: A * go (4x2)
    float ag[8];
    ag[0] = go[0]; ag[1] = go[1];               // row 0: [1,0] * go
    ag[2] = go[0] + go[2]; ag[3] = go[1] + go[3]; // row 1: [1,1]
    ag[4] = go[0] - go[2]; ag[5] = go[1] - go[3]; // row 2: [1,-1]
    ag[6] = -go[2]; ag[7] = -go[3];              // row 3: [0,-1]

    // (A * go) * A^T (4x4)
    float goT[16];
    for (uint r = 0; r < 4; r++) {
      float r0 = ag[r * 2 + 0];
      float r1 = ag[r * 2 + 1];
      goT[r * 4 + 0] = r0;                // col 0: * [1,0]^T
      goT[r * 4 + 1] = r0 + r1;          // col 1: * [1,1]^T
      goT[r * 4 + 2] = r0 - r1;          // col 2: * [1,-1]^T
      goT[r * 4 + 3] = -r1;              // col 3: * [0,-1]^T
    }

    // Load transposed weight (for backward: use weight[oc, icInGroup])
    uint wBase = oc * groupSize * 16 + icInGroup * 16;

    // Element-wise multiply and accumulate
    for (uint i = 0; i < 16; i++) {
      m[i] += goT[i] * transWeight[wBase + i];
    }
  }

  // B * m * B^T  (inverse input transform for gradient)
  // B rows: [[1,0,0,0],[0,1,-1,1],[-1,1,1,0],[0,0,0,-1]]
  // B^T cols: [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
  float temp[16];
  // Left multiply by B (rows)
  for (uint c = 0; c < 4; c++) {
    temp[0 * 4 + c] = m[0 * 4 + c];
    temp[1 * 4 + c] = m[1 * 4 + c] - m[2 * 4 + c] + m[3 * 4 + c];
    temp[2 * 4 + c] = -m[0 * 4 + c] + m[1 * 4 + c] + m[2 * 4 + c];
    temp[3 * 4 + c] = -m[3 * 4 + c];
  }
  float gi[16];
  // Right multiply by B^T (cols)
  for (uint r = 0; r < 4; r++) {
    gi[r * 4 + 0] = temp[r * 4 + 0];
    gi[r * 4 + 1] = temp[r * 4 + 1] - temp[r * 4 + 2] + temp[r * 4 + 3];
    gi[r * 4 + 2] = -temp[r * 4 + 0] + temp[r * 4 + 1] + temp[r * 4 + 2];
    gi[r * 4 + 3] = -temp[r * 4 + 3];
  }

  // Scatter-add 4x4 gradient into gradInput (tiles overlap by 2)
  int tileOriginH = int(tr * 2) - int(p.padH);
  int tileOriginW = int(tc * 2) - int(p.padW);
  uint giBase = n * p.inC * p.inH * p.inW + ic * p.inH * p.inW;

  for (uint r = 0; r < 4; r++) {
    for (uint c = 0; c < 4; c++) {
      int ih = tileOriginH + int(r);
      int iw = tileOriginW + int(c);
      if (ih >= 0 && ih < int(p.inH) && iw >= 0 && iw < int(p.inW)) {
        // Atomic add because overlapping tiles write to same positions
        device atomic_float *dst = (device atomic_float *)&gradInput[giBase + uint(ih) * p.inW + uint(iw)];
        atomic_fetch_add_explicit(dst, gi[r * 4 + c], memory_order_relaxed);
      }
    }
  }
}
