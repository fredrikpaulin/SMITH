// smith/shaders/im2col.metal
// im2col: rearrange input patches into a column matrix for GEMM-based convolution.
// col2im: inverse operation (scatter-add) for backward pass.
// Input layout: NCHW [batch, channels, height, width]

#include <metal_stdlib>
using namespace metal;

struct Im2colParams {
  uint batch;
  uint inC;
  uint inH;
  uint inW;
  uint outH;
  uint outW;
  uint kH;
  uint kW;
  uint strideH;
  uint strideW;
  uint padH;
  uint padW;
  uint dilationH;
  uint dilationW;
};

// --- im2col forward ---
// Transform input into column matrix: [batch, inC*kH*kW, outH*outW]
// One thread per (batch, col_row, col_col) = (n, patch_element, spatial_position)
// gid.x = spatial position (oh * outW + ow), gid.y = patch element (ic * kH*kW + kh*kW + kw)
// gid.z = batch index

kernel void im2col_forward(
  device const float *input  [[buffer(0)]],
  device float *cols         [[buffer(1)]],
  constant Im2colParams &p   [[buffer(2)]],
  uint3 gid                  [[thread_position_in_grid]]
) {
  uint spatial = gid.x;   // oh * outW + ow
  uint patchEl = gid.y;   // ic * kH*kW + kh*kW + kw
  uint n = gid.z;

  uint colRows = p.inC * p.kH * p.kW;
  uint colCols = p.outH * p.outW;

  if (spatial >= colCols || patchEl >= colRows || n >= p.batch) return;

  uint oh = spatial / p.outW;
  uint ow = spatial % p.outW;

  uint kArea = p.kH * p.kW;
  uint ic = patchEl / kArea;
  uint kIdx = patchEl % kArea;
  uint kh = kIdx / p.kW;
  uint kw = kIdx % p.kW;

  int ih = int(oh * p.strideH + kh * p.dilationH) - int(p.padH);
  int iw = int(ow * p.strideW + kw * p.dilationW) - int(p.padW);

  float val = 0.0f;
  if (ih >= 0 && ih < int(p.inH) && iw >= 0 && iw < int(p.inW)) {
    val = input[n * p.inC * p.inH * p.inW + ic * p.inH * p.inW + uint(ih) * p.inW + uint(iw)];
  }

  // cols layout: [batch, colRows, colCols]
  cols[n * colRows * colCols + patchEl * colCols + spatial] = val;
}

// --- col2im backward ---
// Scatter-add column matrix back to input gradient: [batch, inC, inH, inW]
// Each thread handles one (n, ic, ih, iw) and accumulates contributions
// from all (kh, kw, oh, ow) that map to this input position.
// gid.x = iw, gid.y = ih, gid.z = n * inC + ic

kernel void col2im_backward(
  device const float *cols    [[buffer(0)]],
  device float *gradInput     [[buffer(1)]],
  constant Im2colParams &p    [[buffer(2)]],
  uint3 gid                   [[thread_position_in_grid]]
) {
  uint iw = gid.x;
  uint ih = gid.y;
  uint n_ic = gid.z;
  uint ic = n_ic % p.inC;
  uint n  = n_ic / p.inC;

  if (iw >= p.inW || ih >= p.inH || n >= p.batch) return;

  uint colRows = p.inC * p.kH * p.kW;
  uint colCols = p.outH * p.outW;
  uint kArea = p.kH * p.kW;

  float acc = 0.0f;

  for (uint kh = 0; kh < p.kH; kh++) {
    for (uint kw = 0; kw < p.kW; kw++) {
      // Find the output position that used this input position with this kernel offset
      int oh_num = int(ih) + int(p.padH) - int(kh * p.dilationH);
      int ow_num = int(iw) + int(p.padW) - int(kw * p.dilationW);

      if (oh_num >= 0 && (oh_num % int(p.strideH)) == 0 &&
          ow_num >= 0 && (ow_num % int(p.strideW)) == 0) {
        uint oh = uint(oh_num) / p.strideH;
        uint ow = uint(ow_num) / p.strideW;

        if (oh < p.outH && ow < p.outW) {
          uint patchEl = ic * kArea + kh * p.kW + kw;
          uint spatial = oh * p.outW + ow;
          acc += cols[n * colRows * colCols + patchEl * colCols + spatial];
        }
      }
    }
  }

  gradInput[n * p.inC * p.inH * p.inW + ic * p.inH * p.inW + ih * p.inW + iw] = acc;
}
