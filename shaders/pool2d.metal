// smith/shaders/pool2d.metal
// 2D pooling: max pool and average pool.
// Input layout: NCHW (batch, channels, height, width)

#include <metal_stdlib>
using namespace metal;

struct Pool2dParams {
  uint batch;
  uint channels;
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
};

// --- Max pooling forward ---
// Also writes argmax indices for backward pass

kernel void maxpool2d_forward(
  device const float *input  [[buffer(0)]],
  device float *output       [[buffer(1)]],
  device uint *indices       [[buffer(2)]],
  constant Pool2dParams &p   [[buffer(3)]],
  uint3 gid                  [[thread_position_in_grid]]
) {
  uint ow = gid.x;
  uint oh = gid.y;
  uint n_c = gid.z;
  uint c = n_c % p.channels;
  uint n = n_c / p.channels;

  if (ow >= p.outW || oh >= p.outH || n >= p.batch) return;

  float maxVal = -INFINITY;
  uint maxIdx = 0;

  for (uint kh = 0; kh < p.kH; kh++) {
    for (uint kw = 0; kw < p.kW; kw++) {
      int ih = int(oh * p.strideH + kh) - int(p.padH);
      int iw = int(ow * p.strideW + kw) - int(p.padW);

      if (ih >= 0 && ih < int(p.inH) && iw >= 0 && iw < int(p.inW)) {
        uint idx = n * p.channels * p.inH * p.inW
                 + c * p.inH * p.inW
                 + uint(ih) * p.inW + uint(iw);
        float val = input[idx];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = idx;
        }
      }
    }
  }

  uint outIdx = n * p.channels * p.outH * p.outW
              + c * p.outH * p.outW
              + oh * p.outW + ow;
  output[outIdx] = maxVal;
  indices[outIdx] = maxIdx;
}

// --- Max pooling backward ---
// Scatter gradient to the position that was the max

kernel void maxpool2d_backward(
  device const float *gradOutput [[buffer(0)]],
  device const uint *indices     [[buffer(1)]],
  device float *gradInput        [[buffer(2)]],
  constant Pool2dParams &p       [[buffer(3)]],
  uint gid                       [[thread_position_in_grid]]
) {
  uint totalOut = p.batch * p.channels * p.outH * p.outW;
  if (gid >= totalOut) return;

  float go = gradOutput[gid];
  uint srcIdx = indices[gid];
  // Atomic add since multiple output positions could map to same input (with overlapping windows)
  // For non-overlapping pools (stride >= kernel), this is just a write
  // Metal doesn't have atomic float add on all devices, so we use a simple write
  // (correct for non-overlapping pools; for overlapping, caller should use CPU accumulation)
  gradInput[srcIdx] += go;
}

// --- Average pooling forward ---

kernel void avgpool2d_forward(
  device const float *input  [[buffer(0)]],
  device float *output       [[buffer(1)]],
  constant Pool2dParams &p   [[buffer(2)]],
  uint3 gid                  [[thread_position_in_grid]]
) {
  uint ow = gid.x;
  uint oh = gid.y;
  uint n_c = gid.z;
  uint c = n_c % p.channels;
  uint n = n_c / p.channels;

  if (ow >= p.outW || oh >= p.outH || n >= p.batch) return;

  float sum = 0.0f;
  uint count = 0;

  for (uint kh = 0; kh < p.kH; kh++) {
    for (uint kw = 0; kw < p.kW; kw++) {
      int ih = int(oh * p.strideH + kh) - int(p.padH);
      int iw = int(ow * p.strideW + kw) - int(p.padW);

      if (ih >= 0 && ih < int(p.inH) && iw >= 0 && iw < int(p.inW)) {
        uint idx = n * p.channels * p.inH * p.inW
                 + c * p.inH * p.inW
                 + uint(ih) * p.inW + uint(iw);
        sum += input[idx];
        count++;
      }
    }
  }

  uint outIdx = n * p.channels * p.outH * p.outW
              + c * p.outH * p.outW
              + oh * p.outW + ow;
  output[outIdx] = count > 0 ? sum / float(count) : 0.0f;
}

// --- Average pooling backward ---
// Distribute gradient evenly to all input positions in the window

kernel void avgpool2d_backward(
  device const float *gradOutput [[buffer(0)]],
  device float *gradInput        [[buffer(1)]],
  constant Pool2dParams &p       [[buffer(2)]],
  uint3 gid                      [[thread_position_in_grid]]
) {
  // One thread per input element
  uint iw = gid.x;
  uint ih = gid.y;
  uint n_c = gid.z;
  uint c = n_c % p.channels;
  uint n = n_c / p.channels;

  if (iw >= p.inW || ih >= p.inH || n >= p.batch) return;

  float acc = 0.0f;

  // Find all output positions that include this input position
  for (uint oh = 0; oh < p.outH; oh++) {
    for (uint ow = 0; ow < p.outW; ow++) {
      // Check if (ih, iw) falls in the window for (oh, ow)
      int kh_start = int(ih) + int(p.padH) - int(oh * p.strideH);
      int kw_start = int(iw) + int(p.padW) - int(ow * p.strideW);

      if (kh_start >= 0 && kh_start < int(p.kH) &&
          kw_start >= 0 && kw_start < int(p.kW)) {
        // Count valid positions in this window for average divisor
        uint count = 0;
        for (uint kh = 0; kh < p.kH; kh++) {
          for (uint kw = 0; kw < p.kW; kw++) {
            int pih = int(oh * p.strideH + kh) - int(p.padH);
            int piw = int(ow * p.strideW + kw) - int(p.padW);
            if (pih >= 0 && pih < int(p.inH) && piw >= 0 && piw < int(p.inW)) count++;
          }
        }

        uint goIdx = n * p.channels * p.outH * p.outW
                   + c * p.outH * p.outW
                   + oh * p.outW + ow;
        acc += gradOutput[goIdx] / float(count);
      }
    }
  }

  uint inIdx = n * p.channels * p.inH * p.inW
             + c * p.inH * p.inW
             + ih * p.inW + iw;
  gradInput[inIdx] = acc;
}
