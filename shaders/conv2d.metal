// smith/shaders/conv2d.metal
// 2D convolution: forward, backward_input (transposed conv), backward_weight.
// Input layout: NCHW (batch, channels, height, width)
// Weight layout: [outChannels, inChannels/groups, kH, kW]

#include <metal_stdlib>
using namespace metal;

struct Conv2dParams {
  uint batch;
  uint inC;       // input channels
  uint inH;
  uint inW;
  uint outC;      // output channels
  uint outH;
  uint outW;
  uint kH;        // kernel height
  uint kW;        // kernel width
  uint strideH;
  uint strideW;
  uint padH;
  uint padW;
  uint dilationH;
  uint dilationW;
  uint groups;
};

// --- Forward: direct convolution ---
// One thread per output element: out[n, oc, oh, ow]
// For each output position, accumulate over the kernel window.

kernel void conv2d_forward(
  device const float *input   [[buffer(0)]],
  device const float *weight  [[buffer(1)]],
  device const float *bias    [[buffer(2)]],
  device float *output        [[buffer(3)]],
  constant Conv2dParams &p    [[buffer(4)]],
  uint3 gid                   [[thread_position_in_grid]]
) {
  // gid.x = ow, gid.y = oh, gid.z encodes (n * outC + oc)
  uint ow = gid.x;
  uint oh = gid.y;
  uint n_oc = gid.z;
  uint oc = n_oc % p.outC;
  uint n  = n_oc / p.outC;

  if (ow >= p.outW || oh >= p.outH || n >= p.batch) return;

  uint groupSize = p.inC / p.groups;
  uint group = oc / (p.outC / p.groups);
  uint icStart = group * groupSize;

  float acc = 0.0f;

  for (uint ic = 0; ic < groupSize; ic++) {
    for (uint kh = 0; kh < p.kH; kh++) {
      for (uint kw = 0; kw < p.kW; kw++) {
        int ih = int(oh * p.strideH + kh * p.dilationH) - int(p.padH);
        int iw = int(ow * p.strideW + kw * p.dilationW) - int(p.padW);

        if (ih >= 0 && ih < int(p.inH) && iw >= 0 && iw < int(p.inW)) {
          uint inIdx = n * p.inC * p.inH * p.inW
                     + (icStart + ic) * p.inH * p.inW
                     + uint(ih) * p.inW + uint(iw);
          uint wIdx = oc * groupSize * p.kH * p.kW
                    + ic * p.kH * p.kW
                    + kh * p.kW + kw;
          acc += input[inIdx] * weight[wIdx];
        }
      }
    }
  }

  // Add bias
  acc += bias[oc];

  uint outIdx = n * p.outC * p.outH * p.outW
              + oc * p.outH * p.outW
              + oh * p.outW + ow;
  output[outIdx] = acc;
}

// Forward without bias
kernel void conv2d_forward_no_bias(
  device const float *input   [[buffer(0)]],
  device const float *weight  [[buffer(1)]],
  device float *output        [[buffer(2)]],
  constant Conv2dParams &p    [[buffer(3)]],
  uint3 gid                   [[thread_position_in_grid]]
) {
  uint ow = gid.x;
  uint oh = gid.y;
  uint n_oc = gid.z;
  uint oc = n_oc % p.outC;
  uint n  = n_oc / p.outC;

  if (ow >= p.outW || oh >= p.outH || n >= p.batch) return;

  uint groupSize = p.inC / p.groups;
  uint group = oc / (p.outC / p.groups);
  uint icStart = group * groupSize;

  float acc = 0.0f;

  for (uint ic = 0; ic < groupSize; ic++) {
    for (uint kh = 0; kh < p.kH; kh++) {
      for (uint kw = 0; kw < p.kW; kw++) {
        int ih = int(oh * p.strideH + kh * p.dilationH) - int(p.padH);
        int iw = int(ow * p.strideW + kw * p.dilationW) - int(p.padW);

        if (ih >= 0 && ih < int(p.inH) && iw >= 0 && iw < int(p.inW)) {
          uint inIdx = n * p.inC * p.inH * p.inW
                     + (icStart + ic) * p.inH * p.inW
                     + uint(ih) * p.inW + uint(iw);
          uint wIdx = oc * groupSize * p.kH * p.kW
                    + ic * p.kH * p.kW
                    + kh * p.kW + kw;
          acc += input[inIdx] * weight[wIdx];
        }
      }
    }
  }

  uint outIdx = n * p.outC * p.outH * p.outW
              + oc * p.outH * p.outW
              + oh * p.outW + ow;
  output[outIdx] = acc;
}

// --- Backward: gradient w.r.t. input ---
// dInput[n, ic, ih, iw] = sum over oc,kh,kw of weight[oc,ic,kh,kw] * dOutput[n,oc,oh,ow]
// where oh = (ih + padH - kh*dilH) / strH (when divisible and in range)

kernel void conv2d_backward_input(
  device const float *gradOutput [[buffer(0)]],
  device const float *weight     [[buffer(1)]],
  device float *gradInput        [[buffer(2)]],
  constant Conv2dParams &p       [[buffer(3)]],
  uint3 gid                      [[thread_position_in_grid]]
) {
  uint iw = gid.x;
  uint ih = gid.y;
  uint n_ic = gid.z;
  uint ic = n_ic % p.inC;
  uint n  = n_ic / p.inC;

  if (iw >= p.inW || ih >= p.inH || n >= p.batch) return;

  uint groupSize = p.inC / p.groups;
  uint group = ic / groupSize;
  uint ocStart = group * (p.outC / p.groups);
  uint ocEnd = ocStart + (p.outC / p.groups);
  uint icInGroup = ic - group * groupSize;

  float acc = 0.0f;

  for (uint oc = ocStart; oc < ocEnd; oc++) {
    for (uint kh = 0; kh < p.kH; kh++) {
      for (uint kw = 0; kw < p.kW; kw++) {
        int oh_num = int(ih) + int(p.padH) - int(kh * p.dilationH);
        int ow_num = int(iw) + int(p.padW) - int(kw * p.dilationW);

        if (oh_num >= 0 && (oh_num % int(p.strideH)) == 0 &&
            ow_num >= 0 && (ow_num % int(p.strideW)) == 0) {
          uint oh = uint(oh_num) / p.strideH;
          uint ow = uint(ow_num) / p.strideW;

          if (oh < p.outH && ow < p.outW) {
            uint goIdx = n * p.outC * p.outH * p.outW
                       + oc * p.outH * p.outW
                       + oh * p.outW + ow;
            uint wIdx = oc * groupSize * p.kH * p.kW
                      + icInGroup * p.kH * p.kW
                      + kh * p.kW + kw;
            acc += gradOutput[goIdx] * weight[wIdx];
          }
        }
      }
    }
  }

  uint inIdx = n * p.inC * p.inH * p.inW
             + ic * p.inH * p.inW
             + ih * p.inW + iw;
  gradInput[inIdx] = acc;
}

// --- Backward: gradient w.r.t. weight ---
// dWeight[oc, ic, kh, kw] = sum over n,oh,ow of input[n,ic,ih,iw] * dOutput[n,oc,oh,ow]
// where ih = oh*strH + kh*dilH - padH

kernel void conv2d_backward_weight(
  device const float *input      [[buffer(0)]],
  device const float *gradOutput [[buffer(1)]],
  device float *gradWeight       [[buffer(2)]],
  constant Conv2dParams &p       [[buffer(3)]],
  uint3 gid                      [[thread_position_in_grid]]
) {
  // gid.x = kw, gid.y = kh, gid.z encodes (oc * groupSize + ic)
  uint kw = gid.x;
  uint kh = gid.y;
  uint oc_ic = gid.z;

  uint groupSize = p.inC / p.groups;
  uint ic = oc_ic % groupSize;
  uint oc = oc_ic / groupSize;

  if (kw >= p.kW || kh >= p.kH || oc >= p.outC) return;

  uint group = oc / (p.outC / p.groups);
  uint icGlobal = group * groupSize + ic;

  float acc = 0.0f;

  for (uint n = 0; n < p.batch; n++) {
    for (uint oh = 0; oh < p.outH; oh++) {
      for (uint ow = 0; ow < p.outW; ow++) {
        int ih = int(oh * p.strideH + kh * p.dilationH) - int(p.padH);
        int iw = int(ow * p.strideW + kw * p.dilationW) - int(p.padW);

        if (ih >= 0 && ih < int(p.inH) && iw >= 0 && iw < int(p.inW)) {
          uint inIdx = n * p.inC * p.inH * p.inW
                     + icGlobal * p.inH * p.inW
                     + uint(ih) * p.inW + uint(iw);
          uint goIdx = n * p.outC * p.outH * p.outW
                     + oc * p.outH * p.outW
                     + oh * p.outW + ow;
          acc += input[inIdx] * gradOutput[goIdx];
        }
      }
    }
  }

  uint wIdx = oc * groupSize * p.kH * p.kW
            + ic * p.kH * p.kW
            + kh * p.kW + kw;
  gradWeight[wIdx] = acc;
}

// --- Backward: gradient w.r.t. bias ---
// dBias[oc] = sum over n,oh,ow of dOutput[n,oc,oh,ow]

kernel void conv2d_backward_bias(
  device const float *gradOutput [[buffer(0)]],
  device float *gradBias         [[buffer(1)]],
  constant Conv2dParams &p       [[buffer(2)]],
  uint gid                       [[thread_position_in_grid]]
) {
  uint oc = gid;
  if (oc >= p.outC) return;

  float acc = 0.0f;
  for (uint n = 0; n < p.batch; n++) {
    for (uint oh = 0; oh < p.outH; oh++) {
      for (uint ow = 0; ow < p.outW; ow++) {
        uint idx = n * p.outC * p.outH * p.outW
                 + oc * p.outH * p.outW
                 + oh * p.outW + ow;
        acc += gradOutput[idx];
      }
    }
  }

  gradBias[oc] = acc;
}
