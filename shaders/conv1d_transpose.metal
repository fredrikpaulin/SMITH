// smith/shaders/conv1d_transpose.metal
// Transposed 1D convolution (deconvolution / fractionally-strided convolution).
// Used for upsampling in vocoders and audio generation.
// Input layout: [C_in, length] → Output: [C_out, outLen]
// outLen = (length - 1) * stride - 2 * padding + kernelSize + outputPadding

#include <metal_stdlib>
using namespace metal;

struct ConvTranspose1dParams {
  uint cIn;
  uint length;    // input length
  uint cOut;
  uint outLen;    // output length
  uint kernelSize;
  uint stride;
  uint padding;
};

// Forward: scatter input values through transposed weight
// Each output position gathers contributions from input positions
// gid.x = output position t, gid.y = output channel c_out
kernel void conv_transpose_1d_forward(
  device const float *input   [[buffer(0)]],   // [C_in, length]
  device const float *weight  [[buffer(1)]],   // [C_in, C_out, kernelSize]
  device float *output        [[buffer(2)]],   // [C_out, outLen]
  constant ConvTranspose1dParams &p [[buffer(3)]],
  uint2 gid                   [[thread_position_in_grid]]
) {
  uint t_out = gid.x;    // output time position
  uint c_out = gid.y;    // output channel

  if (t_out >= p.outLen || c_out >= p.cOut) return;

  float acc = 0.0f;

  // For each input channel and kernel position, check if this output
  // position receives a contribution
  for (uint c_in = 0; c_in < p.cIn; c_in++) {
    for (uint k = 0; k < p.kernelSize; k++) {
      // In transposed conv: output[t_out] gets input[t_in] * weight[c_in, c_out, k]
      // where t_out = t_in * stride - padding + k
      // so t_in = (t_out + padding - k) / stride
      int numerator = int(t_out) + int(p.padding) - int(k);
      if (numerator >= 0 && (numerator % int(p.stride)) == 0) {
        uint t_in = uint(numerator) / p.stride;
        if (t_in < p.length) {
          float inp = input[c_in * p.length + t_in];
          // weight layout: [C_in, C_out, kernelSize]
          float w = weight[c_in * p.cOut * p.kernelSize + c_out * p.kernelSize + k];
          acc += inp * w;
        }
      }
    }
  }

  output[c_out * p.outLen + t_out] = acc;
}

// f16 variant
kernel void conv_transpose_1d_forward_f16(
  device const half *input    [[buffer(0)]],
  device const half *weight   [[buffer(1)]],
  device half *output         [[buffer(2)]],
  constant ConvTranspose1dParams &p [[buffer(3)]],
  uint2 gid                   [[thread_position_in_grid]]
) {
  uint t_out = gid.x;
  uint c_out = gid.y;

  if (t_out >= p.outLen || c_out >= p.cOut) return;

  float acc = 0.0f;

  for (uint c_in = 0; c_in < p.cIn; c_in++) {
    for (uint k = 0; k < p.kernelSize; k++) {
      int numerator = int(t_out) + int(p.padding) - int(k);
      if (numerator >= 0 && (numerator % int(p.stride)) == 0) {
        uint t_in = uint(numerator) / p.stride;
        if (t_in < p.length) {
          float inp = float(input[c_in * p.length + t_in]);
          float w = float(weight[c_in * p.cOut * p.kernelSize + c_out * p.kernelSize + k]);
          acc += inp * w;
        }
      }
    }
  }

  output[c_out * p.outLen + t_out] = half(acc);
}
