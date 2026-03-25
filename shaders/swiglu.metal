// smith/shaders/swiglu.metal
// Fused SwiGLU activation: out = silu(gate) * up
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// This fuses the SiLU activation and element-wise multiply into one kernel,
// eliminating an intermediate buffer.

#include <metal_stdlib>
using namespace metal;

// Fused SiLU(gate) * up — one thread per element
// gate and up are [rows, cols], output is [rows, cols]
kernel void swiglu_forward(
  device const float *gate   [[buffer(0)]],
  device const float *up     [[buffer(1)]],
  device float *output       [[buffer(2)]],
  uint gid                   [[thread_position_in_grid]]
) {
  float g = gate[gid];
  float silu_g = g / (1.0f + exp(-g));  // silu = x * sigmoid(x)
  output[gid] = silu_g * up[gid];
}

kernel void swiglu_forward_f16(
  device const half *gate    [[buffer(0)]],
  device const half *up      [[buffer(1)]],
  device half *output        [[buffer(2)]],
  uint gid                   [[thread_position_in_grid]]
) {
  float g = float(gate[gid]);
  float silu_g = g / (1.0f + exp(-g));
  output[gid] = half(silu_g * float(up[gid]));
}

// Backward: given gradOutput and saved gate/up values
// d(silu(g) * u)/dg = u * d(silu(g))/dg = u * (sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g)))
//                   = u * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
// d(silu(g) * u)/du = silu(g)
kernel void swiglu_backward(
  device const float *gradOut [[buffer(0)]],
  device const float *gate    [[buffer(1)]],
  device const float *up      [[buffer(2)]],
  device float *gradGate      [[buffer(3)]],
  device float *gradUp        [[buffer(4)]],
  uint gid                    [[thread_position_in_grid]]
) {
  float g = gate[gid];
  float u = up[gid];
  float go = gradOut[gid];

  float sig = 1.0f / (1.0f + exp(-g));
  float silu_g = g * sig;

  // d/dgate = gradOut * up * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
  gradGate[gid] = go * u * sig * (1.0f + g * (1.0f - sig));
  // d/dup = gradOut * silu(gate)
  gradUp[gid] = go * silu_g;
}

kernel void swiglu_backward_f16(
  device const half *gradOut  [[buffer(0)]],
  device const half *gate     [[buffer(1)]],
  device const half *up       [[buffer(2)]],
  device half *gradGate       [[buffer(3)]],
  device half *gradUp         [[buffer(4)]],
  uint gid                    [[thread_position_in_grid]]
) {
  float g = float(gate[gid]);
  float u = float(up[gid]);
  float go = float(gradOut[gid]);

  float sig = 1.0f / (1.0f + exp(-g));
  float silu_g = g * sig;

  gradGate[gid] = half(go * u * sig * (1.0f + g * (1.0f - sig)));
  gradUp[gid] = half(go * silu_g);
}
