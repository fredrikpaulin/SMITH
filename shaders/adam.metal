// smith/shaders/adam.metal
// Fused AdamW parameter update.
// One thread per parameter element: updates moments, applies bias correction,
// computes Adam step + decoupled weight decay, writes updated weight in-place.

#include <metal_stdlib>
using namespace metal;

struct AdamParams {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float bc1; // 1 - beta1^t (bias correction)
  float bc2; // 1 - beta2^t
  uint  n;   // number of elements
};

kernel void adamw_step(
  device float *weights [[buffer(0)]],
  device const float *grads [[buffer(1)]],
  device float *m [[buffer(2)]],
  device float *v [[buffer(3)]],
  constant AdamParams &p [[buffer(4)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= p.n) return;

  float g = grads[gid];

  // Update biased moments
  float m_new = p.beta1 * m[gid] + (1.0f - p.beta1) * g;
  float v_new = p.beta2 * v[gid] + (1.0f - p.beta2) * g * g;
  m[gid] = m_new;
  v[gid] = v_new;

  // Bias-corrected moments
  float m_hat = m_new / p.bc1;
  float v_hat = v_new / p.bc2;

  // AdamW update: step + decoupled weight decay
  weights[gid] -= p.lr * (m_hat / (sqrt(v_hat) + p.eps) + p.weight_decay * weights[gid]);
}
