// smith/shaders/muon.metal
// Muon optimizer element-wise kernels.
// Newton-Schulz matmuls use existing matmul shaders.
// Norm computation and NorMuon scaling done CPU-side via unified memory.

#include <metal_stdlib>
using namespace metal;

// --- Nesterov momentum ---
// momentum_buf' = momentum * momentum_buf + (1 - momentum) * grad
// grad' = grad + momentum * (momentum_buf' - grad)  [Nesterov lookahead]
struct NesterovParams {
  float momentum;
  uint n;
};

kernel void muon_nesterov(
  device float* grad         [[buffer(0)]],
  device float* momentum_buf [[buffer(1)]],
  constant NesterovParams& p [[buffer(2)]],
  uint tid [[thread_position_in_grid]]
) {
  if (tid >= p.n) return;
  float g = grad[tid];
  float m = momentum_buf[tid];
  float m_new = p.momentum * m + (1.0f - p.momentum) * g;
  momentum_buf[tid] = m_new;
  grad[tid] = g + p.momentum * (m_new - g);
}

// --- Newton-Schulz polynomial: B = b*A + c*(A@A) ---
// A and AA (= A@A) precomputed via matmul dispatch.
struct PolyParams {
  float b;
  float c;
  uint n;
};

kernel void muon_ns_poly(
  device const float* A  [[buffer(0)]],
  device const float* AA [[buffer(1)]],
  device float* B        [[buffer(2)]],
  constant PolyParams& p [[buffer(3)]],
  uint tid [[thread_position_in_grid]]
) {
  if (tid >= p.n) return;
  B[tid] = p.b * A[tid] + p.c * AA[tid];
}

// --- Newton-Schulz combine: X = a*X + product ---
struct CombineParams {
  float a;
  uint n;
};

kernel void muon_ns_combine(
  device float* X              [[buffer(0)]],
  device const float* product  [[buffer(1)]],
  constant CombineParams& p   [[buffer(2)]],
  uint tid [[thread_position_in_grid]]
) {
  if (tid >= p.n) return;
  X[tid] = p.a * X[tid] + product[tid];
}

// --- Cautious weight decay + parameter update ---
// mask = (g * param) >= 0
// param -= lr * g + lr * wd * param * mask
struct UpdateParams {
  float lr;
  float wd;
  uint n;
};

kernel void muon_update(
  device float* param      [[buffer(0)]],
  device const float* g    [[buffer(1)]],
  constant UpdateParams& p [[buffer(2)]],
  uint tid [[thread_position_in_grid]]
) {
  if (tid >= p.n) return;
  float pv = param[tid];
  float gv = g[tid];
  float mask = (gv * pv >= 0.0f) ? 1.0f : 0.0f;
  param[tid] = pv - p.lr * gv - p.lr * p.wd * pv * mask;
}
