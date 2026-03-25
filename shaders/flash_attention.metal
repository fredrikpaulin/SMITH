// smith/shaders/flash_attention.metal
// Flash Attention forward and backward kernels.
// Implements the FlashAttention-2 algorithm: tiled attention that never
// materializes the full [seqLen, seqLen] score matrix.
//
// Memory: O(n) instead of O(n²). Each threadgroup processes one row-block
// of Q and iterates over all K/V column-blocks, maintaining running softmax
// statistics (online log-sum-exp).
//
// Reference: Dao et al., "FlashAttention-2: Faster Attention with Better
// Parallelism and Work Partitioning" (2023)

#include <metal_stdlib>
using namespace metal;

// Tile sizes tuned for Apple Silicon (32KB threadgroup memory on M1)
// Br = row block size (Q rows per threadgroup)
// Bc = column block size (K/V columns per iteration)
constant uint Br = 32;
constant uint Bc = 32;

struct FlashAttnParams {
  uint N;         // sequence length
  uint d;         // head dimension
  uint numHeads;  // number of attention heads
  float scale;    // 1 / sqrt(d)
  uint causal;    // 1 for causal mask, 0 for no mask
};

// Forward kernel: one threadgroup per (head, Q-row-block) pair.
// Grid: (numHeads, ceil(N/Br), 1)
// Threadgroup: (Bc, 1, 1) — each thread handles one column during K/V iteration
//
// Inputs:
//   Q [numHeads, N, d]  — queries
//   K [numHeads, N, d]  — keys
//   V [numHeads, N, d]  — values
// Outputs:
//   O [numHeads, N, d]  — attention output
//   L [numHeads, N]     — log-sum-exp per row (for backward)
//   M [numHeads, N]     — row-wise max per row (for backward)

kernel void flash_attention_forward(
  device const float* Q   [[buffer(0)]],
  device const float* K   [[buffer(1)]],
  device const float* V   [[buffer(2)]],
  device float* O         [[buffer(3)]],
  device float* L         [[buffer(4)]],  // log-sum-exp
  device float* M_out     [[buffer(5)]],  // row max
  constant FlashAttnParams& p [[buffer(6)]],
  uint3 pos               [[thread_position_in_grid]],
  uint3 tpg_vec           [[threads_per_threadgroup]]
) {
  const uint tpg = tpg_vec.x;
  const uint h = pos.x / tpg;        // which head
  const uint tid = pos.x % tpg;      // thread index within group
  const uint block_row = pos.y;       // which Q row-block
  const uint N = p.N;
  const uint d = p.d;
  const float scale = p.scale;

  // Bounds for this Q row-block
  const uint row_start = block_row * Br;
  const uint row_end = min(row_start + Br, N);
  const uint num_rows = row_end - row_start;
  if (num_rows == 0) return;

  // Pointers into this head's Q, K, V, O
  device const float* Qh = Q + h * N * d;
  device const float* Kh = K + h * N * d;
  device const float* Vh = V + h * N * d;
  device float* Oh = O + h * N * d;
  device float* Lh = L + h * N;
  device float* Mh = M_out + h * N;

  // Threadgroup shared memory:
  // S_block [Br, Bc]   — score tile
  // O_acc [Br, d]      — output accumulator (each thread covers part of d)
  // m_i [Br]           — running row max
  // l_i [Br]           — running row sum of exp
  threadgroup float S_block[Br * Bc];

  // Each thread maintains accumulators for its assigned rows in registers.
  // Since Br rows * d cols can be large, we loop over d in chunks.

  // We process one Q row-block. Each thread handles parts of the computation.
  // Strategy: threads cooperate to compute S = Q_block @ K_block^T, then
  // each thread handles softmax and V accumulation for its assigned rows.

  // Init running max and sum for each row in this block
  threadgroup float m_i[Br];
  threadgroup float l_i[Br];

  // Init output accumulator to zero
  threadgroup float O_acc[Br * 128]; // max headDim = 128

  if (tid < num_rows) {
    m_i[tid] = -INFINITY;
    l_i[tid] = 0.0f;
    for (uint j = 0; j < d; j++) {
      O_acc[tid * d + j] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Iterate over K/V column-blocks
  const uint num_col_blocks = (N + Bc - 1) / Bc;

  for (uint col_block = 0; col_block < num_col_blocks; col_block++) {
    const uint col_start = col_block * Bc;
    const uint col_end = min(col_start + Bc, N);
    const uint num_cols = col_end - col_start;

    // Causal: skip blocks entirely above the diagonal
    if (p.causal && col_start > row_end - 1) break;

    // Compute S_block = Q_block @ K_block^T * scale
    // S_block[i][j] = sum_k Q[row_start+i, k] * K[col_start+j, k] * scale
    // Each thread computes one or more elements of S_block
    for (uint idx = tid; idx < num_rows * num_cols; idx += tpg) {
      uint i = idx / num_cols;
      uint j = idx % num_cols;
      float dot = 0.0f;
      for (uint k = 0; k < d; k++) {
        dot += Qh[(row_start + i) * d + k] * Kh[(col_start + j) * d + k];
      }
      dot *= scale;

      // Apply causal mask: if col > row, set to -inf
      if (p.causal && (col_start + j) > (row_start + i)) {
        dot = -INFINITY;
      }
      S_block[i * Bc + j] = dot;
    }
    // Pad unused entries
    for (uint idx = tid; idx < Br * Bc; idx += tpg) {
      uint i = idx / Bc;
      uint j = idx % Bc;
      if (i >= num_rows || j >= num_cols) {
        S_block[idx] = -INFINITY;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax + V accumulation, per row
    for (uint i = tid; i < num_rows; i += tpg) {
      // Find max of this tile's row
      float m_ij = -INFINITY;
      for (uint j = 0; j < num_cols; j++) {
        m_ij = max(m_ij, S_block[i * Bc + j]);
      }

      // New running max
      float m_new = max(m_i[i], m_ij);

      // Rescale old accumulator
      float alpha = exp(m_i[i] - m_new);
      float l_new = l_i[i] * alpha;

      // Compute exp(s - m_new) and accumulate
      float block_sum = 0.0f;
      for (uint j = 0; j < num_cols; j++) {
        float p_ij = exp(S_block[i * Bc + j] - m_new);
        S_block[i * Bc + j] = p_ij; // reuse S_block to store P
        block_sum += p_ij;
      }

      l_new += block_sum;

      // Rescale old O accumulator and add new V contribution
      for (uint k = 0; k < d; k++) {
        O_acc[i * d + k] = O_acc[i * d + k] * alpha;
        float v_sum = 0.0f;
        for (uint j = 0; j < num_cols; j++) {
          v_sum += S_block[i * Bc + j] * Vh[(col_start + j) * d + k];
        }
        O_acc[i * d + k] += v_sum;
      }

      m_i[i] = m_new;
      l_i[i] = l_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Final normalization: O = O_acc / l_i
  for (uint i = tid; i < num_rows; i += tpg) {
    float inv_l = 1.0f / l_i[i];
    for (uint k = 0; k < d; k++) {
      Oh[(row_start + i) * d + k] = O_acc[i * d + k] * inv_l;
    }
    // Store log-sum-exp and max for backward
    Lh[row_start + i] = l_i[i];
    Mh[row_start + i] = m_i[i];
  }
}

// Backward kernel: computes dQ, dK, dV given dO and the saved L, M stats.
// Recomputes attention weights from Q, K (no stored attention matrix).
// One threadgroup per (head, Q-row-block) — same decomposition as forward.

kernel void flash_attention_backward(
  device const float* Q     [[buffer(0)]],
  device const float* K     [[buffer(1)]],
  device const float* V     [[buffer(2)]],
  device const float* O     [[buffer(3)]],
  device const float* dO    [[buffer(4)]],
  device const float* L     [[buffer(5)]],  // row sums from forward
  device const float* M_in  [[buffer(6)]],  // row maxes from forward
  device float* dQ          [[buffer(7)]],
  device float* dK          [[buffer(8)]],
  device float* dV          [[buffer(9)]],
  constant FlashAttnParams& p [[buffer(10)]],
  uint3 pos               [[thread_position_in_grid]],
  uint3 tpg_vec           [[threads_per_threadgroup]]
) {
  const uint tpg = tpg_vec.x;
  const uint h = pos.x / tpg;
  const uint tid = pos.x % tpg;
  const uint block_row = pos.y;
  const uint N = p.N;
  const uint d = p.d;
  const float scale = p.scale;

  const uint row_start = block_row * Br;
  const uint row_end = min(row_start + Br, N);
  const uint num_rows = row_end - row_start;
  if (num_rows == 0) return;

  device const float* Qh = Q + h * N * d;
  device const float* Kh = K + h * N * d;
  device const float* Vh = V + h * N * d;
  device const float* Oh = O + h * N * d;
  device const float* dOh = dO + h * N * d;
  device const float* Lh = L + h * N;
  device const float* Mh = M_in + h * N;
  device float* dQh = dQ + h * N * d;
  device float* dKh = dK + h * N * d;
  device float* dVh = dV + h * N * d;

  threadgroup float S_block[Br * Bc];

  // Precompute D[i] = sum_k dO[i,k] * O[i,k] for each row
  threadgroup float D_i[Br];
  for (uint i = tid; i < num_rows; i += tpg) {
    float di = 0.0f;
    for (uint k = 0; k < d; k++) {
      di += dOh[(row_start + i) * d + k] * Oh[(row_start + i) * d + k];
    }
    D_i[i] = di;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Initialize dQ accumulator for this block to zero
  threadgroup float dQ_acc[Br * 128]; // max headDim = 128
  for (uint idx = tid; idx < num_rows * d; idx += tpg) {
    dQ_acc[idx] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint num_col_blocks = (N + Bc - 1) / Bc;

  for (uint col_block = 0; col_block < num_col_blocks; col_block++) {
    const uint col_start = col_block * Bc;
    const uint col_end = min(col_start + Bc, N);
    const uint num_cols = col_end - col_start;

    if (p.causal && col_start > row_end - 1) break;

    // Recompute S = Q_block @ K_block^T * scale
    for (uint idx = tid; idx < num_rows * num_cols; idx += tpg) {
      uint i = idx / num_cols;
      uint j = idx % num_cols;
      float dot = 0.0f;
      for (uint k = 0; k < d; k++) {
        dot += Qh[(row_start + i) * d + k] * Kh[(col_start + j) * d + k];
      }
      dot *= scale;
      if (p.causal && (col_start + j) > (row_start + i)) {
        dot = -INFINITY;
      }
      S_block[i * Bc + j] = dot;
    }
    for (uint idx = tid; idx < Br * Bc; idx += tpg) {
      uint i = idx / Bc;
      uint j = idx % Bc;
      if (i >= num_rows || j >= num_cols) {
        S_block[idx] = -INFINITY;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Recompute P = softmax(S) using saved M and L
    for (uint idx = tid; idx < num_rows * num_cols; idx += tpg) {
      uint i = idx / num_cols;
      uint j = idx % num_cols;
      float p_ij = exp(S_block[i * Bc + j] - Mh[row_start + i]) / Lh[row_start + i];
      S_block[i * Bc + j] = p_ij;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1: dV[j,k] += sum_i P[i,j] * dO[i,k]
    // Each thread handles a unique (j,k) pair — no within-threadgroup race
    for (uint idx = tid; idx < num_cols * d; idx += tpg) {
      uint j = idx / d;
      uint k = idx % d;
      float sum = 0.0f;
      for (uint i = 0; i < num_rows; i++) {
        sum += S_block[i * Bc + j] * dOh[(row_start + i) * d + k];
      }
      dVh[(col_start + j) * d + k] += sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute dS[i,j] = P[i,j] * (dP[i,j] - D[i]) * scale
    // Store in S_block (overwrites P — we're done with P after dV above)
    for (uint idx = tid; idx < num_rows * num_cols; idx += tpg) {
      uint i = idx / num_cols;
      uint j = idx % num_cols;
      float dp_ij = 0.0f;
      for (uint k = 0; k < d; k++) {
        dp_ij += dOh[(row_start + i) * d + k] * Vh[(col_start + j) * d + k];
      }
      S_block[i * Bc + j] = S_block[i * Bc + j] * (dp_ij - D_i[i]) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: dQ[i,k] += sum_j dS[i,j] * K[j,k]
    // Each thread handles a unique (i,k) pair — no race
    for (uint idx = tid; idx < num_rows * d; idx += tpg) {
      uint i = idx / d;
      uint k = idx % d;
      float sum = 0.0f;
      for (uint j = 0; j < num_cols; j++) {
        sum += S_block[i * Bc + j] * Kh[(col_start + j) * d + k];
      }
      dQ_acc[i * d + k] += sum;
    }

    // Step 4: dK[j,k] += sum_i dS[i,j] * Q[i,k]
    // Each thread handles a unique (j,k) pair — no within-threadgroup race
    for (uint idx = tid; idx < num_cols * d; idx += tpg) {
      uint j = idx / d;
      uint k = idx % d;
      float sum = 0.0f;
      for (uint i = 0; i < num_rows; i++) {
        sum += S_block[i * Bc + j] * Qh[(row_start + i) * d + k];
      }
      dKh[(col_start + j) * d + k] += sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write dQ accumulator to global memory
  for (uint idx = tid; idx < num_rows * d; idx += tpg) {
    uint i = idx / d;
    uint k = idx % d;
    dQh[(row_start + i) * d + k] += dQ_acc[idx];
  }
}

// ============================================================
// f16 variants — half I/O, f32 internal computation
// L and M are always f32 (softmax stats need full precision)
// ============================================================

kernel void flash_attention_forward_f16(
  device const half* Q   [[buffer(0)]],
  device const half* K   [[buffer(1)]],
  device const half* V   [[buffer(2)]],
  device half* O         [[buffer(3)]],
  device float* L        [[buffer(4)]],
  device float* M_out    [[buffer(5)]],
  constant FlashAttnParams& p [[buffer(6)]],
  uint3 pos              [[thread_position_in_grid]],
  uint3 tpg_vec          [[threads_per_threadgroup]]
) {
  const uint tpg = tpg_vec.x;
  const uint h = pos.x / tpg;
  const uint tid = pos.x % tpg;
  const uint block_row = pos.y;
  const uint N = p.N, d = p.d;
  const float scale = p.scale;

  const uint row_start = block_row * Br;
  const uint row_end = min(row_start + Br, N);
  const uint num_rows = row_end - row_start;
  if (num_rows == 0) return;

  device const half* Qh = Q + h * N * d;
  device const half* Kh = K + h * N * d;
  device const half* Vh = V + h * N * d;
  device half* Oh = O + h * N * d;
  device float* Lh = L + h * N;
  device float* Mh = M_out + h * N;

  threadgroup float S_block[Br * Bc];
  threadgroup float m_i[Br];
  threadgroup float l_i[Br];
  threadgroup float O_acc[Br * 128];

  if (tid < num_rows) {
    m_i[tid] = -INFINITY;
    l_i[tid] = 0.0f;
    for (uint j = 0; j < d; j++) O_acc[tid * d + j] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint num_col_blocks = (N + Bc - 1) / Bc;
  for (uint col_block = 0; col_block < num_col_blocks; col_block++) {
    const uint col_start = col_block * Bc;
    const uint col_end = min(col_start + Bc, N);
    const uint num_cols = col_end - col_start;
    if (p.causal && col_start > row_end - 1) break;

    for (uint idx = tid; idx < num_rows * num_cols; idx += tpg) {
      uint i = idx / num_cols, j = idx % num_cols;
      float dot = 0.0f;
      for (uint k = 0; k < d; k++) dot += float(Qh[(row_start + i) * d + k]) * float(Kh[(col_start + j) * d + k]);
      dot *= scale;
      if (p.causal && (col_start + j) > (row_start + i)) dot = -INFINITY;
      S_block[i * Bc + j] = dot;
    }
    for (uint idx = tid; idx < Br * Bc; idx += tpg) {
      uint i = idx / Bc, j = idx % Bc;
      if (i >= num_rows || j >= num_cols) S_block[idx] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < num_rows; i += tpg) {
      float m_ij = -INFINITY;
      for (uint j = 0; j < num_cols; j++) m_ij = max(m_ij, S_block[i * Bc + j]);
      float m_new = max(m_i[i], m_ij);
      float alpha = exp(m_i[i] - m_new);
      float l_new = l_i[i] * alpha;
      float block_sum = 0.0f;
      for (uint j = 0; j < num_cols; j++) {
        float p_ij = exp(S_block[i * Bc + j] - m_new);
        S_block[i * Bc + j] = p_ij;
        block_sum += p_ij;
      }
      l_new += block_sum;
      for (uint k = 0; k < d; k++) {
        O_acc[i * d + k] = O_acc[i * d + k] * alpha;
        float v_sum = 0.0f;
        for (uint j = 0; j < num_cols; j++) v_sum += S_block[i * Bc + j] * float(Vh[(col_start + j) * d + k]);
        O_acc[i * d + k] += v_sum;
      }
      m_i[i] = m_new;
      l_i[i] = l_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = tid; i < num_rows; i += tpg) {
    float inv_l = 1.0f / l_i[i];
    for (uint k = 0; k < d; k++) Oh[(row_start + i) * d + k] = half(O_acc[i * d + k] * inv_l);
    Lh[row_start + i] = l_i[i];
    Mh[row_start + i] = m_i[i];
  }
}

kernel void flash_attention_backward_f16(
  device const half* Q    [[buffer(0)]],
  device const half* K    [[buffer(1)]],
  device const half* V    [[buffer(2)]],
  device const half* O    [[buffer(3)]],
  device const half* dO   [[buffer(4)]],
  device const float* L   [[buffer(5)]],
  device const float* M_in [[buffer(6)]],
  device half* dQ         [[buffer(7)]],
  device half* dK         [[buffer(8)]],
  device half* dV         [[buffer(9)]],
  constant FlashAttnParams& p [[buffer(10)]],
  uint3 pos              [[thread_position_in_grid]],
  uint3 tpg_vec          [[threads_per_threadgroup]]
) {
  const uint tpg = tpg_vec.x;
  const uint h = pos.x / tpg;
  const uint tid = pos.x % tpg;
  const uint block_row = pos.y;
  const uint N = p.N, d = p.d;
  const float scale = p.scale;

  const uint row_start = block_row * Br;
  const uint row_end = min(row_start + Br, N);
  const uint num_rows = row_end - row_start;
  if (num_rows == 0) return;

  device const half* Qh = Q + h * N * d;
  device const half* Kh = K + h * N * d;
  device const half* Vh = V + h * N * d;
  device const half* Oh = O + h * N * d;
  device const half* dOh = dO + h * N * d;
  device const float* Lh = L + h * N;
  device const float* Mh = M_in + h * N;
  device half* dQh = dQ + h * N * d;
  device half* dKh = dK + h * N * d;
  device half* dVh = dV + h * N * d;

  threadgroup float S_block[Br * Bc];
  threadgroup float D_i[Br];
  threadgroup float dQ_acc[Br * 128];

  for (uint i = tid; i < num_rows; i += tpg) {
    float di = 0.0f;
    for (uint k = 0; k < d; k++) di += float(dOh[(row_start + i) * d + k]) * float(Oh[(row_start + i) * d + k]);
    D_i[i] = di;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx = tid; idx < num_rows * d; idx += tpg) dQ_acc[idx] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint num_col_blocks = (N + Bc - 1) / Bc;
  for (uint col_block = 0; col_block < num_col_blocks; col_block++) {
    const uint col_start = col_block * Bc;
    const uint col_end = min(col_start + Bc, N);
    const uint num_cols = col_end - col_start;
    if (p.causal && col_start > row_end - 1) break;

    for (uint idx = tid; idx < num_rows * num_cols; idx += tpg) {
      uint i = idx / num_cols, j = idx % num_cols;
      float dot = 0.0f;
      for (uint k = 0; k < d; k++) dot += float(Qh[(row_start + i) * d + k]) * float(Kh[(col_start + j) * d + k]);
      dot *= scale;
      if (p.causal && (col_start + j) > (row_start + i)) dot = -INFINITY;
      S_block[i * Bc + j] = dot;
    }
    for (uint idx = tid; idx < Br * Bc; idx += tpg) { uint i = idx / Bc, j = idx % Bc; if (i >= num_rows || j >= num_cols) S_block[idx] = -INFINITY; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < num_rows * num_cols; idx += tpg) {
      uint i = idx / num_cols, j = idx % num_cols;
      S_block[i * Bc + j] = exp(S_block[i * Bc + j] - Mh[row_start + i]) / Lh[row_start + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < num_cols * d; idx += tpg) {
      uint j = idx / d, k = idx % d;
      float sum = 0.0f;
      for (uint i = 0; i < num_rows; i++) sum += S_block[i * Bc + j] * float(dOh[(row_start + i) * d + k]);
      dVh[(col_start + j) * d + k] = half(float(dVh[(col_start + j) * d + k]) + sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < num_rows * num_cols; idx += tpg) {
      uint i = idx / num_cols, j = idx % num_cols;
      float dp_ij = 0.0f;
      for (uint k = 0; k < d; k++) dp_ij += float(dOh[(row_start + i) * d + k]) * float(Vh[(col_start + j) * d + k]);
      S_block[i * Bc + j] = S_block[i * Bc + j] * (dp_ij - D_i[i]) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < num_rows * d; idx += tpg) {
      uint i = idx / d, k = idx % d;
      float sum = 0.0f;
      for (uint j = 0; j < num_cols; j++) sum += S_block[i * Bc + j] * float(Kh[(col_start + j) * d + k]);
      dQ_acc[i * d + k] += sum;
    }

    for (uint idx = tid; idx < num_cols * d; idx += tpg) {
      uint j = idx / d, k = idx % d;
      float sum = 0.0f;
      for (uint i = 0; i < num_rows; i++) sum += S_block[i * Bc + j] * float(Qh[(row_start + i) * d + k]);
      dKh[(col_start + j) * d + k] = half(float(dKh[(col_start + j) * d + k]) + sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint idx = tid; idx < num_rows * d; idx += tpg) {
    uint i = idx / d, k = idx % d;
    dQh[(row_start + i) * d + k] = half(float(dQh[(row_start + i) * d + k]) + dQ_acc[idx]);
  }
}
