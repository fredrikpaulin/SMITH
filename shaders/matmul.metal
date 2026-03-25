// smith/shaders/matmul.metal
// Tiled matrix multiplication using threadgroup shared memory.
// C[M,N] = A[M,K] @ B[K,N]
//
// Optimization strategy:
// 1. Tile-based: each threadgroup computes a TILE_M x TILE_N block of C
// 2. Shared memory: tiles of A and B are loaded into threadgroup memory
// 3. Each thread computes a THREAD_M x THREAD_N sub-tile of the output
// 4. Memory coalescing: threads load sequential addresses
//
// This is the most performance-critical shader in Smith.

#include <metal_stdlib>
using namespace metal;

// Tile dimensions. Tuned for Apple Silicon GPU (M1-M5).
// 32x32 threadgroup, each thread computes 4x4 output elements.
constant uint TILE_M = 32;
constant uint TILE_N = 32;
constant uint TILE_K = 32;
constant uint THREAD_M = 4;
constant uint THREAD_N = 4;
constant uint THREADS_PER_GROUP = (TILE_M / THREAD_M) * (TILE_N / THREAD_N); // 64

struct MatmulParams {
    uint M;
    uint N;
    uint K;
};

kernel void matmul_f32(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float* C              [[buffer(2)]],
    constant MatmulParams& p     [[buffer(3)]],
    threadgroup float* shared    [[threadgroup(0)]],
    uint2 group_id               [[threadgroup_position_in_grid]],
    uint tid_in_group            [[thread_index_in_threadgroup]])
{
    // Threadgroup-local shared memory layout:
    // [0 .. TILE_M*TILE_K) = tile of A
    // [TILE_M*TILE_K .. TILE_M*TILE_K + TILE_K*TILE_N) = tile of B
    threadgroup float* As = shared;
    threadgroup float* Bs = shared + TILE_M * TILE_K;

    uint M = p.M, N = p.N, K = p.K;

    // Which sub-tile this thread is responsible for
    uint thread_row = (tid_in_group / (TILE_N / THREAD_N)) * THREAD_M;
    uint thread_col = (tid_in_group % (TILE_N / THREAD_N)) * THREAD_N;

    // Global row/col of the top-left corner of this threadgroup's tile
    uint row0 = group_id.y * TILE_M;
    uint col0 = group_id.x * TILE_N;

    // Accumulator: THREAD_M x THREAD_N
    float acc[THREAD_M][THREAD_N];
    for (uint i = 0; i < THREAD_M; i++)
        for (uint j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    // Iterate over K dimension in tiles
    uint numTiles = (K + TILE_K - 1) / TILE_K;
    for (uint t = 0; t < numTiles; t++) {
        uint k0 = t * TILE_K;

        // Collaborative load: each thread loads multiple elements
        // Load tile of A [TILE_M x TILE_K]
        uint elems_A = TILE_M * TILE_K;
        for (uint i = tid_in_group; i < elems_A; i += THREADS_PER_GROUP) {
            uint r = i / TILE_K;
            uint c = i % TILE_K;
            uint gr = row0 + r;
            uint gc = k0 + c;
            As[r * TILE_K + c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        // Load tile of B [TILE_K x TILE_N]
        uint elems_B = TILE_K * TILE_N;
        for (uint i = tid_in_group; i < elems_B; i += THREADS_PER_GROUP) {
            uint r = i / TILE_N;
            uint c = i % TILE_N;
            uint gr = k0 + r;
            uint gc = col0 + c;
            Bs[r * TILE_N + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: each thread does THREAD_M x THREAD_N x TILE_K multiply-accumulate
        for (uint k = 0; k < TILE_K; k++) {
            for (uint i = 0; i < THREAD_M; i++) {
                float a_val = As[(thread_row + i) * TILE_K + k];
                for (uint j = 0; j < THREAD_N; j++) {
                    acc[i][j] += a_val * Bs[k * TILE_N + (thread_col + j)];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results to C
    for (uint i = 0; i < THREAD_M; i++) {
        uint gr = row0 + thread_row + i;
        if (gr >= M) continue;
        for (uint j = 0; j < THREAD_N; j++) {
            uint gc = col0 + thread_col + j;
            if (gc >= N) continue;
            C[gr * N + gc] = acc[i][j];
        }
    }
}

// --- Simple matmul (for small matrices or when tiling overhead isn't worth it) ---

kernel void matmul_simple(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float* C              [[buffer(2)]],
    constant MatmulParams& p     [[buffer(3)]],
    uint2 tid                    [[thread_position_in_grid]])
{
    uint row = tid.y;
    uint col = tid.x;
    if (row >= p.M || col >= p.N) return;

    float sum = 0.0f;
    for (uint k = 0; k < p.K; k++) {
        sum += A[row * p.K + k] * B[k * p.N + col];
    }
    C[row * p.N + col] = sum;
}

// --- Batched matmul ---
// C[b,M,N] = A[b,M,K] @ B[b,K,N]
// Grid: (N, M, batchSize)

struct BatchMatmulParams {
    uint M;
    uint N;
    uint K;
    uint batch;
};

kernel void matmul_batched(
    device const float* A          [[buffer(0)]],
    device const float* B          [[buffer(1)]],
    device float* C                 [[buffer(2)]],
    constant BatchMatmulParams& p   [[buffer(3)]],
    uint3 tid                       [[thread_position_in_grid]])
{
    uint col = tid.x;
    uint row = tid.y;
    uint b = tid.z;
    if (row >= p.M || col >= p.N || b >= p.batch) return;

    uint aOff = b * p.M * p.K;
    uint bOff = b * p.K * p.N;
    uint cOff = b * p.M * p.N;

    float sum = 0.0f;
    for (uint k = 0; k < p.K; k++) {
        sum += A[aOff + row * p.K + k] * B[bOff + k * p.N + col];
    }
    C[cOff + row * p.N + col] = sum;
}
