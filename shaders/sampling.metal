// smith/shaders/sampling.metal
// GPU-side sampling for autoregressive text generation.
// Kernels: argmax, penalty application, top-k masking, multinomial sample.
// Goal: only 4 bytes (one uint32 token index) cross GPU→CPU per token.

#include <metal_stdlib>
using namespace metal;

// ============================================================
// argmax_reduce — parallel reduction to find index of maximum value.
// Each threadgroup produces one winner. For vocabSize > threadgroup_size,
// dispatch a second pass (argmax_reduce_final) over partial results.
// ============================================================

struct ArgmaxPair {
    float value;
    uint index;
};

kernel void argmax_reduce(
    device const float* input       [[buffer(0)]],
    device uint* outIndices         [[buffer(1)]],
    device float* outValues         [[buffer(2)]],
    constant uint& size             [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]],
    uint lid                        [[thread_index_in_threadgroup]],
    uint group_id                   [[threadgroup_position_in_grid]],
    uint group_size                 [[threads_per_threadgroup]])
{
    threadgroup ArgmaxPair shared[256];

    ArgmaxPair best;
    best.value = tid < size ? input[tid] : -INFINITY;
    best.index = tid;
    shared[lid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            ArgmaxPair other = shared[lid + stride];
            if (other.value > shared[lid].value) {
                shared[lid] = other;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        outIndices[group_id] = shared[0].index;
        outValues[group_id] = shared[0].value;
    }
}

// Reduce partial winners from pass 1 to single argmax.
kernel void argmax_reduce_final(
    device const uint* partialIndices   [[buffer(0)]],
    device const float* partialValues   [[buffer(1)]],
    device uint* output                 [[buffer(2)]],
    constant uint& numPartials          [[buffer(3)]],
    uint lid                            [[thread_index_in_threadgroup]],
    uint group_size                     [[threads_per_threadgroup]])
{
    threadgroup ArgmaxPair shared[256];

    ArgmaxPair best;
    if (lid < numPartials) {
        best = { partialValues[lid], partialIndices[lid] };
    } else {
        best = { -INFINITY, 0 };
    }
    shared[lid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (shared[lid + stride].value > shared[lid].value) {
                shared[lid] = shared[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[0] = shared[0].index;
    }
}

// ============================================================
// apply_rep_penalty — repetition penalty on recent token positions.
// If logit > 0: divide by penalty. If logit <= 0: multiply by penalty.
// Modifies logits in-place.
// ============================================================

struct PenaltyParams {
    uint vocabSize;
    uint numRecent;
    float temperature;
    float repPenalty;
};

kernel void apply_rep_penalty(
    device float* logits              [[buffer(0)]],
    device const uint* recentTokens   [[buffer(1)]],
    constant PenaltyParams& p         [[buffer(2)]],
    uint tid                          [[thread_position_in_grid]])
{
    if (tid >= p.numRecent) return;
    uint tokenId = recentTokens[tid];
    if (tokenId >= p.vocabSize) return;

    float val = logits[tokenId];
    logits[tokenId] = val > 0 ? val / p.repPenalty : val * p.repPenalty;
}

// ============================================================
// apply_temperature — multiply all logits by 1/temperature. In-place.
// ============================================================

kernel void apply_temperature(
    device float* logits          [[buffer(0)]],
    constant uint& vocabSize      [[buffer(1)]],
    constant float& invTemp       [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]])
{
    if (tid >= vocabSize) return;
    logits[tid] *= invTemp;
}

// ============================================================
// topk_find_threshold — find the K-th largest value.
// Single-thread insertion sort maintaining K best values.
// O(N) expected, O(NK) worst case. For K=40, N=32K → ~32μs.
// Output: threshold float (the K-th largest value).
// ============================================================

struct TopKParams {
    uint size;
    uint k;
};

kernel void topk_find_threshold(
    device const float* input    [[buffer(0)]],
    device float* threshold      [[buffer(1)]],
    constant TopKParams& p       [[buffer(2)]],
    uint lid                     [[thread_index_in_threadgroup]])
{
    if (lid != 0) return;

    // Maintain sorted array of K best values (descending).
    // Use thread-local array. K capped at 256.
    float topVals[256];
    uint kk = min(p.k, 256u);
    for (uint i = 0; i < kk; i++) topVals[i] = -INFINITY;

    for (uint i = 0; i < p.size; i++) {
        float v = input[i];
        // If v beats the current K-th best, insert it
        if (v > topVals[kk - 1]) {
            topVals[kk - 1] = v;
            // Bubble up to maintain descending order
            for (uint j = kk - 1; j > 0; j--) {
                if (topVals[j] > topVals[j - 1]) {
                    float tmp = topVals[j - 1];
                    topVals[j - 1] = topVals[j];
                    topVals[j] = tmp;
                } else {
                    break;
                }
            }
        }
    }
    threshold[0] = topVals[kk - 1];
}

// Apply top-k mask: set logits below threshold to -INFINITY.
kernel void topk_mask(
    device float* logits         [[buffer(0)]],
    device const float* threshold [[buffer(1)]],
    constant uint& size          [[buffer(2)]],
    uint tid                     [[thread_position_in_grid]])
{
    if (tid >= size) return;
    if (logits[tid] < threshold[0]) {
        logits[tid] = -INFINITY;
    }
}

// ============================================================
// multinomial_sample — sample one token from probabilities.
// Sequential prefix-sum scan to find the CDF bucket.
// Input: prob array (after softmax), random float in [0,1).
// Output: single uint32 token index.
// ============================================================

struct SampleParams {
    uint size;
    float randomValue;
};

kernel void multinomial_sample(
    device const float* probs   [[buffer(0)]],
    device uint* output         [[buffer(1)]],
    constant SampleParams& p    [[buffer(2)]],
    uint lid                    [[thread_index_in_threadgroup]])
{
    if (lid != 0) return;

    float cumulative = 0.0f;
    for (uint i = 0; i < p.size; i++) {
        cumulative += probs[i];
        if (p.randomValue < cumulative) {
            output[0] = i;
            return;
        }
    }
    output[0] = p.size - 1;
}
