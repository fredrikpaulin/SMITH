# Changelog

## 0.12.0 — Phase 12: KV Cache for GGUF Models (2026-03-25)

### Added

- **KV cache management** (`src/gguf_cache.js`) — Pre-allocated fixed-size `[kvHeads, maxSeqLen, headDim]` buffers per layer with position tracking. `createGGUFCache`, `resetCache`, `cacheAppend` (single token), `cachePrefill` (full prompt), `cacheSlice` (extract active range). `repeatKV` for GQA head repetition.
- **Cached Llama forward** — `forwardLlamaCachedPrefill` processes the full prompt in one pass using flash attention (causal) and populates the cache. `forwardLlamaCachedDecode` processes a single token using matmul+softmax attention against cached K/V. Both support GQA.
- **`generateGGUF`** — High-level generation entry point. Prefill phase processes the prompt via flash attention, then decode phase generates tokens one-by-one with cached K/V. Supports temperature, top-k, top-p, repetition penalty, EOS stop, and `onToken` callback.
- **`loadGGUF` updated** — Return object now includes `createCache`, `forwardPrefill`, `forwardDecode`, `generate`, and `resetCache` alongside the existing `forward`.
- **Tests** (`tests/gguf_cache.test.js`) — Cache allocation, append, prefill, slice, GQA head repetition, cached forward shapes and finiteness, prefill-vs-decode consistency, generateGGUF with maxTokens/EOS/callbacks/determinism, weight-tied model support.

## 0.11.0 — Phase 11: GPU RoPE, RMSNorm, SwiGLU (2026-03-25)

### Added

- **RoPE Metal shader** (`shaders/rope.metal`) — GPU-accelerated Rotary Position Embeddings. One thread per (position, frequency-pair) with configurable `startPos` offset for KV cache compatibility. Forward and backward kernels. Cos/sin tables stay f32 for precision even with f16 input. f16 variant included.
- **RMSNorm Metal shader** (`shaders/rmsnorm.metal`) — Llama-style RMS normalization on GPU. One threadgroup per row with parallel reduction for sum-of-squares. f32 accumulator for reduction in both f32 and f16 variants. Forward and backward kernels (backward computes `gradInput` on GPU, `gradGamma` via CPU accumulation).
- **SwiGLU Metal shader** (`shaders/swiglu.metal`) — Fused `silu(gate) * up` in a single elementwise kernel, eliminating one intermediate buffer vs separate SiLU + multiply. Forward and backward with analytic SiLU derivative. f16 variant included.
- **JS dispatch files** (`src/ops/rope.js`, `src/ops/rmsnorm.js`, `src/ops/swiglu.js`) — GPU dispatch with dtype-aware kernel selection via `k()`. `precomputeRoPE(dim, maxSeqLen, freqBase)` builds cos/sin frequency tables.
- **Autograd integration** (`src/autograd.js`) — `rope()`, `rmsNorm()`, `swiglu()` as autograd-aware ops with backward closures. `precomputeRoPE` re-exported.
- **GGUF loader updated** (`src/gguf_loader.js`) — `applyRoPE`, `rmsNorm`, and `swiGLUForward` now dispatch to GPU shaders instead of CPU loops. `precomputeRoPE` imported from `ops/rope.js`.
- **Tests** (`tests/gpu_ops.test.js`) — CPU reference implementations for all three ops. Forward correctness, position offset, identity at pos-0, backward inversion (RoPE), numerical gradient checks (RMSNorm, SwiGLU gate/up), autograd pipeline test (rmsNorm → matmul → swiglu → matmul with full backward).

## 0.10.0 — Phase 10: Convolutions (2026-03-25)

### Added

- **Conv2d** (`shaders/conv2d.metal`, `src/ops/conv2d.js`) — 2D convolution with groups, dilation, stride, padding support. Forward kernel dispatches one thread per output element. Separate backward kernels for input gradient (transposed convolution), weight gradient (accumulation over batch and spatial), and bias gradient (channel-wise sum). NCHW layout throughout.
- **Pool2d** (`shaders/pool2d.metal`, `src/ops/pool2d.js`) — Max pooling with argmax index tracking for backward pass. Average pooling with count-based divisor for padded regions. Forward and backward for both. Backward avg pool uses one thread per input element, iterating output positions.
- **Batch normalization** (`shaders/batchnorm.metal`, `src/ops/batchnorm.js`) — Training mode computes batch mean/variance per channel, normalizes, updates running stats. Inference mode uses running stats. Backward computes gradInput, gradGamma, gradBeta in a single kernel.
- **Autograd integration** (`src/autograd.js`) — `conv2d()`, `maxPool2d()`, `avgPool2d()`, `batchnorm()` as autograd-aware ops with backward closures. Batchnorm supports training and inference modes.
- **Helper exports** — `createBatchNorm()`, `convOutputSize()`, `poolOutputSize()` for building CNN architectures.
- **Tests** (`tests/conv.test.js`) — CPU reference implementations for conv2d, maxpool2d, avgpool2d. Forward correctness, backward shape verification, gradient scatter checks, numerical gradient verification, autograd pipeline (conv→relu→pool), batchnorm normalization and running stats.

## 0.9.0 — Phase 9: GGUF Import (2026-03-25)

### Added

- **GGUF parser** (`src/gguf.js`) — Full binary parser for GGUF v2/v3. Reads header, metadata KV pairs (all types), tensor info, aligned data section. Supports GGML types: F32, F16, BF16, Q4_0, Q4_1, Q8_0, Q8_1.
- **Dequantization** (`src/gguf.js`) — Block dequantizers for Q4_0, Q4_1, Q8_0. F16/BF16 decode to f32.
- **Architecture mapping** (`src/gguf_loader.js`) — Weight name maps for Llama, Phi, GPT-2. Handles weight transposition (GGUF [outDim, inDim] → Smith [inDim, outDim]).
- **Llama forward** (`src/gguf_loader.js`) — RMSNorm, RoPE, SwiGLU, GQA, flash attention, weight-tied output head.
- **RoPE** — Precomputed cos/sin tables with configurable frequency base.
- **RMSNorm** — Llama-style normalization (no mean subtraction, no beta).
- **SwiGLU FFN** — Gated FFN with SiLU activation (gate, up, down projections).
- **GQA** — KV head repetition for grouped query attention.
- **Q8 matmul** (`shaders/matmul_q8.metal`, `src/ops/quantize.js`) — 8-bit quantized matmul with fp16 per-block scale.
- **`loadGGUF(path)`** — Top-level async loader: parse → config → model → weights → forward function.
- **Tests** (`tests/gguf.test.js`) — Synthetic GGUF builder, parser verification, dequantization, config extraction, weight mapping, RoPE, RMSNorm.

## 0.8.0 — Phase 8: Mixed Precision f16 Compute (2026-03-25)

### Added

- **f16 Metal shader variants** — All compute shaders now have `_f16` variants: elementwise (add, sub, mul, div, scale, fill, neg), activations (relu, gelu, silu, sigmoid, tanh, exp, log, sqrt + backward), matmul (simple, tiled, batched), softmax, layernorm (forward + backward), reduce (sum, max, axis variants), flash attention (forward + backward), broadcast ops, dtype cast kernels (`cast_f32_to_f16`, `cast_f16_to_f32`). All f16 variants use half I/O with f32 accumulators for numerical stability.
- **Dtype-based kernel dispatch** (`src/dispatch.js`) — `k(baseName, dtype)` appends `_f16` suffix when dispatching half-precision tensors. All op files updated to use this.
- **Cast op** (`src/ops/cast.js`) — `cast(tensor, targetDtype)` for f32↔f16 conversion via GPU kernels.
- **f16 mode toggle** (`src/f16mode.js`) — `f16Mode(enabled)` global toggle, `defaultDtype()` returns `'f16'` when enabled.
- **Dynamic loss scaler** (`src/f16mode.js`) — `createLossScaler(opts)` for mixed precision training. Starts high (2^16), halves on NaN/Inf, doubles after consecutive good steps.
- **Tensor setValue** (`src/tensor.js`) — `setValue(t, index, value)` handles f16 encode/decode.
- **Tests** (`tests/f16.test.js`) — f16 tensor creation, cast roundtrip, elementwise ops, matmul (accuracy vs f32), activations, softmax, layernorm, reduce, f16Mode toggle, loss scaler (basic operation, NaN detection, min scale floor).

### Fixed

- **Layernorm backward CPU loop** — grad_gamma/grad_beta accumulation now uses f32 accumulators with `getValue`/`setValue`, handling f16 tensors correctly.
- **Flash attention L/M stats** — Always allocated as f32 regardless of input dtype (reduction stats need full precision).

## 0.7.0 — Phase 7: Flash Attention (2026-03-25)

### Added

- **Flash attention Metal shader** (`shaders/flash_attention.metal`) — Forward and backward kernels implementing FlashAttention-2. Tiled attention with online softmax (running max + sum), never materializes the full [seqLen, seqLen] score matrix. Causal masking applied per-tile with early-exit. Backward recomputes attention weights from saved log-sum-exp stats. Tile sizes Br=Bc=32 for Apple Silicon.
- **Flash attention dispatch** (`src/ops/flash_attention.js`) — `flashAttentionForward` and `flashAttentionBackward` with multi-buffer dispatch.
- **Autograd** — `flashAttention(q, k, v, causal?)` with full backward support.
- **Neural network** — `multiHeadAttentionFlash`, `transformerBlockFlash` replacing decomposed attention with single fused op.
- **Model** — `forwardFlash` for full model forward pass using flash attention.
- **Tests** (`tests/flash_attention.test.js`) — numerical equivalence, gradient correctness, CPU reference backward, causal masking, model-level equivalence.

### Fixed

- **Flash backward missing threadgroup_barrier** — Added barrier between dV computation (reads P from S_block) and dS computation (overwrites S_block). Without it, threads could corrupt P values still being read.
- **`T.zeros()` not zeroing recycled pool buffers** — The buffer pool returns stale buffers from previous operations. `zeros()` now calls `data.fill(0)` instead of relying on Metal allocation zeroing.
- **`addGrad` storing non-contiguous gradient views** — `gpuTranspose` returns a view with different physical layout. `addGrad` now calls `T.contiguous(grad)` before storing, ensuring `.data` always matches the logical shape. This was the root cause of dK gradient failures: physically-transposed data appeared sign-flipped when compared element-by-element.

## 0.6.0 — Phase 6: Safetensors / Weight Loading (2026-03-25)

### Added

- **Safetensors parser** (`src/safetensors.js`) — `parseSafetensors` reads the binary format (8-byte LE u64 header length + JSON header + raw tensor data). `readTensor` extracts a named tensor as a typed array. `listTensors` enumerates all tensor metadata. No dependencies — parsed with `DataView` and `TextDecoder`.
- **GPT-2 weight loader** (`src/safetensors.js`) — `loadGPT2Safetensors` loads a GPT-2 safetensors file, infers model config (vocabSize, dim, maxSeqLen, numLayers, numHeads) from tensor shapes, creates a Smith model, and copies weights into GPU tensors via unified memory. `mapGPT2Weights` splits the fused `c_attn` projection `[dim, 3*dim]` into separate Q/K/V `[dim, dim]` weights. Handles f16→f32 conversion. `lm_head.weight` is skipped (weight-tied with `wte` in GPT-2).
- **Safetensors export** (`src/safetensors.js`) — `exportSafetensors` serializes a Smith model to safetensors format, re-fusing Q/K/V projections back into `c_attn`. `saveSafetensors` writes the buffer to disk.
- **Tests** (`tests/safetensors.test.js`) — parsing (single/multiple tensors, metadata skip, oversized header rejection), tensor reading (correct offsets, missing tensor error, empty tensor), round-trip (build→parse→read data integrity), GPT-2 weight mapping (c_attn split into Q/K/V with correct values), export (tensor names/shapes, Q/K/V fusion, full export→load weight preservation).

## 0.5.0 — Phase 5: KV Cache (2026-03-25)

### Added

- **Cached multi-head attention** (`src/nn.js`) — `multiHeadAttentionCached` processes a single token and appends K/V to a running cache. `catAlongAxis1` concatenates cached tensors along the sequence dimension. No causal mask needed — Q has length 1, all cached positions are valid.
- **Cached transformer block** (`src/nn.js`) — `transformerBlockCached` wraps cached attention with pre-norm layernorm, FFN, and residual connections.
- **Cached forward pass** (`src/model.js`) — `forwardCached(model, tokenId, position, kvCaches)` embeds a single token at an absolute position, runs through all cached blocks, returns logits and updated caches.
- **Cached generation** (`src/generate.js`) — `generateCached(model, promptIds, config)` processes the prompt token-by-token to fill the cache (prefill), then generates one token at a time using the cache (decode). O(1) compute per new token instead of O(n).
- **Tests** (`tests/kvcache.test.js`) — cached attention shape/growth, cached forward logits shape, cache growth over positions, cached generation (token output, determinism), equivalence between cached and non-cached output at temperature=0, onToken early stopping.

### Fixed

- **Reshape on non-contiguous tensors** (`src/ops/reshape.js`) — reshape was purely virtual (just swapped shape/strides) without checking contiguity. When a transpose view was reshaped, the new strides didn't match the actual data layout. This caused `multiHeadAttention` to incorrectly interleave head outputs when seqLen > 1 (`A.reshape(A.transpose(attnOut, [1,0,2]), [seqLen, dim])` produced garbage for seqLen > 1). Fix: reshape now enforces contiguity via `T.contiguous()` when the input has non-standard strides. This is the same class of bug as the matmul transpose fix from Phase 2.

## 0.4.0 — Phase 4: Inference + Polish (2026-03-25)

### Added

- **Text generation** (`src/generate.js`) — autoregressive generation with temperature, top-k, top-p (nucleus), and repetition penalty sampling. Full-context forward (no KV cache). `generate()` and `topKPredictions()`.
- **Checkpoints** (`src/checkpoint.js`) — `saveCheckpoint` packs all model weights into a flat binary + JSON manifest. `loadCheckpoint` recreates the model and copies weights into GPU tensors.
- **Q4 quantized matmul** (`shaders/matmul_q4.metal`, `src/ops/quantize.js`) — 4-bit weight quantization with per-group (32 elements) scale and zero-point. `quantizeQ4` packs f32 weights to ~4.6 bits/weight. `matmulQ4` dispatches the dequant-fused matmul on GPU.
- **Benchmark suite** (`bench/bench.js`) — matmul, softmax, layernorm, elementwise, Q4 matmul, and GPT forward pass benchmarks with median/mean/min timing.
- **Documentation** (`docs/quickstart.md`, `docs/api.md`) — quick start guide and full API reference.
- **Tests** (`tests/inference.test.js`) — sampling utilities, generation (token output, deterministic with temp=0), checkpoint save/load roundtrip, Q4 quantization accuracy and compression ratio.

## 0.3.0 — Phase 3: Transformer (2026-03-25)

### Added

- **Softmax shader** (`shaders/softmax.metal`) — numerically stable softmax along last dimension. Per-row threadgroup with shared memory for max-reduction and sum-reduction.
- **Layernorm shader** (`shaders/layernorm.metal`) — forward (normalize + scale + shift) and backward (dx, dgamma, dbeta) kernels. Saves xhat for backward pass.
- **Softmax op** (`src/ops/softmax.js`) — GPU dispatch wrapper.
- **Layernorm op** (`src/ops/layernorm.js`) — forward returns `{ out, xhat }`, backward computes all three gradients.
- **Sub/Div ops** (`src/ops/sub.js`, `src/ops/div.js`) — element-wise with broadcasting support.
- **Autograd ops** — `softmax` (backward: `s * (grad - sum(grad * s))`), `layernorm` (backward via GPU kernel), `crossEntropy` (CPU forward, GPU softmax backward).
- **Neural network module** (`src/nn.js`) — `createLinear`, `linear`, `createCausalMask`, `scaledDotProductAttention`, `createMultiHeadAttention`, `multiHeadAttention`, `createTransformerBlock`, `transformerBlock`. Pre-norm architecture (GPT-2 style).
- **Model module** (`src/model.js`) — `createModel` (configs: tiny/small/medium), `forward` (embedding + blocks + layernorm + weight-tied head), `modelParams`, `modelInfo`.
- **Tokenizer** (`src/tokenizer.js`) — BPE tokenizer ported from TinyFormer. `train`, `encode`, `decode`, `save`, `load`.
- **Tests** (`tests/transformer.test.js`) — softmax (correctness, stability, backward), layernorm (zero mean/unit variance, backward), crossEntropy (loss value, gradient), tokenizer (roundtrip, compression), GPT forward shape check, GPT training convergence over 10 steps.

### Fixed

- **`inverseAxes` crash** — removed dead code branch that accessed `.length` on undefined.
- **`transposeVar` with no axes** — now resolves default axes eagerly so backward closure has concrete values. Fixes crash when weight-tied output head does `A.transpose(tokenWeight)`.

## 0.2.0 — Phase 2: Optimizer + Training (2026-03-25)

### Added

- **Fused AdamW shader** (`shaders/adam.metal`) — single kernel handles moment updates, bias correction, Adam step, and decoupled weight decay per element. One GPU dispatch per parameter tensor.
- **Optimizer module** (`src/optim.js`) — `createAdamW` allocates GPU-resident moment buffers, `adamwStep` dispatches the fused kernel per parameter. Cosine LR schedule with linear warmup (`createSchedule`, `getLr`). Gradient clipping by global norm (`clipGradNorm`) reads grads via unified memory.
- **Training test** (`tests/optim.test.js`) — unit tests for schedule, grad clipping, and single-step weight update. End-to-end MLP test: 2-layer net learns `y = sum(x)` over 50 steps, verifying the full forward → backward → clip → AdamW → repeat pipeline converges.

### Fixed

- **Contiguous copy** (`src/tensor.js`) — added `isContiguous()` and `contiguous()`. Virtual transpose produced strided views that matmul shaders read as contiguous memory, causing incorrect gradients. Matmul now enforces contiguous inputs.
- **Async test callbacks** — pool.test.js and tensor.test.js used `await import()` in non-async test functions.

## 0.1.0 — Phase 1: Foundation (2026-03-25)

### Added

- **Native Metal bridge** (`native/gpu_bridge.m`) — Objective-C wrapper exposing ~20 C functions via flat API. Buffer allocation (shared/private), shader loading (precompiled metallib or runtime source), compute dispatch (sync/async), device info queries.
- **FFI device layer** (`src/device.js`) — bun:ffi bindings to libsmith.dylib. Lazy shader library loading, pipeline cache by kernel name, zero-copy `viewBuffer()` for typed array views into GPU memory.
- **Tensor module** (`src/tensor.js`) — GPU-backed tensors as plain objects `{ buffer, data, shape, strides, dtype, size }`. Factory functions: `tensor`, `zeros`, `ones`, `full`, `rand`, `randn`, `scalar`. Shape utilities ported from TinyFormer: `computeStrides`, `shapeSize`, `broadcastShapes`.
- **Dtype module** (`src/dtype.js`) — IEEE 754 half-precision f16 encode/decode, ported from TinyFormer's optimized.js. Bulk conversion functions `float32ToFloat16`, `float16ToFloat32`.
- **Buffer pool** (`src/pool.js`) — Power-of-2 bucketed free-list for GPU buffer recycling. Separate pools for shared/private storage modes. `poolAlloc`, `poolFree`, `poolStats`, `poolDrain`.
- **Metal shaders:**
  - `elementwise.metal` — add, sub, mul, div, scale, neg, fill, fused add+relu. Broadcasting variants with shape/stride params.
  - `activation.metal` — relu, gelu, silu, sigmoid, tanh (forward + backward). Exp, log, sqrt forwards.
  - `matmul.metal` — tiled GEMM (32x32 tiles, threadgroup shared memory, 4x4 per-thread sub-tiles), simple fallback for small matrices, batched variant.
  - `reduce.metal` — parallel sum/max reduction (full and per-axis).
- **Dispatch helper** (`src/dispatch.js`) — translates op-level calls to Metal command encoding. Param builders for matmul, broadcast, axis reduce, scale.
- **Op modules** (`src/ops/`) — `add.js`, `mul.js`, `matmul.js`, `relu.js`, `gelu.js`, `reduce.js`, `transpose.js` (virtual), `reshape.js` (virtual).
- **Autograd** (`src/autograd.js`) — DAG-based reverse-mode AD ported from TinyFormer. Variables with `_deps`/`_backward`, topological sort backward, broadcast-aware `addGrad`. Ops: add, sub, mul, matmul, scale, neg, relu, gelu, sum, reshape, transpose, embedding.
- **Public API** (`src/index.js`) — flat namespace re-exporting all creation, autograd, and utility functions.
- **Tests** — tensor creation, matmul correctness + gradient check, autograd chain rule + gradient accumulation, buffer pool, f16 roundtrip.
- **Build script** (`build.sh`) — compiles native bridge + Metal shaders in two commands.
