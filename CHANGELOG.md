# Changelog

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
