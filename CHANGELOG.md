# Changelog

## 0.31.1 — Muon/Matmul Tensor Leak Fixes, Test Timeout Hardening (2026-04-13)

### Fixed

- **muon.js: newtonSchulz() leaked 20+ tensors per step** — Each of the 5 Newton-Schulz iterations created A, AA, B, product tensors without releasing them. Also leaked the initial normalized copy X after copying the result back into g. With real model sizes (e.g., 1024×1024 weight matrices), this leaks ~80MB/step of GPU buffers that the pool can never recycle.
- **muon.js: stepMuon() leaked gradient copy** — The working gradient copy `g = T.create(...)` was never released after the parameter update. One leaked tensor per Muon parameter per optimizer step.
- **matmul.js: contiguous copies leaked in matmul2d/matmulBatched** — When inputs have non-standard strides (e.g., transposed views), `T.contiguous()` allocates a new tensor for the dispatch. These copies were never released after the GPU operation completed. Particularly impactful in Newton-Schulz where every `matmul2d(transpose(X), X)` created a hidden contiguous copy of the transposed view.
- **sdxl-reference.test.js: CLIP tests timed out at default 5s** — CLIP encoder tests load the full SDXL model and run forward passes. Added 60s timeouts to each CLIP test and the `beforeAll` that imports model/clip/tokenizer modules.
- **gguf_model.test.js: beforeAll timed out under parallel pressure** — Loading a 4GB GGUF file in 30s is tight when `bun test tests/` runs all files concurrently (IO contention from SDXL, Metal shader compilation, muon GPU allocations). Bumped to 120s.

## 0.31.0 — Codebase Sweep: Memory, Correctness, Bun Compliance (2026-04-13)

### Fixed

- **autograd.js: gradient accumulation memory leak** — `addGrad()` replaced `v.grad` with `gpuAdd(v.grad, grad)` without releasing the old tensor. Over repeated backward passes, unreferenced intermediate gradients accumulated. Now disposes old grad before reassignment.
- **autograd.js: crossEntropy in-place mutation** — `crossEntropy()` backward modified the softmax output tensor in-place (`probs.data[i] -= 1`), corrupting results if the same logits were used twice (e.g., gradient checking, loss logging). Now copies probs before mutation.
- **pool.js/tensor.js: buffer mode tracking** — `poolFree()` defaulted to `SHARED` mode, but the tensor struct never stored which mode it was allocated with. When lifecycle's `dispose()` frees a tensor, it couldn't pass the right mode. Tensors now store their allocation mode, and both `release()` and lifecycle `dispose()` pass it through.
- **clip.js: hardcoded ViT-B/32 embedDim** — Visual projection dimension was hardcoded to `CLIP_CONFIGS['ViT-B/32'].embedDim` instead of using the config's `embedDim`. Non-B/32 variants (ViT-B/16, ViT-L/14) would get wrong projection dimensions.
- **quantize.js: boundary conditions** — When all weights in a Q4 group were identical (range=0), scale was 1 but zero-point was 0, producing incorrect quantized values. Now uses mid-point (8) for uniform groups. Partial groups at tensor boundaries also handled explicitly.
- **reshape.js: multiple -1 dimensions** — `reshape([−1, −1, 10])` silently computed wrong dimensions. Now validates that at most one dimension is −1.
- **pool.js: missing poolDrain export** — `poolDrain` was renamed to `poolFlush` in 0.29.3 but never re-exported. Added `poolDrain` as a proper function that resets counters, plus `poolFlush` that clears pools without resetting stats.
- **sdxl-reference.test.js: unawaited resolveModel()** — `resolveModel('sdxl-base')` is async but wasn't awaited, causing `modelDir` to be a Promise object. Path became `[object Promise]/tokenizer/vocab.json`. Added missing `await`.

### Changed

- **models.js: Bun compliance** — `registerModel()` now uses `Bun.write()` instead of `writeFileSync()` (async). Removed `fs/promises` import; atomic renames use `renameSync`.
- **tokenizer.js: Bun compliance** — Replaced all `Buffer.from()`/`.toString()` with `TextEncoder`/`TextDecoder` and `btoa`/`atob` for base64. Zero Node.js Buffer dependency.
- **rmsnorm.js: backward performance** — Replaced `getValue()`/`setValue()` calls in the gradient accumulation loop with direct `.data` array access and bulk `.set()`. Eliminates per-element function call overhead.

## 0.30.1 — TTS Decoder Bug Fixes (2026-03-28)

### Fixed

- **Interleaved talker+predictor generation** — The decode loop ran the talker to completion then the predictor once. Now runs predictor at each talker step with all 16 group embeddings summed back as input. This is required for the talker to receive acoustic feedback.
- **Predictor 2-position prefix** — Code predictor now receives projected talker hidden (position 0) + projected group-0 embedding (position 1) before generating groups 1-15. Previously only saw the hidden state.
- **Causal conv1d right-side trimming** — `conv1dForward` uses symmetric padding. Causal behavior requires trimming `effectiveKernel - 1` samples from the right to prevent future leakage. Was producing wind/static noise.
- **Depthwise conv1d kernel direction** — Manual depthwise implementation used `input[t - k*d]` (convolution order). Fixed to `input[t - (K-1-k)*d]` to match PyTorch's cross-correlation convention where `weight[0]` sees the oldest sample.
- **Dilated convolution support** — `conv1dForward` has no dilation parameter, so residual unit dilations [3, 9] were silently ignored. Added CPU im2col with dilation + GPU GEMM path for dilation > 1.
- **Smith tensor compatibility** — Hidden states stored as plain objects caused crashes in GPU matmul. Added `copyHidden()` to create proper `smith.zeros()` tensors.

## 0.30.0 — Qwen3-TTS Text-to-Speech Example (2026-03-27)

### Added

- **`examples/tts/`** — Full Qwen3-TTS-12Hz-1.7B-Base implementation. Three-stage pipeline: Talker (28-layer transformer, text → group-0 codes), Code Predictor (5-layer transformer, group-0 → groups 1-15), Speech Decoder (dequantize → transformer → vocoder → 24kHz PCM). Zero dependencies — runs entirely on Smith's Metal compute pipeline.
- **`src/ops/conv1d_transpose.js`** + **`shaders/conv1d_transpose.metal`** — Transposed 1D convolution for fractionally-strided upsampling in vocoders.
- **`src/ops/snake.js`** + **`shaders/activation.metal` (snake_forward)** — Snake activation function (`x + sin²(αx)/α`) for audio neural networks.
- **BPE tokenizer** (`examples/tts/tokenizer.js`) — GPT-2-style byte-level BPE, compatible with Qwen tokenizer format.
- **Safetensors BF16 loader** (`examples/tts/model.js`) — Loads 3.86GB main model + 682MB speech tokenizer, maps Python weight names to JS model structs, handles BF16→F32 conversion.
- **QK normalization** — Per-head RMSNorm on Q/K projections before RoPE, as used in Qwen3 attention.
- **WAV encoder + macOS playback** (`examples/tts/audio.js`) — 16-bit PCM WAV encoding, `afplay` integration.

## 0.29.3 — Metal Buffer Pool Fix (2026-03-27)

### Fixed

- **OOM kill persisted after 0.29.2** — The `using()` scope fix returned tensors to the pool, but Metal buffers used in compute dispatches accumulate driver-level retains that are never released through Objective-C reference counting. `poolDrain()` called `CFRelease` on these buffers but the retain count never reached zero, so the Metal allocations persisted. Meanwhile the JS-side pool references were cleared, forcing fresh allocations next step. Net effect: ~928MB/step of leaked GPU memory, OOM around step 120–180.

### Changed

- **Removed `poolDrain`** — Since Metal buffers used in dispatches cannot be freed, draining the pool only leaks memory. Buffers now stay pooled across steps and are reused. GPU memory stabilizes after step 2 (~450MB for 1-layer model).
- **Removed `MAX_PER_BIN` cap** — The per-bucket cap of 16 caused pool overflow, sending buffers to `releaseBuffer` (which couldn't free them). Pool bins are now uncapped so every buffer stays available for reuse.
- **Per-sequence scoping in `train.js`** — Each of the 8 gradient accumulation sequences now runs in its own `using()` scope with gradient retention, reducing peak memory from 8× to 1× intermediate set.
- **Proper `@autoreleasepool` + `CFRetain`/`CFRelease` in `gpu_bridge.m`** — `smith_begin` wraps command buffer creation in `@autoreleasepool` with explicit `CFRetain` to manage lifecycle in C structs (ARC doesn't manage `id` fields in C structs). `smith_end_sync`/`smith_end_async`/`smith_end_timed`/`smith_wait` all use `@autoreleasepool` with explicit `nil` to ensure command buffers are deallocated inside the pool.
- **DAG reference cycle cleanup in `backward()`** — Non-parameter nodes have `_backward`, `_deps`, and `grad` nulled after the backward pass so JS GC can reclaim the computation graph.

### Added

- **`smith_test_release` / `smith_test_release_after_use`** — Self-tests in `gpu_bridge.m` that verify Metal buffer alloc/release works. Run at startup via FFI.
- **`smith_buffer_retain_count`** — Diagnostic function exposing `CFGetRetainCount` for Metal buffers.
- **`gpuAllocatedBytes`** — Exposes `device.currentAllocatedSize` for GPU memory tracking.

## 0.29.2 — Training Memory Leak Fix (2026-03-27)

### Fixed

- **OOM kill during long training runs** — `train.js` never freed intermediate tensors (activations, gradients, Newton-Schulz temporaries) between steps. Memory grew linearly with step count, crashing around step 36 with `--time-budget 600`. Wrapped training step body in `smith.using()` scope so all intermediates are returned to the buffer pool after each step. Model params and optimizer state survive because they were allocated before the scope. Same fix applied to the eval loop (one `using()` per eval step).

## 0.29.1 — Tiled Matmul Threadgroup Memory Fix (2026-03-26)

### Fixed

- **Tiled GEMM kernels produced all-zero output** — `matmul_f32` and `matmul_f16` in `shaders/matmul.metal` declared dynamic threadgroup memory via `[[threadgroup(0)]]`, but the native bridge never called `setThreadgroupMemoryLength:atIndex:` to allocate it. With zero bytes of threadgroup memory, shared tile loads read back zeros, making the entire tiled accumulation produce zero. Replaced with static `threadgroup float As[TILE_M * TILE_K]` / `Bs[TILE_K * TILE_N]` arrays that Metal allocates automatically.
- **Same bug in `reduce_sum`, `reduce_max`, `reduce_sum_f16`, `reduce_max_f16`** — all used `[[threadgroup(0)]]` for their parallel reduction shared memory. Replaced with static `threadgroup float shared[1024]`.
- Training loss was stuck at ln(vocabSize) = 8.3178 because every matmul with all dimensions ≥ 64 (the `TILE_THRESHOLD`) produced zeros — both forward (all-zero logits → uniform softmax) and backward (all-zero gradients → no learning).

### Added

- **`smith_set_threadgroup_memory`** in native bridge (`gpu_bridge.m/.h`) and FFI binding (`device.js`) — for future kernels that need dynamic threadgroup memory allocation.

## 0.29.0 — Autoresearch Example (2026-03-26)

### Added

- **`examples/autoresearch/model.js`** — GPT model ported from Karpathy's autoresearch. Uses Smith's autograd throughout: RoPE, RMSNorm, GQA flash attention with sliding windows, ReluSquared MLP, value embeddings (ResFormer) with sigmoid-gated residual, logit soft-capping via tanh. 3D reshape for flash attention (`[T, nHead*headDim]` → `[nHead, T, headDim]`) with tiled RoPE tables to apply per-head rotations through the 2D rope kernel. Includes `createModel()`, `initWeights()`, `forward()`, `setupOptimizer()` with reference parameter grouping (matrix→Muon, embeddings/scalars→AdamW).
- **`examples/autoresearch/data.js`** — Data loading utilities. Binary uint16 token format, sequential data loader with wraparound, BPB (bits per byte) evaluation metric, and tokenizer training via Smith's built-in BPE.
- **`examples/autoresearch/prepare.js`** — Data preparation CLI. Downloads public domain texts from Project Gutenberg, trains BPE tokenizer, saves tokenized train/val splits.
- **`examples/autoresearch/train.js`** — Training loop with LR warmup/warmdown schedule, Muon momentum ramp, time-budgeted training, gradient accumulation across sequences, and final BPB evaluation.
- **`examples/autoresearch/research.js`** — Experiment runner for the autonomous research loop. Wraps training execution, parses output metrics, logs results to both `experiments.json` and `research_log.md`. Commands: `run`, `last`, `status`, `best`. Colorized TUI output with experiment timeline, improvement tracking, and hit rate statistics.
- **`examples/autoresearch/CLAUDE.md`** — Agent prompt for Claude Code. Describes the setup procedure, experiment loop, rules (what's modifiable vs read-only), and research runner commands. Follows the Karpathy autoresearch pattern: one machine, one file, one metric, never stop. Includes bug detection heuristics (stuck loss, zero gradients, unchanged weights) that instruct the agent to stop and report framework issues instead of blindly iterating. Agent log at `results/agent_log.md` for cross-session observations.
- **`examples/autoresearch/reset.sh`** — Resets autoresearch to clean state: restores train.js/model.js from git, clears results and Claude Code session, optionally deletes experiment branch.
- **Smith API skill** symlinked into autoresearch `.claude/skills/` so the agent has the full API reference without reading source code.
- **Training telemetry** in `train.js` — captures gradient norms and weight norms at diagnostic steps (0, 1, 2, then every 10% of training). Prints `=== TELEMETRY ===` block after the summary with loss trajectory, per-param grad/weight snapshots, weight delta from init, and a WARNING if loss is stuck at ln(vocabSize).
- **Tests** — Model tests (creation, init, forward shape, loss, backward gradients, optimizer step, loss reduction, VE placement, soft-capping bounds, window pattern, GQA forward/backward, T=1 single token, T=seqLen full length, full gradient flow after one optimizer step, training loop with gradient accumulation, all param groups get updates). Data tests (loader shapes, advancement, wraparound, reset, BPB computation, special token handling). Research runner tests (metric parsing, JSON roundtrip, best selection, status computation, markdown format, CLI args).

### Changed

- **Strengthened test suite with finite-difference gradient checks.** Added `numGradCheck` helpers to `autograd.test.js` and `transformer.test.js` that perturb each element ±ε and compare `(L+ - L-) / 2ε` against the analytical GPU backward pass. New tests: relu(64), gelu(64), mul chain(32), matmul(16×16), scale+add chain(64), layernorm input/gamma(4×8), softmax(4×16), cross-entropy(8×32). Also added larger-scale GPU-verified tests to `matmul.test.js` (16×16 identity, non-square [32×64]@[64×16], associativity), `conv1d.test.js` (multi-channel weight/input grad finite-diff with stride 2), and `div_gather.test.js` (256-element round trip, 2D 64×8 round trip, div finite-diff(64), gather backward(128, 32 indices)). Tolerances tuned for f32 GPU accumulation (0.01–0.1 depending on op).

### Fixed

- **AdamW step counter in MuonAdamW** — `_adamwStep` was incrementing once per AdamW group per optimizer step instead of once per step. With 5 AdamW groups, bias correction used step=1→5 on step 1, step=6→10 on step 2, etc. All groups now share the correct step counter, producing consistent bias correction across groups.
- **`research.js` delta variable self-reference** — `printExperiment` referenced `delta` during its own initialization in the template literal. Changed `delta > 0` to `exp.improvement > 0`.

### Known Issues

- **`sliceScalar` in autoresearch model breaks gradient flow** — `residLambdas` and `x0Lambdas` are read as plain JS numbers via `sliceScalar()`, bypassing autograd. The optimizer's AdamW step skips them (`if (!p.grad) continue`), so they remain at their init values (1.0 and 0.1). Fixing requires a differentiable scalar index op or switching from `scale(x, number)` to `mul(x, variable)` with broadcast support.

## 0.28.0 — MuonAdamW Optimizer (2026-03-26)

### Added

- **`src/muon.js`** — MuonAdamW optimizer ported from Karpathy's autoresearch. Hybrid optimizer: Muon for 2D matrix params (Newton-Schulz orthogonalization via polar express), AdamW for everything else. Includes Nesterov momentum, NorMuon variance reduction, and cautious weight decay.
- **`shaders/muon.metal`** — GPU kernels: `muon_nesterov` (Nesterov momentum), `muon_ns_poly` (Newton-Schulz polynomial B = b·A + c·A²), `muon_ns_combine` (X = a·X + product), `muon_update` (cautious weight decay + param update).
- **`createMuonAdamW(groups)`** — Creates optimizer with param groups of kind `'adamw'` or `'muon'`.
- **`muonAdamWStep(opt)`** — Performs one optimizer step across all groups.
- **Tests** — 14 tests covering AdamW groups (update, weight decay, convergence), Muon groups (2D update, finite values, zero grad, weight decay, tall/wide matrices, stability over 20 steps), mixed groups, Newton-Schulz coefficient verification, and skipped params.

## 0.27.0 — Tanh, Sigmoid, ReluSquared Activations (2026-03-26)

### Added

- **`autograd.tanh(a)`** — Hyperbolic tangent with backward: `dA = dOut * (1 - tanh(x)²)`. Uses saved output for backward (no recompute).
- **`autograd.sigmoid(a)`** — Sigmoid activation with backward: `dA = dOut * σ(x) * (1 - σ(x))`. Uses saved output.
- **`autograd.reluSquared(a)`** — Fused `relu(x)²` with backward: `dA = dOut * 2 * max(0, x)`. Single kernel dispatch (no intermediate relu buffer).
- **`shaders/activation.metal`** — Added `relusquared_forward`, `relusquared_backward` kernels (f32 + f16).
- **`src/ops/tanh.js`** — GPU dispatch for tanh forward/backward.
- **`src/ops/sigmoid.js`** — GPU dispatch for sigmoid forward/backward.
- **`src/ops/relusquared.js`** — GPU dispatch for relu-squared forward/backward.
- **Tests** — 20+ tests covering tanh/sigmoid/reluSquared forward values, backward gradients, chaining, 2D tensors, soft-capping composition, and relu-squared equivalence to manual relu+square.

## 0.26.0 — GQA + Sliding Window Flash Attention (2026-03-26)

### Changed

- **`shaders/flash_attention.metal`** — Extended `FlashAttnParams` struct with `numKVHeads` and `windowSize`. All four kernels (f32/f16, forward/backward) now support grouped query attention (GQA) and sliding window masking. KV head mapping via integer division; window mask applied per-element with block-level skip optimization.
- **`src/ops/flash_attention.js`** — `flashAttentionForward` and `flashAttentionBackward` accept options object `{ causal, numKVHeads, windowSize }` with backward-compatible boolean support. Output/gradient shapes respect asymmetric Q vs KV head counts.
- **`src/autograd.js`** — `flashAttention` op accepts options object, passes through to GPU dispatch. Boolean arg still works for backward compat.

### Added

- **Tests** — 15+ tests covering backward compat, sliding window (shape, full context equivalence, distant position isolation, nearby position visibility, backward), GQA (shape, MHA equivalence, backward gradient shapes), and combined GQA+window.

## 0.25.0 — Autograd `div` and Gather/Scatter (2026-03-26)

### Added

- **`autograd.div(a, b)`** — Element-wise division with full backward support. Gradients: `dA = dOut / b`, `dB = -dOut * a / b²`. Chains through existing mul/neg/div GPU ops (no fused kernel needed).
- **`shaders/gather_scatter.metal`** — Three Metal kernels: `gather_forward` (indexed read along any axis), `scatter_add` (atomic accumulation for duplicate indices), `scatter_forward` (non-atomic write, last-write-wins).
- **`src/ops/gather.js`** — GPU dispatch layer for gather and scatter operations. Input treated as `[outer, dimSize, inner]` layout for arbitrary-axis indexing.
- **`autograd.gather(input, axis, indices)`** — Differentiable gather. Backward: scatter-add gradient into input-shaped zero tensor.
- **`autograd.scatter(input, axis, indices, src)`** — Differentiable scatter. Backward: `dSrc = gather(grad, indices)`, `dInput = grad` with scattered positions zeroed.
- **`gpuGatherOp`, `gpuScatterAdd`, `gpuScatterOp`** — Raw GPU ops exported for direct use outside autograd.
- **Tests** — 25+ tests covering div forward/backward/chaining, gather 1D/2D/duplicate indices, scatter-add with accumulation, scatter overwrite, autograd backward for both ops, round-trip correctness, and combined div+gather pipeline.

## 0.24.0 — GPU-Side Sampling (2026-03-26)

### Added

- **`shaders/sampling.metal`** — Six GPU kernels for autoregressive sampling: `argmax_reduce` / `argmax_reduce_final` (parallel reduction argmax), `apply_rep_penalty` (repetition penalty in-place), `apply_temperature` (temperature scaling in-place), `topk_find_threshold` / `topk_mask` (top-K filtering via threshold), `multinomial_sample` (prefix-sum CDF sampling).
- **`src/ops/sampling.js`** — GPU dispatch layer. `gpuArgmax(logits)` returns token index via two-pass parallel reduction. `gpuSample(logits, config)` runs the full pipeline: penalties → temperature → top-K → softmax → top-P → multinomial. Only 4 bytes cross GPU→CPU per token.
- **`gpuSampling` config option** — Pass `gpuSampling: true` to `generateGGUF` to use the GPU sampling path. Defaults to `false` for backward compatibility.
- **Tests** — 30+ sampling tests covering argmax, penalties, temperature, top-K, top-P, multinomial, full pipeline, and generateGGUF integration.

### Changed

- **`src/gguf_cache.js`** — `generateGGUF` refactored with `extractLastLogits()` and `sampleToken()` helpers supporting both CPU and GPU (`gpuSampling: true`) sampling paths.

## 0.23.0 — Chunked Whisper Audio (2026-03-26)

### Added

- **`examples/whisper/chunk.js`** — Chunked audio processing for Whisper. `chunkAudio(samples, opts)` splits audio into overlapping 30-second windows. `stitchTranscriptions(chunkResults, tokenizer)` merges per-chunk token arrays with text-based deduplication in overlap regions. `transcribeChunk(model, chunk, opts)` and `transcribeChunked(model, samples, opts)` provide the full pipeline with progress callbacks.
- **Tests** — 16 chunk tests: chunkAudio for known durations (10s, 30s, 31s, 60s, 90s, zero-length), overlap sample verification, custom chunk sizes, last-chunk-shorter, stitch empty/single/non-overlapping/overlapping/three-chunk, WHISPER_CHUNK_SAMPLES constant.

### Changed

- **`examples/whisper/cli.js`** — Replaced single-chunk transcription with chunked pipeline. Audio longer than 30 seconds is automatically split, transcribed per-chunk, and stitched. Per-chunk progress in verbose mode. JSON output includes chunk count.
- **`examples/whisper/README.md`** — Architecture diagram updated for chunk.js. Added Long Audio section. Updated limitations.

## 0.22.0 — Tensor Lifecycle Management (2026-03-26)

### Added

- **`src/lifecycle.js`** — Tensor lifecycle management module. `dispose(t)` releases a tensor's buffer to the pool immediately. `retain(t)` increments a ref count so the tensor survives scope cleanup. `isDisposed(t)` checks status. `using(fn)` / `usingAsync(fn)` run a function and dispose all tensors allocated inside when it returns (retained tensors survive). `withNoAlloc(fn)` throws if any tensor is allocated inside — useful for verifying buffer reuse. `activeScopeDepth()` reports nesting level.
- **Lifecycle fields on tensors** — `tensor.create()` now calls `trackAllocation()` which sets `_refCount` and `_disposed` on every tensor. No-op when no scope is active; zero overhead for existing code.
- **`poolStats().totalAllocated`** — Cumulative bytes allocated (not recycled) since last `poolDrain()`. Helps detect memory leaks in inference loops.
- **Tests** — 25 lifecycle tests: dispose nulls buffer/data, double-dispose safety, retain/dispose ref counting, scoped cleanup, nested scopes, exception safety, async scopes, withNoAlloc guard, poolStats integration, all tensor factory types.

### Changed

- **`src/tensor.js`** — `create()` now calls `trackAllocation()` from lifecycle.js. Import of lifecycle uses a pluggable release callback (`setReleaseFn`) to avoid circular dependency with device.js.
- **`src/pool.js`** — `poolStats()` now includes `totalAllocated` field. `poolDrain()` resets the counter.

## 0.21.0 — Model Registry and Fetcher (2026-03-26)

### Added

- **`src/models.js`** — Model registry, fetcher, and resolver. Downloads model files from Hugging Face Hub or direct URLs, verifies SHA-256 checksums, caches in `models/<id>/`. API: `listModels()`, `getModel(id)`, `modelPath(id)`, `fetchModel(id)`, `fetchUrl(url)`, `registerModel(id, entry)`, `removeModel(id)`.
- **`models/registry.json`** — Declarative registry of known models: Whisper (tiny/base/small/medium), ResNet-18/50, CLIP ViT-B/32/B/16/L/14, Nemotron-4B Q4_K_M. Each entry declares HF repo, format, loader, variant, and file list with optional checksums.
- **`models/registry.schema.json`** — JSON Schema (draft 2020-12) for validating registry entries.
- **Tests** — 31 tests: registry reads, path resolution, HF URL building, SHA-256 hashing, model registration, removal, mock HTTP fetch with progress and checksum verification, schema validation.

## 0.20.0 — GPU FFT and Mel Spectrogram (2026-03-26)

### Added

- **`shaders/fft.metal`** — Radix-2 Cooley-Tukey FFT kernel using threadgroup shared memory. Single kernel (`fft_radix2`) handles both forward and inverse transforms. Supports FFT lengths up to 1024 (Metal threadgroup size limit). Bit-reversal permutation in-kernel. Batch mode: one threadgroup per independent FFT for STFT.
- **`shaders/mel.metal`** — Four kernels for mel spectrogram extraction: `stft_window` (Hann window + interleaved complex), `stft_magnitude` (|FFT|² for positive bins), `mel_filterbank` (dense matmul), `mel_log` / `mel_normalize` (Whisper-style two-pass normalization).
- **`src/ops/fft.js`** — GPU dispatch: `gpuFFT(input, n?)`, `gpuIFFT(re, im)`, `gpuBatchFFT(complexIn, n, batch, inverse)`.
- **`src/ops/mel.js`** — GPU mel spectrogram pipeline: `gpuMelSpectrogram(samples, opts)` — window → batch FFT → magnitude → filterbank → log normalization. Matches CPU `melSpectrogram()` output.
- **`fft(input)`** autograd op — Differentiable FFT returning `{ re, im }` Variables. Backward via inverse FFT. Enables spectral loss functions.
- **Tests** — 12 FFT tests, 6 GPU mel spectrogram tests.

## 0.19.0 — KV-Cached Whisper Decoding (2026-03-26)

### Added

- **`whisperDecodePrefill(model, encoderOut, tokens)`** — Process the full initial prompt through the decoder in one pass. Returns logits, per-block self-attention KV caches, and pre-projected encoder K/V for cross-attention. The encoder K/V projections happen once and are reused for every subsequent decode step.
- **`whisperDecodeStep(model, encoderKV, tokenId, position, selfCaches)`** — Single-token cached decode step. Uses `multiHeadAttentionCached` for self-attention (cache grows by 1 per step) and `multiHeadCrossAttentionCached` for cross-attention (pre-computed K/V are constant).
- **`whisperTranscribeCached(model, melInput, opts)`** — Drop-in replacement for `whisperTranscribe` using prefill + step-by-step cached decoding. Same API surface and sampling logic. Reduces total self-attention work from O(n²) to O(n) over a full generation.
- **`precomputeEncoderKV(model, encoderOut)`** — Pre-compute encoder K/V projections for all decoder cross-attention layers. Returns `[{ k, v }]` per block.
- **Tests** — 9 KV-cached decoding tests: prefill shapes, step cache growth, encoder KV constancy, prefill/full-decode equivalence, cached/non-cached token equivalence, EOT stopping, single-token generation, onToken callback, early stopping.

### Changed

- **`examples/whisper/model.js`** — Sampling logic extracted to shared `sampleToken()` function used by both cached and non-cached transcription paths.

## 0.18.0 — GPU Conv1d (im2col + col2im) (2026-03-26)

### Changed

- **Conv1d fully GPU-accelerated** — Both forward (im2col) and backward (col2im) now run on Metal GPU instead of CPU loops. Forward uses a dedicated `im2col_1d_forward` shader to extract patches, followed by the existing tiled matmul. Backward uses `col2im_1d_backward` shader for scatter-add input gradient reconstruction. The API is unchanged: `conv1d(input, weight, bias, { stride, padding })`.

### Added

- **`shaders/conv1d.metal`** — Two Metal compute kernels: `im2col_1d_forward` (gather input patches into column matrix) and `col2im_1d_backward` (scatter-add columns back to input gradient). Handles arbitrary stride and padding. No batch dimension — operates on single `[C_in, length]` inputs matching Smith's existing conv1d signature.
- **`src/ops/conv1d.js`** — GPU dispatch module for 1D convolution: `im2col1d`, `col2im1d`, `conv1dForward`, `conv1dBackwardInput`, `conv1dBackwardWeight`. Follows the same structure as `conv2d_im2col.js`.
- **Tests** — 2 additional conv1d tests: multi-channel backward with stride+padding, and GPU col2im analytical verification.

## 0.17.0 — Conv1d, Cross-Attention, Sinusoidal PE (2026-03-26)

### Added

- **Conv1d** (`conv1d(input, weight, bias, opts)`) — 1D convolution with full autograd backward. Input `[C_in, length]`, weight `[C_out, C_in, kernel]`. Im2col + matmul strategy matching conv2d's approach. Options: `{ stride, padding }`. `conv1dOutputSize()` utility.
- **Cross-attention** (`multiHeadCrossAttention(x, kv, layer, mask?)`) — Q from one source, K/V from another. Reuses `createMultiHeadAttention` structure. Essential for encoder-decoder architectures (Whisper, T5, BART).
- **Cached cross-attention** (`multiHeadCrossAttentionCached(x, encoderKV, layer)`) — Pre-computed encoder K/V projections for efficient autoregressive decoding.
- **Sinusoidal PE** (`sinusoidalPE(maxLen, dim)`) — Fixed positional embeddings using sin/cos at geometrically-spaced frequencies. Returns raw tensor (not variable).
- **Tests** — 11 conv1d tests (shapes, known values, backward gradients), 12 cross-attention/sinusoidal PE tests.

### Changed

- **Whisper example** (`examples/whisper/model.js`) rewritten to use Smith's core `conv1d`, `multiHeadCrossAttention`, `sinusoidalPE`, and `createCausalMask` instead of local implementations. Weight loader maps updated to match Smith's MHA structure (`qProj.weight`, `kProj.weight`, etc.).

## 0.16.2 — Whisper Example Project (2026-03-26)

### Added

- **Whisper speech-to-text example** (`examples/whisper/`) — Complete Whisper implementation as a CLI application, demonstrating Smith's encoder-decoder transformer, cross-attention, and GGML model loading.
  - **WAV audio decoder** (`audio.js`) — Pure JS WAV reader supporting PCM int8/int16/int32 and IEEE float. Includes linear interpolation resampler to 16kHz mono.
  - **Mel spectrogram** (`mel.js`) — Pure JS radix-2 FFT, STFT with Hann window, and mel filterbank. Matches whisper.cpp's log-mel normalization (80 or 128 bins).
  - **Whisper model** (`model.js`) — Encoder-decoder transformer with Conv1d (via im2col+matmul), sinusoidal positional embeddings, self-attention, cross-attention, and greedy/temperature decoding.
  - **GGML loader** (`loader.js`, `ggml_parser.js`) — Reads whisper.cpp `.bin` files (magic `0x67676d6c`): hparams, mel filters, BPE vocab, and tensor data with F32/F16/Q8_0 dequantization.
  - **Tokenizer** (`tokenizer.js`) — GPT-2 byte-level BPE decoder with Whisper special tokens (language, task, timestamps).
  - **CLI** (`cli.js`) — `bun examples/whisper/cli.js --model ggml-tiny.bin --file audio.wav` with JSON/SRT/VTT output formats.
  - **Tests** — 25 tests across audio, mel, tokenizer, and GGML parser (non-GPU tests run anywhere, model tests require macOS + Apple Silicon).

## 0.16.1 — GGUF Parser Fixes and K-Quant Dequantization (2026-03-25)

### Fixed

- **GGUF magic constant** — Fixed endianness bug: `0x46475547` → `0x46554747`. The parser now correctly reads real GGUF files (was only working with synthetic test files that shared the same bug).

### Added

- **Q4_K dequantization** — Super-block dequantizer for Q4_K format (256-element blocks with 6-bit packed scales/mins). Handles the sub-block structure used in GGML's k-quant family.
- **Q6_K dequantization** — 6-bit dequantizer with split low/high nibbles and signed int8 sub-block scales. 210 bytes per 256-element block.
- **Q5_0 dequantization** — 5-bit format with 32-bit high-bit mask. 22 bytes per 32-element block.
- **GGML_TYPE_INFO entries** for Q5_0, Q4_K, Q6_K — block sizes and bytes-per-block now defined, enabling `tensorBytes()` and `listTensors()` for models using these types.
- **Integration test** (`tests/gguf_model.test.js`) — 30 tests, 11485 expect() calls against real Nemotron-H 4B Q4_K_M model. Validates: GGUF parsing, metadata/config extraction, hybrid architecture detection (attention + SSM layers), tensor type distribution, per-type dequantization (F32, Q5_0, Q4_K, Q6_K, Q8_0), tensor shapes vs architecture dimensions, data offset integrity.

## 0.16.0 — Phase 16: Profiling and Benchmarking (2026-03-25)

### Added

- **GPU timing in native bridge** (`native/gpu_bridge.m`) — `smith_end_timed` returns `SmithTiming` struct with `GPUStartTime`/`GPUEndTime` from Metal command buffers. `smith_allocated_size` reports device `currentAllocatedSize`.
- **Profiler** (`src/profile.js`) — `enableProfiling()`/`disableProfiling()` toggles per-kernel timing in `dispatch.js`. `report()` returns sorted kernel stats (calls, totalMs, avgMs, min, max, % of total) and memory info (start, peak, current, delta). `profile(fn)` wraps a function and returns `{ result, cpuMs, gpuMs, dispatches, kernels, memory }`. `benchmark(name, fn, opts)` runs warmup + N iterations, reports mean/median/p95/min/max/stddev for both CPU and GPU time. `memorySnapshot()` returns current device allocation.
- **Instrumented dispatch** — `dispatch.js` checks `isProfilingEnabled()` on every `run()` call and uses `endTimed` instead of `endSync` when active. Zero overhead when profiling is disabled (single boolean check).
- **Tests** (`tests/profile.test.js`) — State management (enable/disable/reset), profile() correctness and observer-effect tests (numerical results unchanged, backward works, conv2d works), benchmark stats, memory tracking, per-kernel stats with percentage validation, sorted output, edge cases.

## 0.15.0 — Phase 15: Vision Model Loading (2026-03-25)

### Added

- **ResNet builder** (`src/resnet.js`) — Creates ResNet-18/34 (BasicBlock) and ResNet-50/101/152 (Bottleneck). Stem: 7×7 conv stride 2 → BN → ReLU → maxpool. Four residual stages with configurable blocks and downsampling. Global average pool → FC classifier head. `loadResNet(path, opts)` loads torchvision-format safetensors.
- **CLIP model** (`src/clip.js`) — ViT-B/32, ViT-B/16, ViT-L/14 vision encoders + transformer text encoder + contrastive projection heads. Non-causal attention for vision, causal for text. `loadCLIP(path, opts)` loads OpenAI CLIP safetensors format, including fused `in_proj_weight` split into Q/K/V.
- **Image preprocessing** (`src/vision.js`) — `resizeBilinear`, `centerCrop`, `normalize` for CPU-side image processing. `preprocessResNet` and `preprocessCLIP` as ready-to-use pipelines. `loadPPM` for dependency-free P6 image loading.
- **Tests** (`tests/vision.test.js`) — Preprocessing correctness, ResNet block shapes and forward/backward, CLIP model structure and forward, l2 normalization, similarity matrix, integration pipelines.

## 0.14.0 — Phase 14: im2col Convolution Path (2026-03-25)

### Added

- **im2col Metal shaders** (`shaders/im2col.metal`) — `im2col_forward` rearranges input patches into column matrix `[batch, inC*kH*kW, outH*outW]` for GEMM-based convolution. `col2im_backward` scatter-adds columns back to input gradient layout. Supports arbitrary kernel sizes, strides, padding, and dilation.
- **JS dispatch** (`src/ops/conv2d_im2col.js`) — `im2col` and `col2im` GPU dispatch. `im2colForward` performs im2col + per-batch GEMM via existing `matmul2d`. `im2colBackwardInput` via weight^T @ grad + col2im. `im2colBackwardWeight` via grad @ cols^T + accumulate. `shouldUseIm2col` selects this path for non-3×3 kernels or strided/dilated 3×3.
- **3-way auto-dispatch** — `conv2d()` in autograd now selects: Winograd (3×3, stride 1, dilation 1) > im2col (larger kernels, strided 3×3) > direct (1×1 pointwise fallback). No code changes needed.
- **Tests** (`tests/im2col.test.js`) — `shouldUseIm2col` selection logic, im2col shape/size verification, forward equivalence vs direct conv (5×5, 7×7 stride 2, batched, no bias, CPU reference), auto-dispatch for 5×5 and 3×3-stride-2, backward gradient finiteness, numerical gradient checks (input and weight), autograd pipeline.

## 0.13.0 — Phase 13: Winograd Convolution (2026-03-25)

### Added

- **Winograd F(2×2, 3×3) Metal shader** (`shaders/conv2d_winograd.metal`) — Computes 2×2 output tiles from 4×4 input tiles using 16 multiplications instead of 36 (2.25× reduction). Forward kernels with and without bias. Backward kernel for input gradient using atomic scatter-add for overlapping tiles.
- **JS dispatch** (`src/ops/conv2d_winograd.js`) — `transformFilter3x3` pre-transforms a single 3×3 filter via G×g×G^T (CPU, once). `transformWeights` batch-transforms all filters. `canUseWinograd` checks eligibility (3×3, stride 1, dilation 1, groups 1). `winogradForward` and `winogradBackwardInput` dispatch to GPU.
- **Auto-dispatch** — `conv2d()` in autograd automatically selects Winograd for eligible 3×3 convolutions, direct conv for everything else. Weight gradient always uses direct conv (Winograd weight gradient too complex for marginal benefit).
- **Tests** (`tests/winograd.test.js`) — Filter transform properties, `canUseWinograd` selection logic, numerical equivalence vs direct conv (no padding, padding=1, batched, no bias, odd dims, CPU reference), auto-dispatch verification, backward gradient finiteness, numerical gradient checks (input and weight), autograd pipeline (conv→relu→conv→sum).

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
