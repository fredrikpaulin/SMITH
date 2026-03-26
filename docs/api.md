# Smith API Reference

## Tensor Creation

All tensors are GPU-backed via Metal shared buffers. CPU reads/writes happen through unified memory (zero-copy).

| Function | Signature | Description |
|----------|-----------|-------------|
| `tensor` | `(values, shape, dtype?)` | Create from flat JS array |
| `zeros` | `(shape, dtype?)` | All zeros |
| `ones` | `(shape, dtype?)` | All ones |
| `full` | `(shape, value, dtype?)` | Fill with scalar |
| `rand` | `(shape, dtype?)` | Uniform [0, 1) |
| `randn` | `(shape, dtype?)` | Normal N(0, 1) |
| `scalar` | `(value, dtype?)` | 0-dimensional tensor |
| `toArray` | `(tensor)` | Read back to nested JS array |

Supported dtypes: `'f32'` (default), `'f16'`.

## Autograd

Variables wrap tensors with gradient tracking.

| Function | Description |
|----------|-------------|
| `variable(tensor, opts?)` | Wrap tensor. `opts.requiresGrad: true` to track. |
| `param(shape, initFn?)` | Convenience: create + requiresGrad. |
| `backward(v)` | Reverse-mode AD from variable `v`. |
| `zeroGrad(params)` | Reset all gradients to null. |
| `noGrad(fn)` | Execute `fn` without building the graph. |

## Operations (on Variables)

All ops return new Variables with backward functions.

**Arithmetic:** `add(a, b)`, `sub(a, b)`, `mul(a, b)`, `scale(a, scalar)`, `neg(a)`

**Matrix:** `matmul(a, b)` — supports 2D and batched

**Attention:** `flashAttention(q, k, v, causal?)` — fused tiled attention, O(n) memory

**Activation:** `relu(a)`, `gelu(a)`

**Normalization:** `softmax(a, axis?)`, `layernorm(a, gamma, beta, eps?)`

**Reduction:** `sum(a, axis?)` — scalar sum if no axis

**Shape:** `reshape(a, shape)`, `transpose(a, axes?)`

**Conv1d:** `conv1d(input, weight, bias, opts?)` — 1D convolution with full autograd. Input `[C_in, length]`, weight `[C_out, C_in, kernel]`, bias `[C_out]` or null. Options: `{ stride, padding }`. GPU im2col + matmul forward, GPU col2im backward.

`conv1dOutputSize(length, kernelSize, stride, padding)` — compute output length.

**Loss:** `crossEntropy(logits, targets)` — logits: `[batch, vocab]`, targets: int array

**Embedding:** `embedding(indices, weight)` — CPU gather, GPU scatter-add backward

## Optimizer

```js
const opt = createAdamW(params, { lr, beta1, beta2, eps, weightDecay })
adamwStep(opt)  // GPU-fused parameter update
```

**LR Schedule:**
```js
const sched = createSchedule({ warmupSteps, totalSteps, maxLr, minLr })
opt.lr = getLr(sched, step)
```

**Gradient Clipping:**
```js
const norm = clipGradNorm(params, maxNorm)
```

## Neural Network

| Function | Description |
|----------|-------------|
| `createLinear(in, out, bias?)` | Linear projection |
| `linear(x, layer)` | Forward pass |
| `createMultiHeadAttention(dim, heads)` | MHA with Q/K/V/out projections |
| `multiHeadAttention(x, layer, mask)` | Forward with causal mask |
| `multiHeadCrossAttention(x, kv, layer, mask?)` | Cross-attention: Q from x, K/V from kv |
| `multiHeadCrossAttentionCached(x, encoderKV, layer)` | Cross-attention with pre-computed K/V |
| `multiHeadAttentionFlash(x, layer, causal?)` | Flash attention forward (O(n) memory) |
| `multiHeadAttentionCached(x, layer, cache)` | Single-token forward, appends K/V to cache |
| `sinusoidalPE(maxLen, dim)` | Sinusoidal positional embeddings (returns tensor, not variable) |
| `createTransformerBlock(dim, heads, ffnDim?)` | Pre-norm block |
| `transformerBlock(x, block, mask)` | Attention + FFN + residuals |
| `transformerBlockFlash(x, block)` | Flash attention block (O(n) memory) |
| `transformerBlockCached(x, block, cache)` | Single-token block with KV cache |
| `createCausalMask(seqLen)` | Upper-triangle -Infinity mask |

## Model

```js
const model = createModel({ vocabSize, numLayers, numHeads, dim, maxSeqLen })
const { logits } = forward(model, tokenIds)
const { logits } = forwardFlash(model, tokenIds)  // O(n) memory attention
const { logits, newCaches } = forwardCached(model, tokenId, position, kvCaches)
const params = modelParams(model)
const info = modelInfo(model)
```

Preset configs: `CONFIGS.tiny`, `CONFIGS.small`, `CONFIGS.medium`.

## Generation

```js
// Full context (recomputes everything each token)
const ids = generate(model, promptIds, { maxTokens, temperature, topK, topP, repetitionPenalty })

// KV cache (O(1) per token after prompt)
const ids = generateCached(model, promptIds, { maxTokens, temperature, topK, topP, repetitionPenalty })
```

## Safetensors

```js
// Parse a safetensors file
const buf = await Bun.file('model.safetensors').arrayBuffer()
const parsed = parseSafetensors(buf)

// Inspect contents
const tensors = listTensors(parsed) // [{ name, shape, dtype }]
const t = readTensor(parsed, 'transformer.wte.weight') // { data, shape, dtype }

// Load GPT-2 weights (infers config from tensor shapes)
const model = await loadGPT2Safetensors('model.safetensors')

// Load into existing model
const parsed = (await loadSafetensors('model.safetensors')).parsed
mapGPT2Weights(parsed, model)

// Export model to safetensors format
const buf = exportSafetensors(model)
await saveSafetensors(model, 'output.safetensors')
```

| Function | Description |
|----------|-------------|
| `parseSafetensors(buffer)` | Parse safetensors ArrayBuffer → `{ tensors, buffer, header }` |
| `readTensor(parsed, name)` | Read named tensor → `{ data, shape, dtype }` |
| `listTensors(parsed)` | List all tensor names, shapes, dtypes |
| `loadSafetensors(path)` | Load file → `{ parsed, tensors }` |
| `loadGPT2Safetensors(path, overrides?)` | Load GPT-2 model from safetensors (infers config) |
| `mapGPT2Weights(parsed, model)` | Map GPT-2 weights into existing Smith model |
| `exportSafetensors(model)` | Export model → safetensors ArrayBuffer |
| `saveSafetensors(model, path)` | Export and write to disk |

## Checkpoint

```js
import { saveCheckpoint, loadCheckpoint } from './src/checkpoint.js'
await saveCheckpoint(model, 'path/prefix')
const model = await loadCheckpoint('path/prefix')
```

## Tokenizer

```js
import { train, encode, decode, save, load } from './src/tokenizer.js'
```

## Mixed Precision (f16)

```js
// Toggle f16 mode — tensor creation defaults to f16
smith.f16Mode(true)
smith.defaultDtype()  // 'f16'

// Cast between dtypes
const h = smith.cast(f32Tensor, 'f16')  // GPU kernel
const f = smith.cast(f16Tensor, 'f32')

// Dynamic loss scaler for mixed precision training
const scaler = smith.createLossScaler({
  initScale: 65536,     // starting scale (default 2^16)
  growthInterval: 2000, // steps between growth attempts
  growthFactor: 2,      // multiply scale on growth
  backoffFactor: 0.5,   // multiply scale on NaN
  minScale: 1,          // floor
})

// Training loop:
const scaledLoss = scaler.scaleUp(loss)
backward(scaledLoss)
const ok = scaler.unscale(gradArrays)  // returns false if NaN/Inf
if (ok) adamwStep(opt)
scaler.update(ok)
```

All ops automatically dispatch f16 kernels when given f16 tensors. f16 shaders use f32 accumulators for reductions (matmul inner loop, softmax, layernorm, reduce).

## Quantization

```js
import { quantizeQ4, matmulQ4 } from './src/ops/quantize.js'
const wq = quantizeQ4(weightTensor)    // f32 → q4
const out = matmulQ4(activations, wq)  // q4 matmul on GPU
// Q8 matmul also available: matmulQ8(activations, bQuant)
```

## Vision Models

Load pretrained ResNet and CLIP models from safetensors files. Includes image preprocessing utilities.

```js
import smith from './src/index.js'

// ResNet inference
const { forward } = await smith.loadResNet('resnet50.safetensors', { variant: 'resnet50' })
const input = smith.preprocessResNet(rgbaPixels, width, height)
const logits = forward(smith.variable(input, { requiresGrad: false }), false)

// CLIP: encode image and text
const clip = await smith.loadCLIP('clip-vit-b-32.safetensors', { variant: 'ViT-B/32' })
const imgEmbed = clip.encodeImage(imageInput)
const txtEmbed = clip.encodeText([49406, 320, 1125, 539, 320, 2368, 49407])
const similarity = clip.similarity(imgEmbed, txtEmbed)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `createResNet` | `(variant, numClasses)` | Build ResNet model (resnet18/34/50/101/152) |
| `forwardResNet` | `(model, x, training)` | Forward pass, returns `[N, numClasses]` logits |
| `loadResNet` | `(path, opts?)` | Load pretrained weights from safetensors |
| `createCLIP` | `(variant)` | Build CLIP model (ViT-B/32, ViT-B/16, ViT-L/14) |
| `forwardVision` | `(model, x)` | Encode images → `[N, embedDim]` |
| `forwardText` | `(model, tokenIds)` | Encode text → `[1, embedDim]` |
| `loadCLIP` | `(path, opts?)` | Load pretrained CLIP from safetensors |
| `clipSimilarity` | `(imgFeats, txtFeats, logitScale)` | Cosine similarity matrix |
| `preprocessResNet` | `(pixels, w, h, opts?)` | Resize → crop 224 → ImageNet normalize → tensor |
| `preprocessCLIP` | `(pixels, w, h, opts?)` | Resize → crop 224 → CLIP normalize → tensor |
| `loadPPM` | `(buffer)` | Parse PPM P6 image → `{ pixels, width, height }` |

Image utilities: `resizeBilinear`, `centerCrop`, `normalize`, `rgbaToChw`, `rgbToChw`.

## GGUF Import

Load models from GGUF files (llama.cpp, ollama format). Supports Llama, Phi, and GPT-2 architectures for inference. Dequantization supports Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K, F16, BF16, and F32 weight types.

```js
// Load a GGUF model
const { model, config, forward, loaded, skipped } = await smith.loadGGUF('model.gguf')

// Run inference
const { logits } = forward([1, 2, 3, 4]) // token IDs

// Inspect without loading
const buf = await Bun.file('model.gguf').arrayBuffer()
const parsed = smith.parseGGUF(buf)
const tensors = smith.listGGUFTensors(parsed) // [{ name, shape, type, bytes }]
const config = smith.extractGGUFConfig(parsed.metadata)
```

| Function | Description |
|----------|-------------|
| `loadGGUF(path)` | Load GGUF file → model with forward function |
| `parseGGUF(buffer)` | Parse GGUF ArrayBuffer → `{ version, metadata, tensors, dataOffset }` |
| `listGGUFTensors(parsed)` | List tensor names, shapes, types, byte sizes |
| `extractGGUFConfig(metadata)` | Extract model config (arch, dim, layers, heads, RoPE, etc.) |

Supported architectures for forward pass: `llama` (Llama 2/3, Mistral, CodeLlama, TinyLlama), `phi`/`phi2`/`phi3`, `gpt2`. The GGUF parser supports any architecture — use `parseGGUF` + `extractConfig` + `dequantizeTensor` for unsupported archs.

Supported quantization types for dequantization: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K.

## Convolutions

NCHW-layout 2D convolution, pooling, and batch normalization for building CNNs.

```js
import smith from './src/index.js'

// Conv2d — autograd-aware
const input = smith.variable(smith.rand([1, 3, 32, 32]), { requiresGrad: true })
const weight = smith.variable(smith.rand([16, 3, 3, 3]), { requiresGrad: true })
const bias = smith.variable(smith.zeros([16]), { requiresGrad: true })
const conv = smith.conv2d(input, weight, bias, { padding: 1 })

// Pooling
const pooled = smith.maxPool2d(conv, { kernelSize: 2 })
const avgPooled = smith.avgPool2d(conv, { kernelSize: 2, stride: 2 })

// Batch normalization
const bn = smith.createBatchNorm(16)
const normed = smith.batchnorm(conv, bn, true) // true = training

// Backward
const loss = smith.sum(pooled)
smith.backward(loss)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `conv2d` | `(input, weight, bias, opts?)` | 2D convolution on Variables |
| `maxPool2d` | `(input, opts?)` | Max pooling with gradient support |
| `avgPool2d` | `(input, opts?)` | Average pooling with gradient support |
| `batchnorm` | `(input, layer, training?)` | Batch normalization |
| `createBatchNorm` | `(channels, opts?)` | Create BN layer state (gamma, beta, running stats) |
| `convOutputSize` | `(inSize, kSize, stride, pad, dilation)` | Compute conv output dimension |
| `poolOutputSize` | `(inSize, kSize, stride, pad)` | Compute pool output dimension |

Conv2d options: `{ stride, padding, dilation, groups }` — each accepts scalar or `[H, W]` array.

**Winograd auto-dispatch:** `conv2d()` automatically uses Winograd F(2×2, 3×3) for 3×3 kernels with stride 1, dilation 1, groups 1. This reduces arithmetic from 36 to 16 multiplications per 2×2 output tile. No code changes needed — the dispatch is transparent. For manual control:

| Function | Signature | Description |
|----------|-----------|-------------|
| `canUseWinograd` | `(weight, opts)` | Check if Winograd is applicable |
| `winogradTransformWeights` | `(weight)` | Pre-transform `[outC, inC, 3, 3]` → `[outC, inC, 4, 4]` |

**im2col auto-dispatch:** For kernels larger than 3×3 (e.g. 5×5, 7×7) or strided/dilated 3×3, `conv2d()` automatically uses im2col + GEMM. Input patches are rearranged into a column matrix `[inC*kH*kW, outH*outW]`, then multiplied by the flattened weight matrix using the existing tiled matmul shader. Trades memory for compute efficiency on larger kernels.

| Function | Signature | Description |
|----------|-----------|-------------|
| `shouldUseIm2col` | `(weight, opts)` | Check if im2col path will be used |

**3-way dispatch priority:** Winograd (3×3 stride-1 dilation-1) > im2col (larger kernels, strided 3×3) > direct (1×1 pointwise).

Pool options: `{ kernelSize, stride, padding }` — each accepts scalar or `[H, W]` array. Stride defaults to kernelSize.

BatchNorm options: `{ eps, momentum }` — defaults: `1e-5`, `0.1`.

## RoPE, RMSNorm, SwiGLU

GPU-accelerated Llama-style operations. Used internally by the GGUF loader and available as standalone autograd ops for custom architectures.

```js
import smith from './src/index.js'

// RoPE: precompute frequency tables, then apply
const table = smith.precomputeRoPE(64, 2048, 10000)  // dim, maxSeqLen, freqBase
const q = smith.variable(smith.rand([16, 64]), { requiresGrad: true })
const rotated = smith.rope(q, table, 0)  // startPos for KV cache

// RMSNorm: Llama-style normalization (no mean subtraction, no beta)
const x = smith.variable(smith.rand([16, 256]), { requiresGrad: true })
const gamma = smith.variable(smith.ones([256]), { requiresGrad: true })
const normed = smith.rmsNorm(x, gamma, 1e-5)

// SwiGLU: fused silu(gate) * up
const gate = smith.variable(smith.rand([16, 512]), { requiresGrad: true })
const up = smith.variable(smith.rand([16, 512]), { requiresGrad: true })
const activated = smith.swiglu(gate, up)

// All support backward
const loss = smith.sum(activated)
smith.backward(loss)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `precomputeRoPE` | `(dim, maxSeqLen, freqBase?)` | Build cos/sin frequency tables (always f32) |
| `rope` | `(input, ropeTable, startPos?)` | Apply rotary position embeddings (autograd) |
| `rmsNorm` | `(input, gamma, eps?)` | RMS normalization (autograd) |
| `swiglu` | `(gate, up)` | Fused SiLU(gate) × up (autograd) |

## GGUF KV Cache

KV cache for GGUF-loaded Llama-style models. Prefill processes the full prompt in one pass via flash attention, then decode generates tokens one at a time with O(1) per-token compute. Supports Grouped Query Attention (GQA) where KV heads < Q heads.

```js
import { loadGGUF } from './src/gguf_loader.js'

const { model, generate, createCache, forwardPrefill, forwardDecode, resetCache } = await loadGGUF('model.gguf')

// Option 1: high-level generation
const tokens = generate([1, 2, 3], {
  maxTokens: 50,
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  repetitionPenalty: 1.1,
  eosToken: 2,
}, {
  onToken: (tok, step) => process.stdout.write(tokenizer.decode([tok])),
})

// Option 2: manual cache control
const caches = createCache()
const prefillResult = forwardPrefill([1, 2, 3], caches)       // full prompt
const decodeResult = forwardDecode(nextToken, 3, caches)       // one token at position 3
resetCache(caches)                                              // reuse for new prompt
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `loadGGUF` | `(path) → { model, generate, createCache, ... }` | Load GGUF and return model with cached generation |
| `generate` | `(promptIds, config?, callbacks?)` | Prefill+decode generation with sampling |
| `createCache` | `() → caches` | Allocate KV cache buffers for the model |
| `forwardPrefill` | `(tokenIds, caches) → { logits }` | Process full prompt via flash attention |
| `forwardDecode` | `(tokenId, position, caches) → { logits }` | Generate one token using cached K/V |
| `resetCache` | `(caches)` | Zero positions to reuse cache buffers |

Generation config: `{ maxTokens, temperature, topK, topP, repetitionPenalty, eosToken }`. Callbacks: `{ onToken(token, step) }` — return `true` to stop.

## Profiling and Benchmarking

Instrument Metal dispatches to collect per-kernel GPU timing and memory usage. Uses `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` for accurate GPU measurement. Zero overhead when disabled.

```js
import smith from './src/index.js'

// Wrap a function to get timing
const p = smith.profile(() => {
  const a = smith.variable(smith.rand([256, 256]), { requiresGrad: false })
  const b = smith.variable(smith.rand([256, 256]), { requiresGrad: false })
  return smith.noGrad(() => smith.matmul(a, b))
})
// p = { result, cpuMs, gpuMs, dispatches, kernels, memory }

// Benchmark with warmup + iterations
const b = smith.benchmark('matmul-256', () => {
  smith.noGrad(() => smith.matmul(a, b))
}, { warmup: 3, iterations: 10 })
// b = { name, iterations, cpu: { mean, median, p95, min, max, stddev }, gpu: {...} }

// Manual profiling
smith.enableProfiling()
// ... do work ...
const report = smith.profileReport()
// report = { dispatches, totalGpuMs, memory, kernels: [{ kernel, calls, totalMs, avgMs, pct }] }
smith.disableProfiling()
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `profile` | `(fn) → { result, cpuMs, gpuMs, ... }` | Wrap function with full profiling |
| `benchmark` | `(name, fn, opts?) → stats` | Run N iterations, report CPU/GPU stats |
| `enableProfiling` | `()` | Start collecting per-kernel timing |
| `disableProfiling` | `()` | Stop collecting timing |
| `profileReport` | `() → report` | Get kernel stats, sorted by total time |
| `resetProfile` | `()` | Clear all collected stats |
| `memorySnapshot` | `() → { allocatedBytes }` | Current GPU memory allocation |

Benchmark options: `{ warmup, iterations }` — defaults: `3`, `10`.
