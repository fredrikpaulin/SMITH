---
name: smith
description: |
  How to use the Smith GPU compute library for ML in Bun on Apple Silicon. Use this skill whenever the user wants to build something with Smith — training models, running inference, loading GGUF/safetensors weights, image preprocessing, profiling GPU work, or writing custom autograd pipelines. Also trigger when the user mentions Metal shaders, GPU tensors in JavaScript, or Apple Silicon ML from Bun, even if they don't say "Smith" by name. If the user is working in a project that imports from smith's src/ directory, this skill applies.
---

# Smith — GPU Compute for Bun on Apple Silicon

Smith gives JavaScript direct access to Metal compute shaders through Bun's FFI. All tensors live in GPU shared memory — a `Float32Array` in JS and a `device float*` in Metal point to the same physical bytes. Zero copies, zero dependencies.

## Requirements

- macOS on Apple Silicon (M1–M5)
- Bun runtime
- Xcode Command Line Tools
- Build first: `bash build.sh` (compiles native bridge + Metal shaders)

## Import

```js
import smith from './src/index.js'
// or named imports:
import { tensor, variable, matmul, backward } from './src/index.js'
```

Everything lives in a single flat namespace. No nested modules to import.

## Core Concepts

### Tensors vs Variables

Smith has two data types. Tensors are raw GPU storage. Variables wrap tensors with gradient tracking for autograd.

```js
// Tensor — GPU-backed, no gradients
const t = smith.tensor([1, 2, 3, 4], [2, 2])
t.data        // Float32Array view into GPU memory
t.shape       // [2, 2]

// Variable — wraps a tensor, tracks computation graph
const v = smith.variable(t, { requiresGrad: true })
v.data        // the tensor
v.grad        // null until backward() runs
v.requiresGrad // true
```

All ops (matmul, relu, add, etc.) take and return **Variables**, not tensors. To use a raw tensor in an op, wrap it first:

```js
const x = smith.variable(smith.rand([4, 8]), { requiresGrad: false })
```

### Creating Parameters

For trainable weights, use `param` — it creates a variable with `requiresGrad: true`:

```js
const w = smith.param([inputDim, outputDim], () => (Math.random() - 0.5) * 0.02)
```

`param(shape, initFn, dtype)` — the init function receives no arguments and returns a scalar. It is NOT `param(tensor)`.

### Reading Data Back

```js
smith.toArray(tensor)  // nested JS array matching shape: [[1, 2], [3, 4]]
tensor.data            // flat Float32Array (GPU-shared)
```

## Common Patterns

### Training Loop

```js
const model = smith.createModel({
  vocabSize: 256, numLayers: 2, numHeads: 2,
  dim: 64, maxSeqLen: 128,
})
const params = smith.modelParams(model)
const opt = smith.createAdamW(params, { lr: 1e-3 })
const sched = smith.createSchedule({ warmupSteps: 100, totalSteps: 1000, maxLr: 1e-3, minLr: 1e-5 })

for (let step = 0; step < 1000; step++) {
  smith.zeroGrad(params)
  const { logits } = smith.forward(model, tokens)
  const loss = smith.crossEntropy(logits, targets)
  smith.backward(loss)
  smith.clipGradNorm(params, 1.0)
  opt.lr = smith.getLr(sched, step)
  smith.adamwStep(opt)
}
```

### Inference (No Gradient Tracking)

Wrap in `noGrad` to skip graph construction — faster and uses less memory:

```js
smith.noGrad(() => {
  const { logits } = smith.forward(model, tokenIds)
  // use logits...
})
```

### Text Generation

```js
// Simple — recomputes full context each token
const ids = smith.generate(model, promptIds, {
  maxTokens: 100, temperature: 0.8, topK: 40, topP: 0.95,
})

// Fast — O(1) per token after prompt via KV cache
const ids = smith.generateCached(model, promptIds, {
  maxTokens: 100, temperature: 0.8, topK: 40,
  repetitionPenalty: 1.1, eosToken: 2,
})
```

### Loading GGUF Models (llama.cpp / ollama format)

```js
const llama = await smith.loadGGUF('path/to/model.gguf')
// Returns: { model, config, forward, generate, createCache, forwardPrefill, forwardDecode, resetCache }

// High-level generation
const tokens = llama.generate([1, 2, 3], {
  maxTokens: 50, temperature: 0.8, topK: 40,
  eosToken: 2,
}, {
  onToken: (tok, step) => process.stdout.write(decode(tok)),
})

// Manual cache control
const caches = llama.createCache()
llama.forwardPrefill([1, 2, 3], caches)
llama.forwardDecode(nextToken, 3, caches)
llama.resetCache(caches)
```

Supported architectures for full forward pass: `llama`, `phi`, `phi2`, `phi3`, `gpt2`.

For unsupported architectures, use the parser directly to inspect and dequantize:

```js
const buf = await Bun.file('model.gguf').arrayBuffer()
const parsed = smith.parseGGUF(buf)
const config = smith.extractGGUFConfig(parsed.metadata)
const tensors = smith.listGGUFTensors(parsed)
```

Dequantization supports: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K.

### Loading Safetensors

```js
// GPT-2 format (auto-infers config from tensor shapes)
const model = await smith.loadGPT2Safetensors('model.safetensors')

// Manual loading into existing model
const { parsed } = await smith.loadSafetensors('model.safetensors')
smith.mapGPT2Weights(parsed, model)

// Export
await smith.saveSafetensors(model, 'output.safetensors')
```

### Vision: ResNet

```js
const { forward } = await smith.loadResNet('resnet50.safetensors', { variant: 'resnet50' })
const input = smith.preprocessResNet(rgbaPixels, width, height)
// preprocessResNet: resize short edge 256 → center crop 224 → ImageNet normalize → [1,3,224,224] tensor
const logits = forward(smith.variable(input, { requiresGrad: false }), false)
```

Variants: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`.

### Vision: CLIP

```js
const clip = await smith.loadCLIP('clip-vit-b-32.safetensors', { variant: 'ViT-B/32' })
const imgEmbed = clip.encodeImage(imageInput)
const txtEmbed = clip.encodeText(tokenIds)
const similarity = clip.similarity(imgEmbed, txtEmbed)
```

Variants: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`.

### Image Preprocessing (CPU, zero deps)

```js
// From RGBA Uint8Array (e.g., canvas)
const chw = smith.rgbaToChw(pixels, width, height)  // → CHW Float32Array [0,1]

// From RGB Uint8Array
const chw = smith.rgbToChw(pixels, width, height)

// From PPM file (dependency-free image loading)
const { pixels, width, height } = smith.loadPPM(buffer)

// Full pipelines
const t = smith.preprocessResNet(pixels, width, height)  // ImageNet normalization
const t = smith.preprocessCLIP(pixels, width, height)    // CLIP normalization
```

### Convolutions (NCHW layout)

```js
const input = smith.variable(smith.rand([1, 3, 32, 32]), { requiresGrad: true })
const weight = smith.variable(smith.rand([16, 3, 3, 3]), { requiresGrad: true })
const bias = smith.variable(smith.zeros([16]), { requiresGrad: true })

// Auto-dispatches: Winograd (3x3 stride 1) > im2col (larger) > direct (1x1)
const out = smith.conv2d(input, weight, bias, { padding: 1, stride: 1, dilation: 1 })

const pooled = smith.maxPool2d(out, { kernelSize: 2 })
const avg = smith.avgPool2d(out, { kernelSize: 2, stride: 2 })

const bn = smith.createBatchNorm(16)
const normed = smith.batchnorm(out, bn, true)  // true = training mode
```

### Profiling GPU Work

```js
// Wrap a function — captures timing, memory, per-kernel stats
const p = smith.profile(() => {
  return smith.noGrad(() => smith.matmul(a, b))
})
// p = { result, cpuMs, gpuMs, dispatches, kernels, memory }

// Benchmark with warmup
const b = smith.benchmark('matmul-256', () => {
  smith.noGrad(() => smith.matmul(a, b))
}, { warmup: 3, iterations: 10 })
// b.gpu = { mean, median, p95, min, max, stddev }
```

### Mixed Precision (f16)

```js
smith.f16Mode(true)            // all new tensors default to f16
const h = smith.cast(t, 'f16') // explicit cast
const f = smith.cast(h, 'f32')

// Loss scaler for training stability
const scaler = smith.createLossScaler({ initScale: 65536 })
const scaledLoss = scaler.scaleUp(loss)
smith.backward(scaledLoss)
const ok = scaler.unscale(grads)
if (ok) smith.adamwStep(opt)
scaler.update(ok)
```

### Checkpoints

```js
await smith.saveCheckpoint(model, './my-model')
const loaded = await smith.loadCheckpoint('./my-model')
```

### Quantization

```js
const wq = smith.quantizeQ4(weightTensor)
const out = smith.matmulQ4(activations, wq)
// Also: matmulQ8 for 8-bit
```

## Gotchas

These are the things that catch people. They come from real bugs encountered during development.

**`embedding(indices, weight)` — indices first, weight second.** The argument order is the opposite of PyTorch's `nn.Embedding(weight)(indices)`. Getting this backwards produces a cryptic "undefined is not an object" error deep in autograd.

**`param(shape, initFn)` not `param(tensor)`.** Don't pass a tensor to `param`. Pass a shape and an optional init function.

**Variables vs tensors in ops.** Every autograd op expects Variables. Passing a raw tensor to `matmul`, `add`, etc. will fail. Wrap with `smith.variable(tensor, { requiresGrad: false })`.

**`toArray` returns nested arrays.** `smith.toArray(t)` for a `[2,3]` tensor returns `[[a,b,c],[d,e,f]]`, not a flat array. Use `t.data` for the flat `Float32Array`.

**GGUF weight shapes are transposed from PyTorch.** GGUF stores weights as `[dim, vocabSize]` but PyTorch convention is `[vocabSize, dim]`. The loader handles this, but if you're manually loading weights, remember to transpose.

**Conv layout is NCHW.** All convolution and pooling ops expect `[batch, channels, height, width]`.

**`batchnorm` needs the layer object.** Call `smith.createBatchNorm(channels)` first, then pass the returned object to `smith.batchnorm(input, bnLayer, training)`.

**`noGrad` for inference.** Without `noGrad`, every op builds the computation graph. For inference this wastes memory and time. Always wrap inference in `smith.noGrad(() => { ... })`.

**Build before running.** Smith needs `bash build.sh` to compile `libsmith.dylib` and `smith.metallib`. Without this, the FFI dlopen fails immediately.

## Testing

Tests use `bun:test` and live in `tests/`. They require macOS + Apple Silicon (Metal GPU access).

```js
import { test, expect } from 'bun:test'
import smith from '../src/index.js'

test('matmul gradient', () => {
  const a = smith.variable(smith.tensor([1, 2, 3, 4], [2, 2]), { requiresGrad: true })
  const b = smith.variable(smith.tensor([5, 6, 7, 8], [2, 2]), { requiresGrad: true })
  const c = smith.matmul(a, b)
  const loss = smith.sum(c)
  smith.backward(loss)
  expect(a.grad).not.toBeNull()
})
```

Run all tests: `bun test tests/`

## API Quick Reference

**Tensor creation:** `tensor`, `zeros`, `ones`, `full`, `rand`, `randn`, `scalar`, `toArray`

**Autograd:** `variable`, `param`, `backward`, `zeroGrad`, `noGrad`

**Ops:** `add`, `sub`, `mul`, `matmul`, `scale`, `neg`, `relu`, `gelu`, `softmax`, `layernorm`, `crossEntropy`, `flashAttention`, `sum`, `reshape`, `transpose`, `embedding`

**NN:** `createLinear`, `linear`, `createMultiHeadAttention`, `multiHeadAttentionFlash`, `createTransformerBlock`, `transformerBlockFlash`, `createCausalMask`, `countParams`

**Model:** `CONFIGS`, `createModel`, `forward`, `forwardFlash`, `forwardCached`, `modelParams`

**Optimizer:** `createAdamW`, `adamwStep`, `createSchedule`, `getLr`, `clipGradNorm`

**Generation:** `generate`, `generateCached`, `topKPredictions`

**GGUF:** `loadGGUF`, `parseGGUF`, `listGGUFTensors`, `extractGGUFConfig`, `generateGGUF`

**Safetensors:** `parseSafetensors`, `readTensor`, `listTensors`, `loadSafetensors`, `loadGPT2Safetensors`, `exportSafetensors`, `saveSafetensors`

**Vision:** `createResNet`, `loadResNet`, `forwardResNet`, `createCLIP`, `loadCLIP`, `forwardVision`, `forwardText`, `clipSimilarity`

**Preprocessing:** `preprocessResNet`, `preprocessCLIP`, `rgbaToChw`, `rgbToChw`, `loadPPM`, `resizeBilinear`, `centerCrop`

**Conv/Pool:** `conv2d`, `maxPool2d`, `avgPool2d`, `batchnorm`, `createBatchNorm`

**Llama ops:** `rope`, `rmsNorm`, `swiglu`, `precomputeRoPE`

**Profiling:** `profile`, `benchmark`, `enableProfiling`, `disableProfiling`, `profileReport`, `memorySnapshot`

**Precision:** `f16Mode`, `cast`, `createLossScaler`

**Checkpoint:** `saveCheckpoint`, `loadCheckpoint`

**Quantization:** `quantizeQ4`, `matmulQ4`, `matmulQ8`
