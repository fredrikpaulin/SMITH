# Best Practices for Using Smith

This guide covers patterns and pitfalls for integrating Smith into your own Bun projects. It assumes you've built the native library (`bash build.sh`) and can `import smith from './src/index.js'`.

## Project Setup

Smith requires macOS on Apple Silicon, Bun, and Xcode Command Line Tools. Point your import at Smith's `src/index.js` — everything is exported from a single flat namespace. There are no sub-modules to wire up.

```js
import smith from '../smith/src/index.js'
```

Smith loads `libsmith.dylib` and `smith.metallib` at import time via `bun:ffi`. If the build artifacts are missing, the process crashes immediately with a dlopen error. Run `bash build.sh` once before your first import, and again after pulling changes that touch `native/` or `shaders/`.

Bun loads `.env` automatically — no dotenv needed. If your project has GPU-related config (like max sequence length or model paths), plain environment variables work fine.

## Tensors and Variables

Smith has two core types. Tensors are raw GPU-backed storage. Variables wrap tensors with autograd tracking. Every operation in the autograd system — `matmul`, `add`, `relu`, `softmax`, and so on — takes and returns Variables, not tensors.

This means you need to wrap raw tensors before passing them to ops:

```js
// Wrong — ops expect Variables
const t = smith.rand([4, 8])
const out = smith.relu(t) // TypeError

// Right
const v = smith.variable(t, { requiresGrad: false })
const out = smith.relu(v)
```

Use `requiresGrad: true` for trainable weights and `requiresGrad: false` for inputs, labels, and frozen weights. When creating trainable parameters from scratch, `smith.param(shape, initFn)` is the shorthand — it returns a Variable with `requiresGrad: true` already set. The init function takes no arguments and returns a scalar value; do not pass a tensor to `param`.

To read data back from the GPU, `smith.toArray(tensor)` returns a nested JS array matching the tensor's shape. For a flat `Float32Array` view into GPU memory, access `tensor.data` directly — this is zero-copy.

## Memory and Performance

Smith uses Metal shared memory, so CPU and GPU read the same bytes without copying. This is the biggest performance advantage on Apple Silicon, but it means you need to think about buffer lifetimes.

Smith recycles GPU buffers internally through a power-of-2 pool. You generally don't need to manage memory manually, but for long-running processes you can call `smith.poolDrain()` to release unused buffers back to the system. `smith.poolStats()` shows current allocation counts.

For inference, always wrap your forward pass in `smith.noGrad()`. Without it, every op builds a computation graph node with backward closures and dependency arrays. On a 12-layer transformer that's thousands of allocations per forward pass that serve no purpose if you never call `backward`.

```js
// Inference — skip graph construction
smith.noGrad(() => {
  const { logits } = smith.forward(model, tokens)
  // ...
})
```

For training, call `smith.zeroGrad(params)` at the start of each step. Gradients accumulate by default — if you forget `zeroGrad`, your updates compound across steps and training diverges.

## Training Loop Structure

A complete training step follows this sequence: zero gradients, forward pass, compute loss, backward pass, clip gradients, step optimizer. The ordering matters.

```js
smith.zeroGrad(params)
const { logits } = smith.forward(model, tokens)
const loss = smith.crossEntropy(logits, targets)
smith.backward(loss)
smith.clipGradNorm(params, 1.0)
smith.adamwStep(opt)
```

Gradient clipping before the optimizer step prevents exploding gradients. For transformers, a max norm of 1.0 is a reasonable default. The cosine learning rate schedule pairs well with AdamW:

```js
const sched = smith.createSchedule({
  warmupSteps: 100, totalSteps: 1000,
  maxLr: 1e-3, minLr: 1e-5,
})
// In the loop:
opt.lr = smith.getLr(sched, step)
```

## Loading Pretrained Models

Smith reads two formats: GGUF (llama.cpp/ollama models) and safetensors (HuggingFace/torchvision/OpenAI).

For GGUF, `loadGGUF` returns an object with `generate`, `forward`, and cache management functions. The supported architectures for inference are `llama`, `phi`, `phi2`, `phi3`, and `gpt2`. For other architectures (like Nemotron-H or Mamba hybrids), you can still parse metadata and dequantize individual tensors with `parseGGUF` + `extractGGUFConfig` + `listGGUFTensors` — you'd just need to write the forward pass yourself.

```js
const model = await smith.loadGGUF('model.gguf')
const output = model.generate(promptTokenIds, {
  maxTokens: 100, temperature: 0.8, topK: 40,
})
```

For vision models, `loadResNet` and `loadCLIP` load from safetensors format. Both expect specific weight naming conventions (torchvision for ResNet, OpenAI for CLIP). The loaders handle weight transposition automatically — torchvision FC layers store weights as `[outDim, inDim]` while Smith's linear layer expects `[inDim, outDim]`.

When loading weights manually, the GGUF format stores tensors in the opposite dimension order from PyTorch. Smith's loaders handle this, but if you're writing a custom loader, transpose at load time.

## Convolutions

All conv and pooling ops use NCHW layout: `[batch, channels, height, width]`. This is the Metal-native layout and matches PyTorch's default.

Smith auto-dispatches convolutions through three paths based on kernel size and stride. You don't pick the path — `conv2d` chooses for you:

- 3x3 kernels with stride 1 and no dilation go through Winograd (2.25x fewer multiplications)
- Larger kernels or strided 3x3 go through im2col + the existing tiled GEMM
- 1x1 pointwise convolutions fall through to direct dispatch

Batch normalization requires creating the layer object first, then passing it to the op. The layer tracks running mean/variance across training steps:

```js
const bn = smith.createBatchNorm(numChannels)
const out = smith.batchnorm(input, bn, true)  // true = training
const out = smith.batchnorm(input, bn, false) // inference — uses running stats
```

## Text Generation

For single-use generation, `smith.generate` recomputes the full context at each step. This is simple but O(n²) in sequence length.

For interactive or streaming use, `smith.generateCached` uses a KV cache — it processes the prompt once via flash attention (prefill), then generates tokens one at a time with O(1) per step. The cache uses pre-allocated fixed-size buffers with no GC pressure during generation.

Both support temperature, top-k, top-p, repetition penalty, and an `onToken` callback for streaming:

```js
smith.generateCached(model, promptIds, {
  maxTokens: 200, temperature: 0.7,
  topK: 40, topP: 0.9,
  repetitionPenalty: 1.1, eosToken: 2,
}, {
  onToken: (token, step) => {
    process.stdout.write(decode(token))
    if (someCondition) return true // return true to stop early
  },
})
```

## Image Preprocessing

Smith includes CPU-side image preprocessing with no dependencies. The pipeline converts raw pixel data to normalized CHW tensors ready for vision models.

For pixel input, you need either RGBA bytes (like from a canvas) or RGB bytes (like from a PPM file). Smith provides `rgbaToChw` and `rgbToChw` to convert to CHW `Float32Array` in [0, 1] range.

The `preprocessResNet` and `preprocessCLIP` functions handle the full pipeline — resize, center crop to 224x224, and normalize with the appropriate constants (ImageNet means/stds for ResNet, CLIP means/stds for CLIP). If you need to load an image file without external libraries, `loadPPM` reads binary PPM (P6 format).

## Profiling

`smith.profile(fn)` wraps a function and captures wall-clock time, GPU time, dispatch count, per-kernel breakdown, and memory delta. It resets profiling state before the call and restores it after, so it's safe to nest or use in tests without side effects.

```js
const p = smith.profile(() => {
  return smith.noGrad(() => smith.forward(model, tokens))
})
console.log(`GPU: ${p.gpuMs.toFixed(2)}ms across ${p.dispatches} dispatches`)
for (const k of p.kernels) {
  console.log(`  ${k.kernel}: ${k.calls}x, ${k.totalMs.toFixed(2)}ms (${k.pct.toFixed(1)}%)`)
}
```

When disabled (the default), profiling adds a single boolean check per dispatch — effectively zero overhead. `smith.benchmark` runs warmup iterations before measuring, and reports mean, median, p95, min, max, and stddev for both CPU and GPU time.

## Mixed Precision

`smith.f16Mode(true)` makes all new tensors default to `f16`. This halves memory usage but can cause training instability without loss scaling. Use `smith.createLossScaler` to handle the dynamic scale factor:

```js
smith.f16Mode(true)
const scaler = smith.createLossScaler({ initScale: 65536 })

// In the training loop:
const scaledLoss = scaler.scaleUp(loss)
smith.backward(scaledLoss)
const ok = scaler.unscale(grads)
if (ok) smith.adamwStep(opt)
scaler.update(ok)
```

The scaler increases scale when gradients are finite and decreases when they overflow. This keeps training stable without manual tuning. Use `smith.cast(tensor, 'f32')` to upcast for operations that need full precision (like loss computation).

## Testing

Write tests with `bun:test`. Smith tests require macOS + Apple Silicon since they dispatch real Metal shaders. Structure tests to verify both forward and backward passes — autograd bugs often hide in the backward path.

```js
import { test, expect } from 'bun:test'
import smith from '../smith/src/index.js'

test('custom layer backward', () => {
  const x = smith.variable(smith.rand([2, 8]), { requiresGrad: true })
  const w = smith.param([8, 4], () => Math.random() * 0.1)
  const out = smith.matmul(x, w)
  const loss = smith.sum(out)
  smith.backward(loss)
  // Verify gradients exist and have correct shapes
  expect(x.grad.shape).toEqual([2, 8])
  expect(w.grad.shape).toEqual([8, 4])
})
```

For numerical gradient checking, compute the finite-difference gradient and compare against the autograd gradient. A tolerance of 1e-3 to 1e-2 is typical for f32; f16 needs wider tolerance (1e-1 to 1e-2).

## Common Mistakes

These come from real bugs encountered during Smith's development.

`embedding(indices, weight)` takes indices first, weight second. This is the opposite of PyTorch's `nn.Embedding`. Getting it backwards produces a confusing "undefined is not an object" error inside autograd because it tries to read `.data.shape` on the indices array.

Forgetting `noGrad` during inference doesn't crash — it just silently builds a computation graph you never use, wasting memory proportional to the model size times the number of ops. For a 12-layer transformer generating 100 tokens without `noGrad`, that's roughly 50,000 graph nodes allocated and never freed.

Passing raw tensors to autograd ops is the most common type error. If you see "Cannot read properties of undefined (reading 'shape')" inside an op, you probably passed a tensor where a Variable was expected.

Convolution inputs must be `[batch, channels, height, width]`. Passing `[height, width, channels]` (the web/canvas convention) silently produces wrong results because the shader interprets the dimensions differently.
