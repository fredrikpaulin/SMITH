# Smith

GPU-accelerated tensors, autograd, and ML inference/training for Bun on Apple Silicon.

Smith gives JavaScript direct access to Metal compute shaders through a thin C bridge and Bun's FFI. Unified memory means zero-copy between CPU and GPU — a `Float32Array` in JS and a `device float*` in a Metal shader point to the same physical bytes.

## What it does

- **GPU tensors** backed by Metal shared buffers, accessed as typed arrays (f32, f16)
- **Reverse-mode autograd** (DAG-based, topological sort backward)
- **Fused Metal shaders** for matmul (tiled GEMM), softmax, layernorm, flash attention, AdamW, and more
- **GPT-2 transformer** with multi-head attention, pre-norm blocks, weight tying, and KV cache
- **Convolutions** — direct conv2d, Winograd F(2x2,3x3), im2col+GEMM, pooling, batch normalization
- **Vision models** — ResNet-18/34/50/101/152, CLIP ViT-B/32, ViT-B/16, ViT-L/14
- **Model loading** — GGUF (llama.cpp format: Llama, Phi, GPT-2) and safetensors (torchvision, OpenAI CLIP)
- **Quantization** — Q4 and Q8 matmul for inference, 4-bit weight quantization
- **Mixed precision** — f16 mode with loss scaling
- **Profiling** — per-kernel GPU timing, memory tracking, benchmarking
- **BPE tokenizer**, checkpoint save/load, text generation with temperature/top-k/top-p/repetition penalty

Zero dependencies. No npm packages. Just Bun, Metal, and JavaScript + Metal shaders.

## Requirements

- macOS on Apple Silicon (M1–M5)
- [Bun](https://bun.sh) runtime
- Xcode Command Line Tools (`xcode-select --install`)

## Quick start

```sh
git clone https://github.com/fredrikpaulin/smith.git
cd smith
bash build.sh
bun test tests/
```

`build.sh` compiles the Objective-C Metal bridge (`native/libsmith.dylib`) and all `.metal` shaders into `shaders/smith.metallib`.

## Usage

```js
import smith from './src/index.js'

// GPU info
console.log(smith.info().device) // "Apple M1 Pro"

// Tensors (GPU-backed, zero-copy)
const a = smith.tensor([1, 2, 3, 4], [2, 2])
const b = smith.tensor([5, 6, 7, 8], [2, 2])

// Autograd
const va = smith.variable(a, { requiresGrad: true })
const vb = smith.variable(b, { requiresGrad: true })
const vc = smith.matmul(va, vb)
const loss = smith.sum(vc)
smith.backward(loss)

console.log(smith.toArray(va.grad)) // [[11, 15], [11, 15]]
```

### Train a GPT

```js
const model = smith.createModel({
  vocabSize: 256, numLayers: 2, numHeads: 2,
  dim: 64, maxSeqLen: 128,
})
const params = smith.modelParams(model)
const opt = smith.createAdamW(params, { lr: 1e-3 })

for (let step = 0; step < 100; step++) {
  smith.zeroGrad(params)
  const { logits } = smith.forward(model, tokens)
  const loss = smith.crossEntropy(logits, targets)
  smith.backward(loss)
  smith.clipGradNorm(params, 1.0)
  smith.adamwStep(opt)
}
```

### Load a GGUF model

```js
const llama = await smith.loadGGUF('tinyllama.gguf')
const output = llama.generate([1, 2, 3], {
  maxTokens: 50, temperature: 0.8, topK: 40,
})
```

### Load a ResNet

```js
const { forward } = await smith.loadResNet('resnet50.safetensors', {
  variant: 'resnet50',
})
const input = smith.preprocessResNet(rgbaPixels, width, height)
const logits = forward(smith.variable(input, { requiresGrad: false }), false)
```

### Load CLIP

```js
const clip = await smith.loadCLIP('clip-vit-b-32.safetensors', {
  variant: 'ViT-B/32',
})
const imgEmbed = clip.encodeImage(imageInput)
const txtEmbed = clip.encodeText(tokenIds)
const similarity = clip.similarity(imgEmbed, txtEmbed)
```

### Convolutions

```js
const input = smith.variable(smith.rand([1, 3, 32, 32]), { requiresGrad: true })
const weight = smith.variable(smith.rand([16, 3, 3, 3]), { requiresGrad: true })
const out = smith.conv2d(input, weight, null, { padding: 1 })
// Auto-dispatches: Winograd (3x3 stride 1) > im2col (larger) > direct
```

### Profile GPU work

```js
const p = smith.profile(() => {
  smith.noGrad(() => smith.matmul(a, b))
})
console.log(p.gpuMs, p.kernels)

const b = smith.benchmark('matmul-256', () => {
  smith.noGrad(() => smith.matmul(a, b))
}, { warmup: 3, iterations: 10 })
console.log(b.gpu.median, b.gpu.p95)
```

### Checkpoints and quantization

```js
await smith.saveCheckpoint(model, './my-model')
const loaded = await smith.loadCheckpoint('./my-model')

const wq = smith.quantizeQ4(weightTensor)
const out = smith.matmulQ4(activations, wq)
```

## Architecture

```
JS (Bun)                  Native (C/ObjC)           GPU (Metal)
─────────────             ──────────────             ──────────
src/autograd.js    ──►    native/gpu_bridge.m  ──►   shaders/*.metal
src/ops/*.js              (bun:ffi dlopen)           (smith.metallib)
src/nn.js
src/model.js
src/resnet.js
src/clip.js
src/profile.js
```

All intelligence lives in JavaScript. The native layer is a dumb pipe: allocate buffer, load shader, dispatch compute, wait, return timing.

## Project structure

```
smith/
├── native/          Objective-C Metal bridge (~250 lines)
│   ├── gpu_bridge.h
│   └── gpu_bridge.m
├── shaders/         Metal compute shaders
│   ├── matmul.metal          Tiled GEMM, simple, batched
│   ├── matmul_q4.metal       4-bit quantized matmul
│   ├── matmul_q8.metal       8-bit quantized matmul
│   ├── flash_attention.metal Fused scaled dot-product attention
│   ├── conv2d.metal          Direct 2D convolution + backward
│   ├── conv2d_winograd.metal Winograd F(2x2,3x3) forward + backward
│   ├── im2col.metal          im2col/col2im for GEMM-based conv
│   ├── pool2d.metal          Max/avg pooling
│   ├── batchnorm.metal       Batch normalization
│   ├── rope.metal            Rotary position embeddings
│   ├── rmsnorm.metal         RMS normalization
│   ├── swiglu.metal          Fused SiLU gate
│   ├── softmax.metal         Fused, numerically stable
│   ├── layernorm.metal       Forward + backward
│   ├── activation.metal      relu, gelu, silu, sigmoid, tanh
│   ├── adam.metal             Fused AdamW step
│   ├── elementwise.metal     add, sub, mul, div + broadcasting
│   └── reduce.metal          sum, max — full and per-axis
├── src/
│   ├── device.js       FFI bindings to libsmith.dylib
│   ├── tensor.js       GPU-backed tensors + shape utilities
│   ├── pool.js         Power-of-2 buffer recycling
│   ├── dispatch.js     Shader dispatch (instrumented for profiling)
│   ├── autograd.js     DAG-based reverse-mode AD
│   ├── optim.js        AdamW + cosine schedule + grad clipping
│   ├── nn.js           Linear, MHA, transformer blocks
│   ├── model.js        GPT assembly + weight tying
│   ├── resnet.js       ResNet-18/34/50/101/152 builder + loader
│   ├── clip.js         CLIP ViT + text encoder + loader
│   ├── vision.js       Image preprocessing (resize, crop, normalize)
│   ├── gguf.js         GGUF binary parser + dequantization
│   ├── gguf_loader.js  Llama/Phi/GPT-2 model loading from GGUF
│   ├── gguf_cache.js   KV cache + cached generation
│   ├── safetensors.js  Safetensors parser + GPT-2 weight loader
│   ├── generate.js     Text generation + sampling
│   ├── profile.js      Per-kernel GPU profiling + benchmarking
│   ├── checkpoint.js   Save/load model weights
│   ├── tokenizer.js    BPE tokenizer
│   ├── dtype.js        f16 encode/decode
│   ├── f16mode.js      Mixed precision mode
│   ├── index.js        Public API
│   └── ops/            Per-op GPU dispatch wrappers
├── tests/              Tests across all phases
├── bench/              Benchmark suite
├── docs/               API reference
├── pr/                 Dev notes for blog
└── build.sh            One-command build
```

## Metal shaders

| Shader | What it does |
|--------|-------------|
| `matmul_f32` | Tiled GEMM: 32x32 tiles, shared memory, 4x4 per-thread sub-tiles |
| `matmul_q4` | 4-bit dequant-fused matmul with per-group scale/zero |
| `matmul_q8` | 8-bit quantized matmul with f16 per-block scale |
| `flash_attention` | Fused scaled dot-product attention (causal + non-causal) |
| `conv2d_forward/backward` | Direct 2D convolution with groups, dilation, stride |
| `conv2d_winograd` | Winograd F(2x2,3x3): 2.25x fewer multiplications for 3x3 kernels |
| `im2col/col2im` | Rearrange patches for GEMM-based convolution on larger kernels |
| `pool2d` | Max pooling (with argmax) and average pooling |
| `batchnorm` | Training + inference mode batch normalization |
| `rope` | Rotary position embeddings (forward + backward) |
| `rmsnorm` | Llama-style RMS normalization |
| `swiglu` | Fused SiLU(gate) * up |
| `softmax_forward` | Fused max-exp-normalize per row via shared memory reductions |
| `layernorm` | Fused mean-variance-normalize-scale+shift, with saved xhat |
| `adamw_step` | Fused moment update + bias correction + weight decay |
| `elementwise_*` | add, sub, mul, div, scale, neg, fill — with broadcasting |
| `activation_*` | relu, gelu, silu, sigmoid, tanh — forward and backward |
| `reduce_sum/max` | Full parallel reduction and per-axis variants |

## Key design decisions

**Unified memory, zero copy.** Apple Silicon shares memory between CPU and GPU. Smith allocates Metal buffers in shared mode and wraps them as typed arrays via `toArrayBuffer()`. Both JS and shaders read/write the same bytes. No staging buffers, no upload queues.

**DAG autograd.** Each variable carries `_deps` and `_backward`. Topological sort on `backward()` handles weight tying, residual connections, and any DAG structure.

**Thin C bridge.** The native layer is ~250 lines of Objective-C exposing ~20 flat C functions. No ObjC types cross the FFI boundary — just opaque pointers and timing structs.

**3-way convolution dispatch.** `conv2d()` auto-selects: Winograd for 3x3 stride-1 (2.25x fewer multiplies), im2col+GEMM for larger kernels (reuses existing tiled matmul), direct for 1x1 pointwise. No user code changes needed.

**Two-phase GGUF generation.** Prefill processes the full prompt via flash attention, then decode generates tokens one-by-one with cached K/V. The KV cache uses pre-allocated fixed-size buffers with no GC pressure.

**Zero-overhead profiling.** When disabled, profiling adds a single boolean check per dispatch. When enabled, every kernel dispatch records GPU start/end time from Metal command buffers.

## Tests

```sh
bun test tests/
```

## Lineage

Smith's autograd, backward formulas, transformer architecture, tokenizer, and training loop are ported from **TinyFormer** — a from-scratch GPT implementation in pure JavaScript. TinyFormer runs on CPU with `Float32Array` loops; Smith replaces the inner compute with Metal GPU shaders while keeping the same JS orchestration layer.

## License

[MIT](LICENSE)
