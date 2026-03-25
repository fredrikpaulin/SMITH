# Smith

GPU-accelerated tensors, autograd, and transformer training for Bun on Apple Silicon.

Smith gives JavaScript direct access to Metal compute shaders through a thin C bridge and Bun's FFI. Unified memory means zero-copy between CPU and GPU — a `Float32Array` in JS and a `device float*` in a Metal shader point to the same physical bytes.

## What it does

- **GPU tensors** backed by Metal shared buffers, accessed as typed arrays
- **Reverse-mode autograd** (DAG-based, ported from [TinyFormer](https://github.com/user/tinyformer))
- **Fused Metal shaders** for matmul (tiled GEMM), softmax, layernorm, AdamW, and more
- **GPT-2 style transformer** with multi-head attention, pre-norm blocks, and weight-tied output head
- **BPE tokenizer**, checkpoint save/load, text generation with temperature/top-k/top-p
- **4-bit quantization** for inference (`quantizeQ4` + `matmulQ4`)

Zero dependencies. No npm packages. Just Bun, Metal, and ~2500 lines of JavaScript + ~500 lines of Metal shaders.

## Requirements

- macOS on Apple Silicon (M1–M5)
- [Bun](https://bun.sh) runtime
- Xcode Command Line Tools (`xcode-select --install`)

## Quick start

```sh
git clone https://github.com/user/smith.git
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

### Generate text

```js
import { generate } from './src/generate.js'

const output = generate(model, promptIds, {
  maxTokens: 100, temperature: 0.8, topK: 40,
})
```

### Checkpoints

```js
import { saveCheckpoint, loadCheckpoint } from './src/checkpoint.js'

await saveCheckpoint(model, './my-model')
const loaded = await loadCheckpoint('./my-model')
```

### Quantize weights for faster inference

```js
const wq = smith.quantizeQ4(weightTensor)  // f32 → 4-bit
const out = smith.matmulQ4(activations, wq) // dequant-fused matmul on GPU
```

## Architecture

```
JS (Bun)                  Native (C/ObjC)           GPU (Metal)
─────────────             ──────────────             ──────────
src/autograd.js    ──►    native/gpu_bridge.m  ──►   shaders/*.metal
src/ops/*.js              (bun:ffi dlopen)           (smith.metallib)
src/nn.js
src/model.js
```

Seven layers, bottom to top: Metal shaders → Objective-C bridge → FFI bindings → buffer pool → tensor module → ops + autograd → model/nn/training.

All intelligence lives in JavaScript. The native layer is a dumb pipe: allocate buffer, load shader, dispatch compute, wait.

## Project structure

```
smith/
├── native/          Objective-C Metal bridge
│   ├── gpu_bridge.h
│   └── gpu_bridge.m
├── shaders/         Metal compute shaders
│   ├── elementwise.metal
│   ├── matmul.metal       (tiled GEMM, simple, batched)
│   ├── matmul_q4.metal    (4-bit quantized)
│   ├── activation.metal   (relu, gelu, silu, sigmoid, tanh)
│   ├── reduce.metal       (sum, max — full and per-axis)
│   ├── softmax.metal      (fused, numerically stable)
│   ├── layernorm.metal    (forward + backward)
│   └── adam.metal         (fused AdamW step)
├── src/
│   ├── device.js          FFI bindings to libsmith.dylib
│   ├── tensor.js          GPU-backed tensors + shape utilities
│   ├── pool.js            Power-of-2 buffer recycling
│   ├── dtype.js           f16 encode/decode
│   ├── dispatch.js        Shader dispatch helper
│   ├── autograd.js        DAG-based reverse-mode AD
│   ├── optim.js           AdamW + cosine schedule + grad clipping
│   ├── nn.js              Linear, MHA, transformer blocks
│   ├── model.js           GPT assembly + weight tying
│   ├── generate.js        Text generation + sampling
│   ├── checkpoint.js      Save/load model weights
│   ├── tokenizer.js       BPE tokenizer
│   ├── index.js           Public API
│   └── ops/               Per-op GPU dispatch wrappers
├── tests/           58 tests across 8 files
├── bench/           Benchmark suite
├── docs/            API reference + quick start
└── build.sh         One-command build
```

## Metal shaders

| Shader | What it does |
|--------|-------------|
| `matmul_f32` | Tiled GEMM: 32×32 tiles, threadgroup shared memory, 4×4 per-thread sub-tiles |
| `matmul_q4` | 4-bit dequant-fused matmul with per-group (32 element) scale/zero |
| `softmax_forward` | Fused max → exp → normalize per row via shared memory reductions |
| `layernorm_forward/backward` | Fused mean → variance → normalize → scale+shift, with saved xhat |
| `adamw_step` | Fused moment update + bias correction + weight decay in one dispatch |
| `elementwise_*` | add, sub, mul, div, scale, neg, fill — with broadcasting variants |
| `activation_*` | relu, gelu, silu, sigmoid, tanh — forward and backward |
| `reduce_sum/max` | Full parallel reduction and per-axis variants |

## Key design decisions

**Unified memory, zero copy.** Apple Silicon shares memory between CPU and GPU. Smith allocates Metal buffers in shared mode and wraps them as typed arrays via `toArrayBuffer()`. Both JS and shaders read/write the same bytes. No staging buffers, no upload queues.

**DAG autograd.** Each variable carries `_deps` and `_backward`. Topological sort on `backward()` handles weight tying, residual connections, and any DAG structure. Ported from TinyFormer's battle-tested autograd.

**Thin C bridge.** The native layer is ~200 lines of Objective-C exposing ~20 flat C functions. No ObjC types cross the FFI boundary — just opaque pointers. JS never touches `MTLBuffer` directly.

**Fused shaders where it matters.** Softmax, layernorm, and AdamW are fused into single-dispatch kernels. Five-kernel-launch softmax has ~25μs of FFI overhead; the fused version has ~5μs.

## Benchmarks

```sh
bun bench/bench.js
```

Measures matmul, softmax, layernorm, elementwise ops, Q4 matmul, and GPT forward pass with median/mean/min timing.

## Tests

```sh
bun test tests/
```

58 tests covering tensor creation, matmul correctness, autograd gradients, buffer pooling, f16 roundtrip, optimizer convergence, softmax/layernorm/cross-entropy, tokenizer, generation, checkpoint roundtrip, and Q4 quantization accuracy.

## Lineage

Smith's autograd, backward formulas, transformer architecture, tokenizer, and training loop are ported from **TinyFormer** — a from-scratch GPT implementation in pure JavaScript. TinyFormer runs on CPU with `Float32Array` loops; Smith replaces the inner compute with Metal GPU shaders while keeping the same JS orchestration layer.

## License

[MIT](LICENSE)
