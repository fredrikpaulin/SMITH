# Smith — Quick Start

Smith gives Bun projects direct access to Apple Silicon GPUs for AI/ML workloads via Metal compute shaders.

## Requirements

- macOS with Apple Silicon (M1–M5)
- Bun runtime
- Xcode Command Line Tools (`xcode-select --install`)

## Build

```sh
bash build.sh
```

This compiles the native Metal bridge (`libsmith.dylib`) and all Metal shaders into `smith.metallib`.

## Basic Usage

```js
import smith from './src/index.js'

// Check GPU
console.log(smith.info().device) // "Apple M1 Pro" etc.

// Create tensors (GPU-backed, zero-copy)
const a = smith.tensor([1, 2, 3, 4], [2, 2])
const b = smith.tensor([5, 6, 7, 8], [2, 2])

// Autograd variables
const va = smith.variable(a, { requiresGrad: true })
const vb = smith.variable(b, { requiresGrad: true })

// Forward pass
const vc = smith.matmul(va, vb)
const loss = smith.sum(vc)

// Backward pass
smith.backward(loss)
console.log(smith.toArray(va.grad)) // gradients on GPU
```

## Training a GPT

```js
import smith from './src/index.js'

const model = smith.createModel({
  vocabSize: 256,
  numLayers: 2,
  numHeads: 2,
  dim: 64,
  maxSeqLen: 128,
})

const params = smith.modelParams(model)
const opt = smith.createAdamW(params, { lr: 1e-3 })

for (let step = 0; step < 100; step++) {
  smith.zeroGrad(params)
  const { logits } = smith.forward(model, inputTokens)
  const loss = smith.crossEntropy(logits, targetTokens)
  smith.backward(loss)
  smith.clipGradNorm(params, 1.0)
  smith.adamwStep(opt)
}
```

## Text Generation

```js
import { generate } from './src/generate.js'

const tokens = generate(model, promptIds, {
  maxTokens: 100,
  temperature: 0.8,
  topK: 40,
})
```

## Checkpoints

```js
import { saveCheckpoint, loadCheckpoint } from './src/checkpoint.js'

await saveCheckpoint(model, 'my-model')   // writes .json + .bin
const loaded = await loadCheckpoint('my-model')
```

## Tokenizer

```js
import { train, encode, decode } from './src/tokenizer.js'

const tok = train(text, 512)
const ids = encode("hello world", tok.merges)
const str = decode(ids, tok.vocab)
```

## Model Registry

Smith includes a model registry for fetching pretrained weights from Hugging Face.

```js
import smith from './src/index.js'

// See what's available
smith.listModels()
// → [{ id: 'whisper-tiny', format: 'ggml', cached: false, ... }, ...]

// Fetch a model (downloads once, then cached in models/<id>/)
await smith.fetchModel('resnet50')

// Resolve the local file path
const path = smith.modelPath('resnet50') // 'models/resnet50/model.safetensors'

// Load it
const { forward } = await smith.loadResNet(path, { variant: 'resnet50' })
```

You can also fetch from direct URLs without a registry entry:

```js
await smith.fetchUrl('https://example.com/weights.bin', { id: 'my-model' })
```

## Memory Management

Use `using()` to scope tensor lifetimes. All tensors allocated inside are freed when the scope exits, unless explicitly retained.

```js
import smith from './src/index.js'

// Intermediates freed automatically
smith.using(() => {
  const x = smith.rand([512, 512])
  const y = smith.rand([512, 512])
  // x and y are freed here
})

// Keep specific tensors alive
const weights = smith.using(() => {
  const w = smith.rand([256, 256])
  smith.retain(w)
  return w
})
// weights is still valid
smith.dispose(weights) // free manually when done
```

## GPU Sampling

Keep the entire sampling pipeline on the GPU. Only 4 bytes per token cross to the CPU.

```js
import { generateGGUF, loadGGUF } from './src/index.js'

const model = await loadGGUF('models/my-model.gguf')
const ids = generateGGUF(model, promptIds, {
  maxTokens: 100,
  temperature: 0.8,
  topK: 40,
  gpuSampling: true,  // penalties, top-K, softmax, and sampling all on GPU
})
```

## Benchmarks

```sh
bun bench/bench.js
```
