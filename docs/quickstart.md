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

## Benchmarks

```sh
bun bench/bench.js
```
