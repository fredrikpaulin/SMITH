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
| `multiHeadAttentionFlash(x, layer, causal?)` | Flash attention forward (O(n) memory) |
| `multiHeadAttentionCached(x, layer, cache)` | Single-token forward, appends K/V to cache |
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

## GGUF Import

Load models from GGUF files (llama.cpp, ollama format). Supports Llama, Phi, and GPT-2 architectures with Q4_0, Q4_1, Q8_0, F16, and F32 weight types.

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

Supported architectures: `llama` (Llama 2/3, Mistral, CodeLlama, TinyLlama), `phi`/`phi2`/`phi3`, `gpt2`.

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
