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

## Quantization

```js
import { quantizeQ4, matmulQ4 } from './src/ops/quantize.js'
const wq = quantizeQ4(weightTensor)    // f32 → q4
const out = matmulQ4(activations, wq)  // q4 matmul on GPU
```
