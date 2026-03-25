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
| `createTransformerBlock(dim, heads, ffnDim?)` | Pre-norm block |
| `transformerBlock(x, block, mask)` | Attention + FFN + residuals |
| `createCausalMask(seqLen)` | Upper-triangle -Infinity mask |

## Model

```js
const model = createModel({ vocabSize, numLayers, numHeads, dim, maxSeqLen })
const { logits } = forward(model, tokenIds)
const params = modelParams(model)
const info = modelInfo(model)
```

Preset configs: `CONFIGS.tiny`, `CONFIGS.small`, `CONFIGS.medium`.

## Generation

```js
import { generate } from './src/generate.js'
const ids = generate(model, promptIds, { maxTokens, temperature, topK, topP, repetitionPenalty })
```

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
