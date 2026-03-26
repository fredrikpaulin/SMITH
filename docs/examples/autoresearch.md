# Autoresearch GPT Pretraining Example

Smith's second example project: from-scratch GPT pretraining ported from Karpathy's autoresearch. Trains a small language model on public domain text using MuonAdamW, GQA flash attention with sliding windows, RoPE, value embeddings, and logit soft-capping. Runs entirely on Apple Silicon via Metal.

This document covers setup, usage, architecture, and how the example exercises Smith's training stack.

## Requirements

- macOS with Apple Silicon (M1–M5)
- Bun runtime
- Smith built (`bash build.sh`)

## Quick Start

```bash
# Build Smith
cd /path/to/smith && bash build.sh

# Prepare data (downloads public domain text, trains BPE tokenizer)
bun examples/autoresearch/prepare.js --out data/autoresearch --vocab 4096

# Train for 5 minutes
bun examples/autoresearch/train.js --data data/autoresearch --time-budget 300
```

## Data Preparation

`prepare.js` downloads 10 public domain books from Project Gutenberg, strips headers/footers, trains a BPE tokenizer, and saves tokenized train/val splits as `.bin` files.

```bash
bun examples/autoresearch/prepare.js [options]

  --out <dir>       Output directory (default: data/autoresearch)
  --vocab <n>       Vocabulary size (default: 4096)
  --chars <n>       Max total characters to use (default: unlimited)
```

The tokenized output contains `train.bin`, `val.bin` (uint16 token arrays), and `tokenizer.json`.

## Training

`train.js` runs the training loop: forward, backward, LR schedule, optimizer step.

```bash
bun examples/autoresearch/train.js [options]

  --data <dir>      Data directory from prepare step
  --depth <n>       Number of transformer layers (default: 4)
  --dim <n>         Model dimension (default: 256)
  --seq-len <n>     Sequence length (default: 512)
  --head-dim <n>    Attention head dimension (default: 64)
  --vocab <n>       Vocabulary size (default: 4096)
  --batch-size <n>  Sequences per optimizer step (default: 4)
  --time-budget <s> Training time limit in seconds (default: 300)
  --lr <f>          Peak learning rate (default: 0.01)
```

Training logs per-step loss and ends with a validation BPB (bits per byte) evaluation.

## Model Architecture

The GPT model follows the autoresearch reference closely:

- **Embeddings:** Token embedding with initial RMSNorm.
- **Residual scaling:** Per-layer `residLambda * x + x0Lambda * x0` mixing (x0 = post-embedding state).
- **Attention:** Pre-norm → Q/K/V projections → 3D reshape → RoPE → QK-norm (per-head RMSNorm) → GQA flash attention with sliding window → output projection.
- **Value embeddings (ResFormer):** On alternating layers, a sigmoid-gated embedding is added to V: `v += 2 * sigmoid(gate) * veEmbed[tokens]`.
- **MLP:** Pre-norm → linear → ReluSquared → linear.
- **Output:** Final RMSNorm → LM head → logit soft-capping (`15 * tanh(logits / 15)`).

Default config targets Apple Silicon: depth=4, dim=256, seq_len=512, vocab=4096.

## Optimizer

MuonAdamW with the reference parameter grouping:

- **Muon path** (2D matrix params): Newton-Schulz polar decomposition, Nesterov momentum, NorMuon variance reduction, cautious weight decay.
- **AdamW path** (embeddings, scalars, norms, LM head): Standard AdamW.

LR schedule: warmup (250 steps) → cosine decay → warmdown (last 1/7 of training). Muon momentum ramps 0.85→0.95 over 300 steps. Weight decay decays linearly to zero.

## Module API

### model.js

```js
import { createModel, initWeights, forward, setupOptimizer, countModelParams, allParams } from './model.js'

const model = createModel({ seqLen: 512, vocabSize: 4096, nLayer: 4, nHead: 4, nKVHead: 4, nEmbd: 256 })
initWeights(model)

const { logits, loss } = forward(model, tokenArray, targetArray)
// logits: Variable [T, vocabSize], loss: Variable scalar

const opt = setupOptimizer(model, { matrixLr: 0.01, weightDecay: 0.01 })
```

### data.js

```js
import { loadTokens, createDataLoader, evaluateBPB, prepareData } from './data.js'

const tokens = loadTokens('data/autoresearch/train.bin')
const loader = createDataLoader(tokens, 512)
const { input, target } = loader.next()  // both Int32Array of length seqLen

const bpb = evaluateBPB(perTokenLosses, tokenIds, tokenizer)
```

## Smith Ops Used

This example exercises the following Smith autograd operations:

- `embedding` — Token and value embedding lookup
- `rmsNorm` — Pre-attention, pre-MLP, and QK normalization
- `rope` — Rotary position embeddings (via tiled tables for multi-head 2D kernel)
- `flashAttention` — GQA flash attention with sliding window and causal masking
- `reluSquared` — MLP activation
- `tanh`, `sigmoid` — Logit soft-capping and VE gating
- `matmul` — All linear projections
- `crossEntropy` — Loss computation
- `reshape`, `transpose` — 2D↔3D conversion for attention heads
- `backward`, `zeroGrad` — Autograd
- `createMuonAdamW`, `muonAdamWStep` — Optimizer

## Tests

```bash
bun test examples/autoresearch/tests/
```

- **model.test.js** — Model creation, weight init, forward logit shape, loss, backward gradients (tests cProj/cMlpProj/lmHead since zero-init projections block upstream gradients on first step), optimizer step finiteness, loss reduction over 10 steps, VE layer placement, soft-capping bounds, window pattern, GQA forward/backward (nKVHead < nHead), T=1 single token edge case, T=seqLen full length, full gradient flow after one optimizer step (verifies cQ/cFc get gradients once projections are non-zero).
- **data.test.js** — Loader shapes, position advancement, wraparound, reset, BPB computation, special token handling, totalTokens property.
- **research.test.js** — Metric parsing from training output, experiment JSON serialization roundtrip, best-experiment selection, status computation, markdown log format, CLI argument parsing.

## Autonomous Research Loop

The autoresearch example includes an autonomous experiment loop powered by Claude Code. An AI agent runs training experiments, keeps improvements, discards regressions, and iterates indefinitely.

### Quick Start

```bash
cd examples/autoresearch
bash start.sh
```

The start script checks prerequisites (Bun, Smith native lib, Claude Code), prepares data if needed, and launches Claude Code. The agent walks you through creating a branch, establishing a baseline, and starting the experiment loop.

Once running, the agent loops indefinitely — leave it overnight and review `results/research_log.md` in the morning.

To set up manually instead:

```bash
bun examples/autoresearch/prepare.js         # prepare data
cd examples/autoresearch && claude "start"    # launch agent
```

### How It Works

The loop follows Karpathy's autoresearch pattern: **one machine, one file, one metric**.

1. The agent reads experiment history and forms a hypothesis
2. Edits `model.js` and/or `train.js` with an experimental change
3. Commits the change and runs training via `research.js`
4. If val_bpb improved, the commit stays. If not, `git reset --hard HEAD~1`
5. Repeat forever — the agent never stops until manually interrupted

### Research Runner

`research.js` handles experiment execution and tracking:

```bash
# Run an experiment
bun examples/autoresearch/research.js run --tag "increase depth to 6"

# Show last result
bun examples/autoresearch/research.js last

# Show experiment history
bun examples/autoresearch/research.js status

# Show best configuration
bun examples/autoresearch/research.js best
```

Results are logged to both `results/experiments.json` (machine-readable) and `results/research_log.md` (human-readable).

### Files

| File | Modifiable | Purpose |
|------|-----------|---------|
| `model.js` | Yes | Model architecture |
| `train.js` | Yes | Training loop, optimizer, hyperparameters |
| `data.js` | No | Data loading, BPB evaluation |
| `prepare.js` | No | Data preparation |
| `research.js` | No | Experiment runner and tracking |
| `CLAUDE.md` | No | Agent instructions |
