#!/usr/bin/env bun
// examples/autoresearch/train.js
// Train a GPT model from scratch using MuonAdamW on Apple Silicon.
// Ported from Karpathy's autoresearch train.py.
//
// Usage:
//   bun examples/autoresearch/prepare.js   # first time only
//   bun examples/autoresearch/train.js [options]
//
// Options:
//   --depth 4           Number of transformer layers
//   --dim 256           Model dimension (rounded to head_dim multiple)
//   --seq-len 512       Context length
//   --head-dim 64       Head dimension
//   --vocab 4096        Vocabulary size
//   --batch-size 8192   Tokens per step (seq_len * num_sequences)
//   --time-budget 60    Training time in seconds
//   --lr 0.02           Matrix (Muon) learning rate
//   --data data/        Path to prepared data directory

import { parseArgs } from 'node:util'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import smith from '../../src/index.js'
import { createModel, initWeights, forward, setupOptimizer, countModelParams, allParams } from './model.js'
import { loadTokens, createDataLoader, evaluateBPB } from './data.js'
import { load as loadTokenizer } from '../../src/tokenizer.js'

const { values: args } = parseArgs({
  options: {
    depth: { type: 'string', default: '2' },
    dim: { type: 'string', default: '128' },
    'seq-len': { type: 'string', default: '512' },
    'head-dim': { type: 'string', default: '32' },
    vocab: { type: 'string', default: '4096' },
    'batch-size': { type: 'string', default: '16384' },
    'time-budget': { type: 'string', default: '60' },
    lr: { type: 'string', default: '0.04' },
    data: { type: 'string', default: join(import.meta.dir, 'data') },
  },
})

// --- Configuration ---

const DEPTH = parseInt(args.depth)
const TARGET_DIM = parseInt(args.dim)
const SEQ_LEN = parseInt(args['seq-len'])
const HEAD_DIM = parseInt(args['head-dim'])
const VOCAB_SIZE = parseInt(args.vocab)
const TOTAL_BATCH_SIZE = parseInt(args['batch-size'])
const TIME_BUDGET = parseInt(args['time-budget'])
const MATRIX_LR = parseFloat(args.lr)
const DATA_DIR = args.data

// Derived
const nEmbd = Math.ceil(TARGET_DIM / HEAD_DIM) * HEAD_DIM
const nHead = Math.floor(nEmbd / HEAD_DIM)

// Optimization hyperparams (matching reference defaults, scaled for small model)
const EMBEDDING_LR = 0.6
const UNEMBEDDING_LR = 0.004
const SCALAR_LR = 0.5
const WEIGHT_DECAY = 0.2
const ADAM_BETAS = [0.8, 0.95]
const WARMUP_RATIO = 0.0
const WARMDOWN_RATIO = 0.0

// --- Data ---

if (!existsSync(join(DATA_DIR, 'train.bin'))) {
  console.error(`Data not found at ${DATA_DIR}. Run prepare.js first.`)
  process.exit(1)
}

const trainTokens = await loadTokens(join(DATA_DIR, 'train.bin'))
const valTokens = await loadTokens(join(DATA_DIR, 'val.bin'))
const tokenizer = await loadTokenizer(join(DATA_DIR, 'tokenizer.json'))
const seqsPerStep = Math.floor(TOTAL_BATCH_SIZE / SEQ_LEN) || 1

console.log(`Train tokens: ${trainTokens.length.toLocaleString()}`)
console.log(`Val tokens: ${valTokens.length.toLocaleString()}`)
console.log(`Sequences per step: ${seqsPerStep}`)

// --- Model ---

const config = {
  seqLen: SEQ_LEN,
  vocabSize: VOCAB_SIZE,
  nLayer: DEPTH,
  nHead,
  nKVHead: nHead,  // full MHA (no GQA for small models)
  nEmbd,
  windowPattern: 'SSSL',
}

console.log(`\nModel config:`, config)

const model = createModel(config)
initWeights(model)

const nParams = countModelParams(model)
console.log(`Parameters: ${(nParams / 1e6).toFixed(2)}M`)

// --- Optimizer ---

const optimizer = setupOptimizer(model, {
  matrixLr: MATRIX_LR,
  embeddingLr: EMBEDDING_LR,
  unembeddingLr: UNEMBEDDING_LR,
  scalarLr: SCALAR_LR,
  weightDecay: WEIGHT_DECAY,
  adamBetas: ADAM_BETAS,
})

// --- LR schedule ---

function getLrMultiplier(progress) {
  if (progress < WARMUP_RATIO) return WARMUP_RATIO > 0 ? progress / WARMUP_RATIO : 1.0
  if (progress < 1.0 - WARMDOWN_RATIO) return 1.0
  return (1.0 - progress) / WARMDOWN_RATIO
}

function getMuonMomentum(step) {
  const frac = Math.min(step / 300, 1)
  return (1 - frac) * 0.85 + frac * 0.95
}

// --- Training loop ---

console.log(`\nTime budget: ${TIME_BUDGET}s`)
console.log('Starting training...\n')

const trainLoader = createDataLoader(trainTokens, SEQ_LEN)
const params = allParams(model)

let totalTime = 0
let smoothLoss = 0
let step = 0

while (true) {
  const t0 = performance.now()

  // Accumulate gradients over multiple sequences
  let stepLoss = 0
  for (let seq = 0; seq < seqsPerStep; seq++) {
    const { input, target } = trainLoader.next()
    const { loss } = forward(model, input, target)
    stepLoss += loss.data.data[0]

    // Scale gradient by 1/seqsPerStep for averaging
    const scaledLoss = smith.scale(loss, 1.0 / seqsPerStep)
    smith.backward(scaledLoss)
  }
  stepLoss /= seqsPerStep

  // LR schedule
  const progress = Math.min(totalTime / TIME_BUDGET, 1.0)
  const lrm = getLrMultiplier(progress)
  const muonMom = getMuonMomentum(step)
  const wd = WEIGHT_DECAY * (1 - progress)

  for (const g of optimizer.groups) {
    g.lr = g.initialLr * lrm
    if (g.kind === 'muon') {
      g.momentum = muonMom
      g.weightDecay = wd
    }
  }

  smith.muonAdamWStep(optimizer)
  smith.zeroGrad(params)

  const dt = (performance.now() - t0) / 1000

  // Don't count first few steps (warmup)
  if (step > 2) totalTime += dt

  // NaN check
  if (!isFinite(stepLoss) || stepLoss > 100) {
    console.log(`\nFAIL: loss=${stepLoss} at step ${step}`)
    process.exit(1)
  }

  // Logging
  const ema = 0.9
  smoothLoss = ema * smoothLoss + (1 - ema) * stepLoss
  const debiased = smoothLoss / (1 - Math.pow(ema, step + 1))
  const tokSec = Math.round(TOTAL_BATCH_SIZE / dt)
  const pct = (100 * progress).toFixed(1)
  const remaining = Math.max(0, TIME_BUDGET - totalTime)

  process.stdout.write(`\rstep ${String(step).padStart(4)} (${pct}%) | loss: ${debiased.toFixed(4)} | lrm: ${lrm.toFixed(2)} | dt: ${(dt * 1000).toFixed(0)}ms | tok/s: ${tokSec.toLocaleString()} | left: ${remaining.toFixed(0)}s   `)

  step++

  if (step > 2 && totalTime >= TIME_BUDGET) break
}

console.log('\n')

// --- Validation ---

console.log('Evaluating on validation set...')
const valLoader = createDataLoader(valTokens, SEQ_LEN)
const evalSteps = Math.min(Math.floor(valTokens.length / SEQ_LEN), 20)

let valLossSum = 0
let valTokenCount = 0
const tokenLosses = []
const tokenIds = []

smith.noGrad(() => {
  for (let i = 0; i < evalSteps; i++) {
    const { input, target } = valLoader.next()
    const { logits } = forward(model, input)

    // Compute per-token loss manually for BPB
    const logitsData = logits.data.data
    const V = config.vocabSize
    for (let t = 0; t < SEQ_LEN; t++) {
      const offset = t * V
      let maxLogit = -Infinity
      for (let v = 0; v < V; v++) {
        if (logitsData[offset + v] > maxLogit) maxLogit = logitsData[offset + v]
      }
      let sumExp = 0
      for (let v = 0; v < V; v++) sumExp += Math.exp(logitsData[offset + v] - maxLogit)
      const logSumExp = Math.log(sumExp)
      const loss = -(logitsData[offset + target[t]] - maxLogit - logSumExp)
      tokenLosses.push(loss)
      tokenIds.push(target[t])
      valLossSum += loss
      valTokenCount++
    }
  }
})

const valLoss = valLossSum / valTokenCount
const valBPB = evaluateBPB(tokenLosses, tokenIds, tokenizer)

// --- Summary ---

console.log('\n---')
console.log(`val_loss:         ${valLoss.toFixed(4)}`)
console.log(`val_bpb:          ${valBPB.toFixed(4)}`)
console.log(`training_seconds: ${totalTime.toFixed(1)}`)
console.log(`total_steps:      ${step}`)
console.log(`total_tokens:     ${(step * TOTAL_BATCH_SIZE / 1e6).toFixed(1)}M`)
console.log(`num_params:       ${(nParams / 1e6).toFixed(2)}M`)
console.log(`depth:            ${DEPTH}`)
console.log(`dim:              ${nEmbd}`)
