// smith/bench/bench.js
// Benchmark suite: measures GPU kernel performance on typical workloads.
// Run: bun bench/bench.js

import smith from '../src/index.js'
import { matmulQ4, quantizeQ4 } from '../src/ops/quantize.js'

function bench(name, fn, warmup = 3, iters = 20) {
  for (let i = 0; i < warmup; i++) fn()
  const times = []
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now()
    fn()
    times.push(performance.now() - t0)
  }
  times.sort((a, b) => a - b)
  const median = times[Math.floor(times.length / 2)]
  const mean = times.reduce((a, b) => a + b, 0) / times.length
  const min = times[0]
  console.log(`  ${name}: median=${median.toFixed(2)}ms  mean=${mean.toFixed(2)}ms  min=${min.toFixed(2)}ms`)
}

console.log('\n=== Smith GPU Benchmarks ===\n')
console.log(`Device: ${smith.info().device}`)
console.log()

// --- Matmul ---
console.log('--- Matmul ---')
{
  const sizes = [[128, 128, 128], [256, 256, 256], [512, 512, 512]]
  for (const [M, N, K] of sizes) {
    const a = smith.rand([M, K])
    const b = smith.rand([K, N])
    bench(`matmul ${M}x${K} @ ${K}x${N}`, () => {
      const { matmul } = require('../src/ops/matmul.js')
      matmul(a, b)
    })
  }
}

// --- Softmax ---
console.log('\n--- Softmax ---')
{
  const { softmax } = require('../src/ops/softmax.js')
  for (const cols of [128, 512, 2048]) {
    const x = smith.rand([32, cols])
    bench(`softmax [32, ${cols}]`, () => softmax(x))
  }
}

// --- Layernorm ---
console.log('\n--- Layernorm ---')
{
  const { layernormForward } = require('../src/ops/layernorm.js')
  for (const dim of [64, 128, 256]) {
    const x = smith.rand([32, dim])
    const gamma = smith.ones([dim])
    const beta = smith.zeros([dim])
    bench(`layernorm [32, ${dim}]`, () => layernormForward(x, gamma, beta))
  }
}

// --- Elementwise ---
console.log('\n--- Elementwise ---')
{
  const { add } = require('../src/ops/add.js')
  for (const size of [1024, 65536, 262144]) {
    const a = smith.rand([size])
    const b = smith.rand([size])
    bench(`add [${size}]`, () => add(a, b))
  }
}

// --- Q4 Matmul ---
console.log('\n--- Q4 Matmul ---')
{
  for (const [M, K, N] of [[32, 256, 256], [32, 512, 512]]) {
    const a = smith.rand([M, K])
    const w = smith.rand([K, N])
    const wq = quantizeQ4(w)
    bench(`q4 matmul ${M}x${K} @ ${K}x${N}`, () => matmulQ4(a, wq))
  }
}

// --- GPT Forward (tiny) ---
console.log('\n--- GPT Forward ---')
{
  const model = smith.createModel({ vocabSize: 256, numLayers: 2, numHeads: 2, dim: 64, maxSeqLen: 128 })
  for (const seqLen of [8, 32, 64]) {
    const input = Array.from({ length: seqLen }, (_, i) => i % 256)
    bench(`GPT forward seqLen=${seqLen} (2L/2H/d64)`, () => {
      smith.noGrad(() => { smith.forward(model, input) })
    })
  }
}

console.log('\nDone.')
