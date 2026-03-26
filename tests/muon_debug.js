// tests/muon_debug.js — Run with: bun tests/muon_debug.js
// Replicates the exact stepMuon flow with a 16x16 matrix to verify all steps
import smith from '../src/index.js'
import * as T from '../src/tensor.js'
import { run, GROUP_1D } from '../src/dispatch.js'
import { matmul2d } from '../src/ops/matmul.js'
import { transpose } from '../src/ops/transpose.js'
import { scale as gpuScale } from '../src/ops/mul.js'

const { tensor, variable } = smith

const N = 16

const show = (label, t) => {
  const d = t.data
  if (!d) { console.log(`  ${label}: NO DATA`); return }
  const vals = Array.from(d)
  const hasNaN = vals.some(v => !isFinite(v))
  const maxAbs = Math.max(...vals.map(Math.abs))
  console.log(`  ${label}: max=${maxAbs.toFixed(6)} [${vals.slice(0, 4).map(v => v.toFixed(4)).join(', ')}...] ${hasNaN ? '❌ NaN/Inf' : '✓'}`)
}

const POLAR_COEFFS = [
  [8.156554524902461, -22.48329292557795, 15.878769915207462],
  [4.042929935166739, -2.808917465908714, 0.5000178451051316],
  [3.8916678022926607, -2.772484153217685, 0.5060648178503393],
  [3.285753657755655, -2.3681294933425376, 0.46449024233003106],
  [2.3465413258596377, -1.7097828382687081, 0.42323551169305323],
]

console.log(`=== Replicate stepMuon flow (${N}x${N} matrix) ===\n`)

// Generate deterministic test data
const gradVals = Array.from({ length: N * N }, (_, i) => 0.3 * Math.sin(i * 0.7 + 0.3))
const paramVals = Array.from({ length: N * N }, (_, i) => {
  const r = Math.floor(i / N), c = i % N
  return r === c ? 1.0 : 0.02 * Math.sin(i * 1.3)
})

const pGrad = tensor(gradVals, [N, N])
const pData = tensor(paramVals, [N, N])
const n = N * N

// 1. Copy grad
const g = T.create([N, N])
run('elementwise_scale', [
  { buffer: pGrad.buffer, index: 0 },
  { buffer: g.buffer, index: 1 },
], { x: n }, null, { data: new Float32Array([1.0]), index: 2 })
show('1. g (grad copy)', g)

// 2. Nesterov
const momBuf = T.zeros([N, N])
const nb = new ArrayBuffer(8)
new Float32Array(nb, 0, 1)[0] = 0.95
new Uint32Array(nb, 4, 1)[0] = n
run('muon_nesterov', [
  { buffer: g.buffer, index: 0 },
  { buffer: momBuf.buffer, index: 1 },
], { x: n }, { x: Math.min(n, GROUP_1D) }, { data: new Uint8Array(nb), index: 2 })
show('2. g (after nesterov)', g)

// 3. newtonSchulz — using WIDE path for square matrix (matches reference)
let normSum = 0
for (let i = 0; i < g.data.length; i++) normSum += g.data[i] * g.data[i]
const norm = Math.sqrt(normSum)
console.log(`  3. frobenius norm = ${norm.toFixed(6)}`)

let X = gpuScale(g, 1.0 / (norm * 1.02 + 1e-6))
show('4. X (normalized)', X)

// 5 NS iterations — WIDE path: A = X @ X^T, product = B @ X
for (let iter = 0; iter < 5; iter++) {
  const [a, b, c] = POLAR_COEFFS[iter]
  console.log(`\n  --- NS iter ${iter} (a=${a.toFixed(3)}, b=${b.toFixed(3)}, c=${c.toFixed(3)}) ---`)

  // A = X @ X^T (wide path for square)
  const A = matmul2d(X, transpose(X))
  show(`  A (X@X^T)`, A)

  // AA = A @ A
  const AA = matmul2d(A, A)
  show(`  AA`, AA)

  // B = b*A + c*AA
  const B = T.create([N, N])
  const pb = new ArrayBuffer(12)
  new Float32Array(pb, 0, 2).set([b, c])
  new Uint32Array(pb, 8, 1)[0] = n
  run('muon_ns_poly', [
    { buffer: A.buffer, index: 0 },
    { buffer: AA.buffer, index: 1 },
    { buffer: B.buffer, index: 2 },
  ], { x: n }, { x: Math.min(n, GROUP_1D) }, { data: new Uint8Array(pb), index: 3 })
  show(`  B`, B)

  // product = B @ X (wide path)
  const product = matmul2d(B, X)
  show(`  product (B@X)`, product)

  // X = a*X + product
  const cb = new ArrayBuffer(8)
  new Float32Array(cb, 0, 1)[0] = a
  new Uint32Array(cb, 4, 1)[0] = n
  run('muon_ns_combine', [
    { buffer: X.buffer, index: 0 },
    { buffer: product.buffer, index: 1 },
  ], { x: n }, { x: Math.min(n, GROUP_1D) }, { data: new Uint8Array(cb), index: 2 })
  show(`  X' (updated)`, X)
}

// Copy X back to g
run('elementwise_scale', [
  { buffer: X.buffer, index: 0 },
  { buffer: g.buffer, index: 1 },
], { x: n }, null, { data: new Float32Array([1.0]), index: 2 })
show('\n5. g (after NS)', g)

// NorMuon
const d = g.data
const rows = N, cols = N
const tall = rows >= cols  // for NorMuon, >= matches reference
const redDim = tall ? cols : rows
const otherDim = tall ? rows : cols
const vMean = new Float32Array(otherDim)

if (tall) {
  for (let r = 0; r < rows; r++) {
    let sum = 0
    const base = r * cols
    for (let c = 0; c < cols; c++) { const v = d[base + c]; sum += v * v }
    vMean[r] = sum / cols
  }
} else {
  for (let c = 0; c < cols; c++) {
    let sum = 0
    for (let r = 0; r < rows; r++) { const v = d[r * cols + c]; sum += v * v }
    vMean[c] = sum / rows
  }
}
console.log(`  6. vMean[0..3] = [${Array.from(vMean).slice(0, 4).map(v => v.toFixed(6)).join(', ')}]`)

let vNormSq = 0
for (let i = 0; i < otherDim; i++) vNormSq += vMean[i] * redDim
console.log(`  6. vNorm = ${Math.sqrt(vNormSq).toFixed(6)}`)

const smBuf = new Float32Array(otherDim)
for (let i = 0; i < otherDim; i++) smBuf[i] = 0.7 * 0 + 0.3 * vMean[i]
const stepSize = new Float32Array(otherDim)
for (let i = 0; i < otherDim; i++) stepSize[i] = 1.0 / Math.sqrt(Math.max(smBuf[i], 1e-10))
console.log(`  6. stepSize[0..3] = [${Array.from(stepSize).slice(0, 4).map(v => v.toFixed(4)).join(', ')}]`)

let vNormNewSq = 0
for (let i = 0; i < otherDim; i++) vNormNewSq += vMean[i] * redDim * stepSize[i] * stepSize[i]
const vNormNew = Math.sqrt(Math.max(vNormNewSq, 1e-10))
const globalScale = Math.sqrt(vNormSq) / vNormNew
console.log(`  6. globalScale = ${globalScale.toFixed(6)}`)

for (let r = 0; r < rows; r++) {
  const s = stepSize[r] * globalScale
  const base = r * cols
  for (let c = 0; c < cols; c++) d[base + c] *= s
}
show('7. g (after NorMuon)', g)

// muonUpdate
const scaledLr = 0.01 * Math.sqrt(Math.max(1.0, rows / cols))
const ub = new ArrayBuffer(12)
new Float32Array(ub, 0, 2).set([scaledLr, 0.0])
new Uint32Array(ub, 8, 1)[0] = n
run('muon_update', [
  { buffer: pData.buffer, index: 0 },
  { buffer: g.buffer, index: 1 },
], { x: n }, { x: Math.min(n, GROUP_1D) }, { data: new Uint8Array(ub), index: 2 })
show('8. params (after update)', pData)

console.log('\n=== Done ===')
