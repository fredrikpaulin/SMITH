// tests/muon_trace.js — Calls actual stepMuon to find where NaN appears
// Run: MUON_DEBUG=1 bun tests/muon_trace.js
import smith from '../src/index.js'

const { tensor, variable } = smith

const N = 16

// Same deterministic values as debug script
const gradVals = Array.from({ length: N * N }, (_, i) => 0.3 * Math.sin(i * 0.7 + 0.3))
const paramVals = Array.from({ length: N * N }, (_, i) => {
  const r = Math.floor(i / N), c = i % N
  return r === c ? 1.0 : 0.02 * Math.sin(i * 1.3)
})

const show = (label, t) => {
  const d = Array.from(t.data)
  const maxAbs = Math.max(...d.map(Math.abs))
  const hasNaN = d.some(v => !isFinite(v))
  console.log(`${label}: max=${maxAbs.toFixed(6)} nan=${hasNaN} [${d.slice(0, 4).map(v => v.toFixed(4)).join(', ')}...]`)
}

const w = variable(tensor(paramVals, [N, N]), { requiresGrad: true })
const opt = smith.createMuonAdamW([{
  kind: 'muon',
  params: [w],
  lr: 0.01,
  momentum: 0.95,
  beta2: 0.7,
  weightDecay: 0.0,
  nsSteps: 5,
}])

w.grad = tensor(gradVals, [N, N])

show('Before: param', w.data)
show('Before: grad', w.grad)

smith.muonAdamWStep(opt)

show('After: param', w.data)
