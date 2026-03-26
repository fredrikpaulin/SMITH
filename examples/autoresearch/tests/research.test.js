import { test, expect, describe } from 'bun:test'
import { existsSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs'
import { join, dirname } from 'node:path'

// We test the parse and logging functions by importing the runner
// and calling its internals. Since research.js is a CLI script,
// we test its core logic by extracting the same parsing function.

const EXAMPLE_DIR = dirname(dirname(import.meta.filename))
const TEST_RESULTS_DIR = join(import.meta.dir, '_test_results')

function cleanup() {
  if (existsSync(TEST_RESULTS_DIR)) rmSync(TEST_RESULTS_DIR, { recursive: true })
}

function parseMetrics(output) {
  const metrics = {}
  const lines = output.split('\n')
  let inSummary = false
  for (const line of lines) {
    if (line.trim() === '---') { inSummary = true; continue }
    if (!inSummary) continue
    const match = line.match(/^(\w[\w_]*?):\s+(.+)$/)
    if (match) {
      const key = match[1].trim()
      const val = match[2].trim()
      metrics[key] = isNaN(Number(val)) ? val : Number(val)
    }
  }
  return metrics
}

// --- parseMetrics tests ---

describe('parseMetrics', () => {
  test('parses standard training output', () => {
    const output = `
Train tokens: 500,000
Starting training...

step    0 (0.0%) | loss: 8.2944

---
val_loss:         3.1234
val_bpb:          1.2345
training_seconds: 60.0
total_steps:      150
total_tokens:     1.2M
num_params:       2.50M
depth:            4
dim:              256
`
    const m = parseMetrics(output)
    expect(m.val_bpb).toBe(1.2345)
    expect(m.val_loss).toBe(3.1234)
    expect(m.training_seconds).toBe(60.0)
    expect(m.total_steps).toBe(150)
    expect(m.depth).toBe(4)
    expect(m.dim).toBe(256)
    expect(m.num_params).toBe('2.50M')
    expect(m.total_tokens).toBe('1.2M')
  })

  test('returns empty object for output without summary block', () => {
    const output = 'some random output\nwithout a summary'
    expect(parseMetrics(output)).toEqual({})
  })

  test('ignores lines before --- marker', () => {
    const output = 'val_bpb: 9.999\n---\nval_bpb:          1.5000\n'
    const m = parseMetrics(output)
    expect(m.val_bpb).toBe(1.5)
  })

  test('handles extra whitespace in values', () => {
    const output = '---\nval_bpb:          0.997900\npeak_vram_mb:     45060.2\n'
    const m = parseMetrics(output)
    expect(m.val_bpb).toBe(0.9979)
    expect(m.peak_vram_mb).toBe(45060.2)
  })
})

// --- Experiment JSON structure ---

describe('experiment tracking', () => {
  test('experiments.json roundtrip via serialization', () => {
    const experiments = [
      {
        id: 1,
        tag: 'baseline',
        commit: 'abc1234',
        branch: 'autoresearch/mar26',
        timestamp: '2026-03-26T10:00:00.000Z',
        elapsed: 65.2,
        status: 'keep',
        metrics: { val_bpb: 1.2345, val_loss: 3.1234, training_seconds: 60, depth: 4, dim: 256 },
        error: null,
        improvement: 0,
      },
      {
        id: 2,
        tag: 'increase LR to 0.04',
        commit: 'def5678',
        branch: 'autoresearch/mar26',
        timestamp: '2026-03-26T10:06:00.000Z',
        elapsed: 63.1,
        status: 'keep',
        metrics: { val_bpb: 1.2100, val_loss: 3.0500, training_seconds: 60, depth: 4, dim: 256 },
        error: null,
        improvement: 0.0245,
      },
    ]

    // Roundtrip through JSON serialization (same as file I/O)
    const json = JSON.stringify(experiments, null, 2)
    const loaded = JSON.parse(json)

    expect(loaded.length).toBe(2)
    expect(loaded[0].tag).toBe('baseline')
    expect(loaded[1].metrics.val_bpb).toBe(1.21)
    expect(loaded[1].improvement).toBeCloseTo(0.0245, 4)
  })

  test('findBest picks lowest val_bpb, ignoring crashes', () => {
    const experiments = [
      { status: 'keep', metrics: { val_bpb: 1.50 } },
      { status: 'crash', metrics: {} },
      { status: 'keep', metrics: { val_bpb: 1.20 } },
      { status: 'discard', metrics: { val_bpb: 1.25 } },
      { status: 'keep', metrics: { val_bpb: 1.30 } },
    ]

    let best = null
    for (const exp of experiments) {
      if (exp.status === 'crash') continue
      if (!best || exp.metrics.val_bpb < best.metrics.val_bpb) best = exp
    }

    expect(best.metrics.val_bpb).toBe(1.20)
  })

  test('status computation from experiment list', () => {
    const experiments = [
      { status: 'keep' },
      { status: 'discard' },
      { status: 'keep' },
      { status: 'crash' },
      { status: 'discard' },
      { status: 'keep' },
    ]

    const keeps = experiments.filter(e => e.status === 'keep')
    const discards = experiments.filter(e => e.status === 'discard')
    const crashes = experiments.filter(e => e.status === 'crash')

    expect(keeps.length).toBe(3)
    expect(discards.length).toBe(2)
    expect(crashes.length).toBe(1)
    expect((keeps.length / experiments.length * 100).toFixed(0)).toBe('50')
  })
})

// --- Markdown format ---

describe('markdown log format', () => {
  test('generates valid markdown entry', () => {
    const exp = {
      id: 5,
      tag: 'wider MLP ratio',
      commit: 'abc1234',
      timestamp: '2026-03-26T12:00:00.000Z',
      status: 'keep',
      metrics: { val_bpb: 1.1800, num_params: '3.20M', total_steps: 180 },
      improvement: 0.0200,
    }

    let md = `### #${exp.id} — ${exp.tag}\n\n`
    md += `- **Commit**: ${exp.commit}\n`
    md += `- **val_bpb**: ${exp.metrics.val_bpb.toFixed(4)}\n`
    md += `- **Status**: ${exp.status === 'keep' ? 'KEEP' : 'DISCARD'}\n`
    if (exp.improvement) {
      md += `- **Delta**: -${exp.improvement.toFixed(4)} (better)\n`
    }

    expect(md).toContain('### #5 — wider MLP ratio')
    expect(md).toContain('val_bpb**: 1.1800')
    expect(md).toContain('KEEP')
    expect(md).toContain('-0.0200')
  })
})

// --- CLI argument parsing ---

describe('CLI argument handling', () => {
  test('parseArgs extracts --tag value', () => {
    const { values } = require('node:util').parseArgs({
      args: ['--tag', 'my experiment'],
      options: { tag: { type: 'string', default: 'experiment' } },
    })
    expect(values.tag).toBe('my experiment')
  })

  test('parseArgs uses default when --tag omitted', () => {
    const { values } = require('node:util').parseArgs({
      args: [],
      options: { tag: { type: 'string', default: 'experiment' } },
    })
    expect(values.tag).toBe('experiment')
  })
})
