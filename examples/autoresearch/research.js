#!/usr/bin/env bun
// examples/autoresearch/research.js
// Experiment runner for autoresearch. Handles training execution,
// output parsing, and result logging (JSON + markdown).
//
// Commands:
//   run --tag "description"   Run training and log results
//   last [--log]              Show the most recent experiment
//   status                    Show experiment history and stats
//   best                      Show the best experiment so far

import { parseArgs } from 'node:util'
import { existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { execSync, spawnSync } from 'node:child_process'

const EXAMPLE_DIR = dirname(import.meta.filename)
const RESULTS_DIR = join(EXAMPLE_DIR, 'results')
const EXPERIMENTS_FILE = join(RESULTS_DIR, 'experiments.json')
const LOG_FILE = join(RESULTS_DIR, 'research_log.md')
const LAST_LOG_FILE = join(RESULTS_DIR, 'last_run.log')

// --- Helpers ---

function ensureResultsDir() {
  if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true })
}

function loadExperiments() {
  if (!existsSync(EXPERIMENTS_FILE)) return []
  return JSON.parse(readFileSync(EXPERIMENTS_FILE, 'utf8'))
}

function saveExperiments(experiments) {
  writeFileSync(EXPERIMENTS_FILE, JSON.stringify(experiments, null, 2))
}

function getGitHash() {
  try {
    return execSync('git rev-parse --short=7 HEAD', { encoding: 'utf8' }).trim()
  } catch { return 'unknown' }
}

function getGitBranch() {
  try {
    return execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf8' }).trim()
  } catch { return 'unknown' }
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

function formatDuration(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const m = Math.floor(seconds / 60)
  const s = (seconds % 60).toFixed(0)
  return `${m}m${s}s`
}

function findBest(experiments) {
  let best = null
  for (const exp of experiments) {
    if (exp.status === 'crash') continue
    if (!best || exp.metrics.val_bpb < best.metrics.val_bpb) best = exp
  }
  return best
}

// --- Colors (ANSI) ---

const c = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
  white: '\x1b[37m',
}

// --- Commands ---

function runExperiment(tag) {
  ensureResultsDir()
  const experiments = loadExperiments()
  const commit = getGitHash()
  const branch = getGitBranch()
  const best = findBest(experiments)
  const experimentNum = experiments.length + 1

  console.log(`${c.cyan}${c.bold}═══ Experiment #${experimentNum} ═══${c.reset}`)
  console.log(`${c.dim}Branch: ${branch} | Commit: ${commit}${c.reset}`)
  console.log(`${c.dim}Tag: ${tag}${c.reset}`)
  if (best) {
    console.log(`${c.dim}Current best: ${best.metrics.val_bpb.toFixed(4)} bpb (${best.tag})${c.reset}`)
  }
  console.log()

  // Run training
  const trainScript = join(EXAMPLE_DIR, 'train.js')
  const startTime = Date.now()

  const result = spawnSync('bun', [trainScript], {
    encoding: 'utf8',
    stdio: ['pipe', 'pipe', 'pipe'],
    timeout: 600_000, // 10 min hard kill
  })

  const elapsed = (Date.now() - startTime) / 1000
  const output = (result.stdout || '') + '\n' + (result.stderr || '')

  // Save raw log
  writeFileSync(LAST_LOG_FILE, output)

  // Parse metrics
  const metrics = parseMetrics(output)
  const crashed = result.status !== 0 || !metrics.val_bpb

  const experiment = {
    id: experimentNum,
    tag,
    commit,
    branch,
    timestamp: new Date().toISOString(),
    elapsed,
    status: crashed ? 'crash' : 'pending', // set to keep/discard by comparison
    metrics,
    error: crashed ? (result.stderr || '').split('\n').slice(-5).join('\n').trim() : null,
  }

  if (!crashed) {
    if (!best || metrics.val_bpb < best.metrics.val_bpb) {
      experiment.status = 'keep'
      experiment.improvement = best ? best.metrics.val_bpb - metrics.val_bpb : 0
    } else {
      experiment.status = 'discard'
      experiment.improvement = best.metrics.val_bpb - metrics.val_bpb
    }
  }

  experiments.push(experiment)
  saveExperiments(experiments)
  appendMarkdownEntry(experiment, best)

  // Print result
  printExperiment(experiment, best)

  return experiment
}

function printExperiment(exp, previousBest) {
  console.log()
  if (exp.status === 'crash') {
    console.log(`${c.red}${c.bold}  CRASH${c.reset} ${c.dim}(${formatDuration(exp.elapsed)})${c.reset}`)
    if (exp.error) console.log(`${c.red}  ${exp.error.split('\n')[0]}${c.reset}`)
  } else {
    const bpb = exp.metrics.val_bpb.toFixed(4)
    const statusIcon = exp.status === 'keep' ? `${c.green}✓ KEEP` : `${c.red}✗ DISCARD`
    const delta = exp.improvement
      ? ` (${delta > 0 ? '-' : '+'}${Math.abs(exp.improvement).toFixed(4)})`
      : ''

    console.log(`  ${statusIcon}${c.reset}  val_bpb: ${c.bold}${bpb}${c.reset}${c.dim}${delta}${c.reset}`)

    if (exp.metrics.training_seconds) {
      console.log(`${c.dim}  steps: ${exp.metrics.total_steps || '?'} | params: ${exp.metrics.num_params || '?'} | time: ${formatDuration(exp.metrics.training_seconds)}${c.reset}`)
    }
  }
  console.log()
}

function showLast(showLog) {
  const experiments = loadExperiments()
  if (experiments.length === 0) {
    console.log('No experiments recorded yet.')
    return
  }
  const exp = experiments[experiments.length - 1]
  const best = findBest(experiments.slice(0, -1))

  console.log(`${c.cyan}${c.bold}═══ Last Experiment (#${exp.id}) ═══${c.reset}`)
  console.log(`${c.dim}Tag: ${exp.tag} | Commit: ${exp.commit} | ${exp.timestamp}${c.reset}`)
  printExperiment(exp, best)

  if (showLog && existsSync(LAST_LOG_FILE)) {
    console.log(`${c.dim}─── Training Output ───${c.reset}`)
    const log = readFileSync(LAST_LOG_FILE, 'utf8')
    // Show last 30 lines
    const lines = log.split('\n')
    const tail = lines.slice(-30).join('\n')
    console.log(tail)
  }
}

function showStatus() {
  const experiments = loadExperiments()
  if (experiments.length === 0) {
    console.log('No experiments recorded yet. Run: bun research.js run --tag "baseline"')
    return
  }

  const best = findBest(experiments)
  const keeps = experiments.filter(e => e.status === 'keep')
  const discards = experiments.filter(e => e.status === 'discard')
  const crashes = experiments.filter(e => e.status === 'crash')

  console.log(`${c.cyan}${c.bold}═══ Autoresearch Status ═══${c.reset}`)
  console.log()
  console.log(`  Total experiments: ${c.bold}${experiments.length}${c.reset}`)
  console.log(`  ${c.green}Kept: ${keeps.length}${c.reset}  ${c.red}Discarded: ${discards.length}${c.reset}  ${c.yellow}Crashed: ${crashes.length}${c.reset}`)
  console.log(`  Hit rate: ${((keeps.length / experiments.length) * 100).toFixed(0)}%`)
  console.log()

  if (best) {
    console.log(`${c.bold}  Best: ${best.metrics.val_bpb.toFixed(4)} bpb${c.reset} ${c.dim}(#${best.id}: ${best.tag})${c.reset}`)
  }

  // Show timeline of recent experiments
  const recent = experiments.slice(-15)
  console.log()
  console.log(`${c.dim}─── Recent Experiments ───${c.reset}`)
  console.log()

  for (const exp of recent) {
    const icon = exp.status === 'keep' ? `${c.green}✓` : exp.status === 'crash' ? `${c.yellow}!` : `${c.red}✗`
    const bpb = exp.status === 'crash' ? 'crash' : exp.metrics.val_bpb.toFixed(4)
    const delta = exp.improvement && exp.improvement !== 0
      ? ` ${c.dim}(${exp.improvement > 0 ? c.green + '-' : c.red + '+'}${Math.abs(exp.improvement).toFixed(4)}${c.reset}${c.dim})${c.reset}`
      : ''
    console.log(`  ${icon}${c.reset} #${String(exp.id).padStart(3)} ${bpb.padStart(8)}${delta}  ${c.dim}${exp.tag}${c.reset}`)
  }

  // Show improvement trend if enough data
  if (keeps.length >= 3) {
    console.log()
    console.log(`${c.dim}─── Improvement Trend ───${c.reset}`)
    const firstBest = keeps[0].metrics.val_bpb
    const currentBest = best.metrics.val_bpb
    const totalImprovement = firstBest - currentBest
    console.log(`  ${c.dim}Baseline: ${firstBest.toFixed(4)} → Current best: ${currentBest.toFixed(4)}${c.reset}`)
    console.log(`  ${c.green}Total improvement: -${totalImprovement.toFixed(4)} bpb${c.reset}`)
  }

  console.log()
}

function showBest() {
  const experiments = loadExperiments()
  const best = findBest(experiments)
  if (!best) {
    console.log('No successful experiments yet.')
    return
  }

  console.log(`${c.cyan}${c.bold}═══ Best Experiment ═══${c.reset}`)
  console.log()
  console.log(`  ${c.bold}val_bpb: ${best.metrics.val_bpb.toFixed(4)}${c.reset}`)
  console.log(`  ${c.dim}Experiment #${best.id}: ${best.tag}${c.reset}`)
  console.log(`  ${c.dim}Commit: ${best.commit} on ${best.branch}${c.reset}`)
  console.log(`  ${c.dim}Date: ${best.timestamp}${c.reset}`)
  console.log()

  if (best.metrics.num_params) console.log(`  Parameters: ${best.metrics.num_params}`)
  if (best.metrics.depth) console.log(`  Depth: ${best.metrics.depth}`)
  if (best.metrics.dim) console.log(`  Dim: ${best.metrics.dim}`)
  if (best.metrics.total_steps) console.log(`  Steps: ${best.metrics.total_steps}`)
  console.log()
}

// --- Markdown logging ---

function appendMarkdownEntry(experiment, previousBest) {
  ensureResultsDir()
  const isFirst = !existsSync(LOG_FILE)

  let md = ''
  if (isFirst) {
    md += '# Autoresearch Experiment Log\n\n'
    md += `Started: ${experiment.timestamp}\n`
    md += `Branch: ${experiment.branch}\n\n`
    md += '---\n\n'
  }

  md += `### #${experiment.id} — ${experiment.tag}\n\n`
  md += `- **Commit**: ${experiment.commit}\n`
  md += `- **Time**: ${experiment.timestamp}\n`

  if (experiment.status === 'crash') {
    md += `- **Status**: CRASH\n`
    if (experiment.error) md += `- **Error**: \`${experiment.error.split('\n')[0]}\`\n`
  } else {
    const statusEmoji = experiment.status === 'keep' ? 'KEEP' : 'DISCARD'
    md += `- **val_bpb**: ${experiment.metrics.val_bpb.toFixed(4)}\n`
    md += `- **Status**: ${statusEmoji}\n`

    if (experiment.improvement && experiment.improvement !== 0) {
      const dir = experiment.improvement > 0 ? 'better' : 'worse'
      md += `- **Delta**: ${experiment.improvement > 0 ? '-' : '+'}${Math.abs(experiment.improvement).toFixed(4)} (${dir})\n`
    }

    if (experiment.metrics.num_params) md += `- **Params**: ${experiment.metrics.num_params}\n`
    if (experiment.metrics.total_steps) md += `- **Steps**: ${experiment.metrics.total_steps}\n`
  }

  md += '\n'
  appendFileSync(LOG_FILE, md)
}

// --- CLI ---

const command = process.argv[2]

if (!command || command === '--help' || command === '-h') {
  console.log(`${c.cyan}${c.bold}autoresearch${c.reset} — experiment runner`)
  console.log()
  console.log('Commands:')
  console.log('  run --tag "desc"   Run training experiment and log results')
  console.log('  last [--log]       Show the most recent experiment result')
  console.log('  status             Show experiment history and statistics')
  console.log('  best               Show the best experiment so far')
  console.log()
  console.log('Usage:')
  console.log('  bun examples/autoresearch/research.js run --tag "baseline"')
  console.log('  bun examples/autoresearch/research.js status')
  process.exit(0)
}

if (command === 'run') {
  const { values } = parseArgs({
    args: process.argv.slice(3),
    options: { tag: { type: 'string', default: 'experiment' } },
  })
  runExperiment(values.tag)
} else if (command === 'last') {
  const showLog = process.argv.includes('--log')
  showLast(showLog)
} else if (command === 'status') {
  showStatus()
} else if (command === 'best') {
  showBest()
} else {
  console.error(`Unknown command: ${command}. Run with --help for usage.`)
  process.exit(1)
}
