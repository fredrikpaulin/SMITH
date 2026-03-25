// smith/src/profile.js
// Profiling and benchmarking for Metal compute kernels.
// Instruments dispatch.js to collect per-kernel GPU timing.
// Uses MTLCommandBuffer.GPUStartTime/GPUEndTime for accurate GPU measurement.

import * as device from './device.js'

// --- Profiler state ---

let _enabled = false
const _kernelStats = new Map()  // kernelName → { calls, totalGpuMs, minMs, maxMs }
let _totalDispatches = 0
let _totalGpuMs = 0
let _peakMemory = 0
let _startMemory = 0

// --- Enable/disable ---

function enableProfiling() {
  _enabled = true
  _startMemory = device.allocatedSize()
  _peakMemory = _startMemory
}

function disableProfiling() {
  _enabled = false
}

function isProfilingEnabled() {
  return _enabled
}

// --- Record a kernel dispatch ---

function recordKernel(kernelName, gpuMs) {
  _totalDispatches++
  _totalGpuMs += gpuMs

  let stats = _kernelStats.get(kernelName)
  if (!stats) {
    stats = { calls: 0, totalGpuMs: 0, minMs: Infinity, maxMs: 0, samples: [] }
    _kernelStats.set(kernelName, stats)
  }
  stats.calls++
  stats.totalGpuMs += gpuMs
  if (gpuMs < stats.minMs) stats.minMs = gpuMs
  if (gpuMs > stats.maxMs) stats.maxMs = gpuMs
  // Keep last 100 samples for percentile computation
  if (stats.samples.length < 100) stats.samples.push(gpuMs)
  else stats.samples[stats.calls % 100] = gpuMs

  // Track peak memory
  const mem = device.allocatedSize()
  if (mem > _peakMemory) _peakMemory = mem
}

// --- Report ---

function report() {
  const entries = []
  for (const [name, stats] of _kernelStats) {
    const avgMs = stats.calls > 0 ? stats.totalGpuMs / stats.calls : 0
    const pct = _totalGpuMs > 0 ? (stats.totalGpuMs / _totalGpuMs) * 100 : 0
    entries.push({
      kernel: name,
      calls: stats.calls,
      totalMs: round(stats.totalGpuMs),
      avgMs: round(avgMs),
      minMs: round(stats.minMs === Infinity ? 0 : stats.minMs),
      maxMs: round(stats.maxMs),
      pct: round(pct),
    })
  }

  // Sort by total time descending
  entries.sort((a, b) => b.totalMs - a.totalMs)

  return {
    dispatches: _totalDispatches,
    totalGpuMs: round(_totalGpuMs),
    memory: {
      startBytes: _startMemory,
      peakBytes: _peakMemory,
      currentBytes: device.allocatedSize(),
      deltaBytes: _peakMemory - _startMemory,
    },
    kernels: entries,
  }
}

// --- Reset ---

function resetProfile() {
  _kernelStats.clear()
  _totalDispatches = 0
  _totalGpuMs = 0
  _peakMemory = device.allocatedSize()
  _startMemory = _peakMemory
}

// --- profile(fn): wrap a function, return result + timing ---

function profile(fn) {
  const wasEnabled = _enabled
  enableProfiling()
  resetProfile()

  const cpuStart = performance.now()
  const result = fn()
  const cpuMs = performance.now() - cpuStart

  const rep = report()

  if (!wasEnabled) disableProfiling()

  return {
    result,
    cpuMs: round(cpuMs),
    gpuMs: rep.totalGpuMs,
    dispatches: rep.dispatches,
    memory: rep.memory,
    kernels: rep.kernels,
  }
}

// --- benchmark(name, fn, opts): run N times, report stats ---

function benchmark(name, fn, opts = {}) {
  const { warmup = 3, iterations = 10 } = opts

  // Warmup
  for (let i = 0; i < warmup; i++) fn()

  const cpuTimes = []
  const gpuTimes = []

  for (let i = 0; i < iterations; i++) {
    const p = profile(fn)
    cpuTimes.push(p.cpuMs)
    gpuTimes.push(p.gpuMs)
  }

  cpuTimes.sort((a, b) => a - b)
  gpuTimes.sort((a, b) => a - b)

  return {
    name,
    iterations,
    cpu: computeStats(cpuTimes),
    gpu: computeStats(gpuTimes),
  }
}

function computeStats(sorted) {
  const n = sorted.length
  if (n === 0) return { mean: 0, median: 0, min: 0, max: 0, p95: 0, stddev: 0 }

  const sum = sorted.reduce((a, b) => a + b, 0)
  const mean = sum / n
  const median = n % 2 === 0
    ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    : sorted[Math.floor(n / 2)]
  const p95Idx = Math.min(Math.ceil(n * 0.95) - 1, n - 1)
  const variance = sorted.reduce((a, v) => a + (v - mean) ** 2, 0) / n
  const stddev = Math.sqrt(variance)

  return {
    mean: round(mean),
    median: round(median),
    min: round(sorted[0]),
    max: round(sorted[n - 1]),
    p95: round(sorted[p95Idx]),
    stddev: round(stddev),
  }
}

// --- Timed dispatch (replaces endSync when profiling) ---
// This is the instrumented version of dispatch.js's run()

function runTimed(kernel, buffers, grid, group, params) {
  const pso = device.pipeline(kernel)
  const enc = device.begin()

  device.setPipeline(enc, pso)

  for (const b of buffers) {
    device.setBuffer(enc, b.buffer, b.index)
  }

  if (params) {
    const data = params.data
    const byteLength = data.byteLength || data.length
    device.setBytes(enc, data, byteLength, params.index)
  }

  const GROUP_1D = 256
  const gx = grid.x || 1
  const gy = grid.y || 1
  const gz = grid.z || 1
  const grpX = group?.x || Math.min(gx, GROUP_1D)
  const grpY = group?.y || 1
  const grpZ = group?.z || 1

  device.dispatch(enc, gx, gy, gz, grpX, grpY, grpZ)

  const timing = device.endTimed(enc)
  recordKernel(kernel, timing.gpuMs)

  return timing
}

// --- Memory snapshot ---

function memorySnapshot() {
  return {
    allocatedBytes: device.allocatedSize(),
    pool: {
      // Import pool stats if available
    },
  }
}

function round(v) {
  return Math.round(v * 1000) / 1000
}

export {
  enableProfiling, disableProfiling, isProfilingEnabled,
  recordKernel,
  report, resetProfile,
  profile, benchmark,
  runTimed,
  memorySnapshot,
}
