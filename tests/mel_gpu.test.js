import { test, expect } from 'bun:test'
import smith from '../src/index.js'
import { melSpectrogram as cpuMelSpectrogram } from '../examples/whisper/mel.js'

const { gpuMelSpectrogram } = smith

// --- GPU mel spectrogram equivalence ---

test('GPU mel spectrogram matches CPU — short signal', () => {
  const sampleRate = 16000
  const duration = 1 // 1 second
  const numSamples = sampleRate * duration
  const freq = 440 // A4

  const samples = new Float32Array(numSamples)
  for (let i = 0; i < numSamples; i++) {
    samples[i] = 0.5 * Math.sin(2 * Math.PI * freq * i / sampleRate)
  }

  const opts = { nFft: 400, hopLength: 160, winLength: 400, nMels: 80, sampleRate }

  const cpuResult = cpuMelSpectrogram(samples, opts)
  const gpuResult = gpuMelSpectrogram(samples, opts)

  // Both pad to 30 seconds internally
  expect(gpuResult.nMels).toBe(cpuResult.nMels)
  expect(gpuResult.numFrames).toBe(cpuResult.numFrames)

  // Compare mel values — allow tolerance for f32 vs f64 differences
  const totalSize = cpuResult.nMels * cpuResult.numFrames
  let maxDiff = 0
  for (let i = 0; i < totalSize; i++) {
    const diff = Math.abs(gpuResult.mel[i] - cpuResult.mel[i])
    if (diff > maxDiff) maxDiff = diff
  }

  // CPU uses f64 for FFT, GPU uses f32 — expect some tolerance
  // After log normalization, values are in [-1, 1] range, so 0.05 tolerance is reasonable
  expect(maxDiff).toBeLessThan(0.05)
})

test('GPU mel spectrogram matches CPU — silence', () => {
  const sampleRate = 16000
  const samples = new Float32Array(16000) // 1 second of silence

  const opts = { nFft: 400, hopLength: 160, winLength: 400, nMels: 80, sampleRate }

  const cpuResult = cpuMelSpectrogram(samples, opts)
  const gpuResult = gpuMelSpectrogram(samples, opts)

  expect(gpuResult.nMels).toBe(cpuResult.nMels)
  expect(gpuResult.numFrames).toBe(cpuResult.numFrames)

  // For silence, all mel values should be the same (the floor value after normalization)
  const totalSize = cpuResult.nMels * cpuResult.numFrames
  for (let i = 0; i < totalSize; i++) {
    expect(gpuResult.mel[i]).toBeCloseTo(cpuResult.mel[i], 2)
  }
})

test('GPU mel spectrogram matches CPU — white noise', () => {
  const sampleRate = 16000
  const numSamples = sampleRate * 2 // 2 seconds
  const samples = new Float32Array(numSamples)
  // Use seeded-style deterministic noise
  for (let i = 0; i < numSamples; i++) {
    samples[i] = Math.sin(i * 0.1) * Math.cos(i * 0.037) * 0.3
  }

  const opts = { nFft: 400, hopLength: 160, winLength: 400, nMels: 80, sampleRate }

  const cpuResult = cpuMelSpectrogram(samples, opts)
  const gpuResult = gpuMelSpectrogram(samples, opts)

  expect(gpuResult.nMels).toBe(cpuResult.nMels)
  expect(gpuResult.numFrames).toBe(cpuResult.numFrames)

  let maxDiff = 0
  const totalSize = cpuResult.nMels * cpuResult.numFrames
  for (let i = 0; i < totalSize; i++) {
    const diff = Math.abs(gpuResult.mel[i] - cpuResult.mel[i])
    if (diff > maxDiff) maxDiff = diff
  }

  expect(maxDiff).toBeLessThan(0.05)
})

test('GPU mel spectrogram — correct output shape', () => {
  const sampleRate = 16000
  const samples = new Float32Array(16000) // 1 second

  const result = gpuMelSpectrogram(samples, {
    nFft: 400, hopLength: 160, winLength: 400, nMels: 80, sampleRate,
  })

  // Padded to 30s = 480000 samples, numFrames = floor((480000 - 400) / 160) + 1 = 2997 + 1 = 2998
  expect(result.nMels).toBe(80)
  expect(result.numFrames).toBe(2998)
  expect(result.mel.length).toBe(80 * 2998)
})

test('GPU mel spectrogram — 128 mel bins (large-v3)', () => {
  const sampleRate = 16000
  const samples = new Float32Array(16000)
  for (let i = 0; i < 16000; i++) samples[i] = Math.sin(2 * Math.PI * 1000 * i / sampleRate) * 0.5

  const opts = { nFft: 400, hopLength: 160, winLength: 400, nMels: 128, sampleRate }

  const cpuResult = cpuMelSpectrogram(samples, opts)
  const gpuResult = gpuMelSpectrogram(samples, opts)

  expect(gpuResult.nMels).toBe(128)
  expect(gpuResult.numFrames).toBe(cpuResult.numFrames)

  let maxDiff = 0
  const totalSize = cpuResult.nMels * cpuResult.numFrames
  for (let i = 0; i < totalSize; i++) {
    const diff = Math.abs(gpuResult.mel[i] - cpuResult.mel[i])
    if (diff > maxDiff) maxDiff = diff
  }
  expect(maxDiff).toBeLessThan(0.05)
})

test('GPU mel spectrogram — values in expected range', () => {
  const sampleRate = 16000
  const samples = new Float32Array(16000)
  for (let i = 0; i < 16000; i++) samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.5

  const result = gpuMelSpectrogram(samples, {
    nFft: 400, hopLength: 160, winLength: 400, nMels: 80, sampleRate,
  })

  // After Whisper normalization, values should be roughly in [-1, 1] range
  let min = Infinity, max = -Infinity
  for (let i = 0; i < result.mel.length; i++) {
    if (result.mel[i] < min) min = result.mel[i]
    if (result.mel[i] > max) max = result.mel[i]
  }
  // Whisper norm: (log10(mel) + 4) / 4
  // For silence: log10(1e-10) + 4 = -10 + 4 = -6, / 4 = -1.5 (but clamped by maxVal - 8)
  // For signal: values should be > -1.5 and < 1.5
  expect(min).toBeGreaterThan(-2)
  expect(max).toBeLessThan(2)
})
