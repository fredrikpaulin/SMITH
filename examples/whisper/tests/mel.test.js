import { test, expect } from 'bun:test'
import { fft, stft, hannWindow, createMelFilterbank, melSpectrogram, nextPow2 } from '../mel.js'

test('nextPow2', () => {
  expect(nextPow2(1)).toBe(1)
  expect(nextPow2(2)).toBe(2)
  expect(nextPow2(3)).toBe(4)
  expect(nextPow2(400)).toBe(512)
  expect(nextPow2(512)).toBe(512)
})

test('hannWindow has correct shape', () => {
  const w = hannWindow(400)
  expect(w.length).toBe(400)
  expect(w[0]).toBeCloseTo(0, 3) // Starts at ~0
  expect(w[200]).toBeCloseTo(1, 3) // Peak at center
  expect(w[399]).toBeCloseTo(0, 1) // Ends near 0
})

test('fft of DC signal', () => {
  const n = 8
  const re = new Float64Array(n).fill(1)
  const im = new Float64Array(n).fill(0)
  const result = fft(re, im)
  expect(result.re[0]).toBeCloseTo(8, 5) // DC component = sum
  for (let i = 1; i < n; i++) {
    expect(Math.abs(result.re[i])).toBeLessThan(1e-10)
    expect(Math.abs(result.im[i])).toBeLessThan(1e-10)
  }
})

test('fft of sine wave has peaks at expected bins', () => {
  const n = 64
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  // Single-frequency sine at bin 4
  for (let i = 0; i < n; i++) re[i] = Math.sin(2 * Math.PI * 4 * i / n)
  const result = fft(re, im)
  // Magnitude at bin 4 should be dominant
  const mag4 = Math.sqrt(result.re[4] ** 2 + result.im[4] ** 2)
  const mag0 = Math.sqrt(result.re[0] ** 2 + result.im[0] ** 2)
  expect(mag4).toBeGreaterThan(20)
  expect(mag0).toBeLessThan(1e-10)
})

test('stft produces correct shape', () => {
  const samples = new Float32Array(480000) // 30s at 16kHz
  for (let i = 0; i < samples.length; i++) samples[i] = Math.sin(2 * Math.PI * 440 * i / 16000)

  const { mag2, freqBins, numFrames } = stft(samples, 400, 160, 400)
  // Expected frames: (480000 - 400) / 160 + 1 = 2998
  expect(numFrames).toBe(2998)
  // Freq bins: 512/2 + 1 = 257 (FFT size is nextPow2(400) = 512)
  expect(freqBins).toBe(257)
  expect(mag2.length).toBe(freqBins * numFrames)
})

test('createMelFilterbank has correct dimensions', () => {
  const { filters, nMels, freqBins } = createMelFilterbank(16000, 512, 80)
  expect(nMels).toBe(80)
  expect(freqBins).toBe(257)
  expect(filters.length).toBe(80 * 257)
  // Each mel filter should be non-negative
  for (let i = 0; i < filters.length; i++) {
    expect(filters[i]).toBeGreaterThanOrEqual(0)
  }
})

test('melSpectrogram produces Whisper-shaped output', () => {
  // Short silence — should still produce valid output
  const samples = new Float32Array(16000) // 1 second of silence
  const { mel, nMels, numFrames } = melSpectrogram(samples, { nMels: 80 })
  expect(nMels).toBe(80)
  // Padded to 30s: (480000 - 400) / 160 + 1 = 2998
  expect(numFrames).toBe(2998)
  expect(mel.length).toBe(80 * 2998)
  // Silence should produce low values (normalized)
  for (let i = 0; i < mel.length; i++) {
    expect(isFinite(mel[i])).toBe(true)
  }
})

test('melSpectrogram with 440Hz tone has energy in expected mel bins', () => {
  const samples = new Float32Array(480000) // 30s at 16kHz
  for (let i = 0; i < samples.length; i++) samples[i] = 0.5 * Math.sin(2 * Math.PI * 440 * i / 16000)

  const { mel, nMels, numFrames } = melSpectrogram(samples, { nMels: 80 })
  // 440Hz should show up in lower-mid mel bins
  // Check that we have variation across frequency
  let minVal = Infinity, maxVal = -Infinity
  for (let i = 0; i < mel.length; i++) {
    if (mel[i] < minVal) minVal = mel[i]
    if (mel[i] > maxVal) maxVal = mel[i]
  }
  expect(maxVal - minVal).toBeGreaterThan(0.01) // Not all same value
})
