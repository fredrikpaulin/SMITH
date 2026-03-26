import { test, expect } from 'bun:test'
import { readWav, resample } from '../audio.js'

function buildWav(samples, sampleRate = 16000, bitsPerSample = 16, numChannels = 1) {
  const bytesPerSample = bitsPerSample / 8
  const dataSize = samples.length * bytesPerSample * numChannels
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  // RIFF header
  view.setUint32(0, 0x52494646, false) // "RIFF"
  view.setUint32(4, 36 + dataSize, true)
  view.setUint32(8, 0x57415645, false) // "WAVE"

  // fmt chunk
  view.setUint32(12, 0x666d7420, false) // "fmt "
  view.setUint32(16, 16, true) // chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true)
  view.setUint16(32, numChannels * bytesPerSample, true)
  view.setUint16(34, bitsPerSample, true)

  // data chunk
  view.setUint32(36, 0x64617461, false) // "data"
  view.setUint32(40, dataSize, true)

  for (let i = 0; i < samples.length; i++) {
    if (bitsPerSample === 16) {
      view.setInt16(44 + i * 2, Math.round(samples[i] * 32767), true)
    }
  }

  return buffer
}

test('readWav decodes 16-bit PCM mono', () => {
  const original = new Float32Array([0, 0.5, -0.5, 1.0, -1.0])
  const buffer = buildWav(original, 44100)
  const { samples, sampleRate, numChannels, numSamples } = readWav(buffer)

  expect(sampleRate).toBe(44100)
  expect(numChannels).toBe(1)
  expect(numSamples).toBe(5)
  // Check values are approximately correct (int16 quantization)
  expect(Math.abs(samples[0])).toBeLessThan(0.001)
  expect(Math.abs(samples[1] - 0.5)).toBeLessThan(0.001)
  expect(Math.abs(samples[2] + 0.5)).toBeLessThan(0.001)
})

test('readWav rejects non-WAV files', () => {
  const buffer = new ArrayBuffer(44)
  expect(() => readWav(buffer)).toThrow('Not a WAV file')
})

test('resample identity (same rate)', () => {
  const samples = new Float32Array([1, 2, 3, 4, 5])
  const result = resample(samples, 16000, 16000)
  expect(result.length).toBe(5)
  expect(result[0]).toBe(1)
  expect(result[4]).toBe(5)
})

test('resample downsamples 2:1', () => {
  const samples = new Float32Array([0, 1, 0, 1, 0, 1, 0, 1])
  const result = resample(samples, 16000, 8000)
  expect(result.length).toBe(4)
  // Linear interpolation at doubled step
  expect(result[0]).toBe(0)
  expect(result[1]).toBe(0)
})

test('resample upsamples 1:2', () => {
  const samples = new Float32Array([0, 1])
  const result = resample(samples, 8000, 16000)
  expect(result.length).toBe(4)
  expect(result[0]).toBe(0)
  expect(Math.abs(result[1] - 0.5)).toBeLessThan(0.01)
  expect(result[2]).toBe(1)
})
