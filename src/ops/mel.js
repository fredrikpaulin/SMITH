// smith/src/ops/mel.js
// GPU mel spectrogram pipeline.
// Orchestrates: window → batch FFT → magnitude → filterbank → log normalization.
// Produces output identical to CPU mel.js's melSpectrogram().

import * as T from '../tensor.js'
import { run } from '../dispatch.js'
import { gpuBatchFFT, nextPow2 } from './fft.js'

// --- Mel filterbank creation (CPU, same as examples/whisper/mel.js) ---

function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700) }
function melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1) }

function createMelFilterbank(sampleRate, nFft, nMels) {
  const freqBins = nFft / 2 + 1
  const melLow = hzToMel(0)
  const melHigh = hzToMel(sampleRate / 2)

  const melPoints = new Float64Array(nMels + 2)
  for (let i = 0; i < nMels + 2; i++) {
    melPoints[i] = melLow + (melHigh - melLow) * i / (nMels + 1)
  }

  const binPoints = melPoints.map(m => {
    const hz = melToHz(m)
    return Math.floor((nFft + 1) * hz / sampleRate)
  })

  const filters = new Float32Array(nMels * freqBins)
  for (let m = 0; m < nMels; m++) {
    const left = binPoints[m]
    const center = binPoints[m + 1]
    const right = binPoints[m + 2]

    for (let f = left; f <= center && f < freqBins; f++) {
      if (center !== left) filters[m * freqBins + f] = (f - left) / (center - left)
    }
    for (let f = center; f <= right && f < freqBins; f++) {
      if (right !== center) filters[m * freqBins + f] = (right - f) / (right - center)
    }
  }

  return { filters, nMels, freqBins }
}

// --- GPU mel spectrogram ---
// Matches whisper.cpp / CPU mel.js normalization exactly.

function gpuMelSpectrogram(samples, opts = {}) {
  const nFft = opts.nFft || 400
  const hopLength = opts.hopLength || 160
  const winLength = opts.winLength || 400
  const nMels = opts.nMels || 80
  const sampleRate = opts.sampleRate || 16000

  // Pad to 30 seconds if needed
  const targetLen = 30 * sampleRate
  let audioData = samples
  if (samples.length < targetLen) {
    audioData = new Float32Array(targetLen)
    audioData.set(samples)
  } else if (samples.length > targetLen) {
    audioData = samples.subarray(0, targetLen)
  }

  const numSamples = audioData.length
  const numFrames = Math.floor((numSamples - winLength) / hopLength) + 1
  const fftSize = nextPow2(nFft > winLength ? nFft : winLength)
  const freqBins = fftSize / 2 + 1

  // Upload audio samples to GPU
  const samplesTensor = T.create([numSamples], 'f32')
  samplesTensor.data.set(audioData)

  // Step 1: Apply Hann window to all frames on GPU
  const windowed = T.create([numFrames * fftSize * 2], 'f32')
  const stftParams = new Uint32Array(5)
  stftParams[0] = fftSize
  stftParams[1] = winLength
  stftParams[2] = hopLength
  stftParams[3] = numFrames
  stftParams[4] = numSamples

  run('stft_window', [
    { buffer: samplesTensor.buffer, index: 0 },
    { buffer: windowed.buffer, index: 1 },
  ], { x: fftSize, y: numFrames },
  { x: Math.min(fftSize, 256), y: 1 },
  { data: stftParams, index: 2 })

  // Step 2: Batch FFT (one FFT per frame)
  const fftOut = gpuBatchFFT(windowed, fftSize, numFrames, false)

  // Step 3: Compute magnitude squared [freqBins, numFrames]
  const mag2 = T.create([freqBins * numFrames], 'f32')
  const magParams = new Uint32Array(3)
  magParams[0] = fftSize
  magParams[1] = freqBins
  magParams[2] = numFrames

  run('stft_magnitude', [
    { buffer: fftOut.buffer, index: 0 },
    { buffer: mag2.buffer, index: 1 },
  ], { x: freqBins, y: numFrames },
  { x: Math.min(freqBins, 256), y: 1 },
  { data: magParams, index: 2 })

  // Step 4: Mel filterbank multiplication
  const { filters } = createMelFilterbank(sampleRate, fftSize, nMels)
  const filtersTensor = T.create([nMels * freqBins], 'f32')
  filtersTensor.data.set(filters)

  const melOut = T.create([nMels * numFrames], 'f32')
  const melParams = new Uint32Array(3)
  melParams[0] = nMels
  melParams[1] = freqBins
  melParams[2] = numFrames

  run('mel_filterbank', [
    { buffer: filtersTensor.buffer, index: 0 },
    { buffer: mag2.buffer, index: 1 },
    { buffer: melOut.buffer, index: 2 },
  ], { x: numFrames, y: nMels },
  { x: Math.min(numFrames, 256), y: 1 },
  { data: melParams, index: 3 })

  // Step 5: Log transform (GPU)
  const totalSize = nMels * numFrames
  const sizeParam = new Uint32Array([totalSize])

  run('mel_log', [
    { buffer: melOut.buffer, index: 0 },
  ], { x: totalSize }, { x: Math.min(totalSize, 256) },
  { data: sizeParam, index: 1 })

  // Step 6: Find max (CPU — small data, not worth a reduction shader)
  let maxVal = -Infinity
  for (let i = 0; i < totalSize; i++) {
    if (melOut.data[i] > maxVal) maxVal = melOut.data[i]
  }

  // Step 7: Clamp and normalize (GPU)
  const normBuf = new ArrayBuffer(8)
  const normF32 = new Float32Array(normBuf, 0, 1)
  const normU32 = new Uint32Array(normBuf, 4, 1)
  normF32[0] = maxVal
  normU32[0] = totalSize
  const normParams = new Uint8Array(normBuf)

  run('mel_normalize', [
    { buffer: melOut.buffer, index: 0 },
  ], { x: totalSize }, { x: Math.min(totalSize, 256) },
  { data: normParams, index: 1 })

  // Return as flat Float32Array (same format as CPU melSpectrogram)
  return { mel: melOut.data, nMels, numFrames }
}

export { gpuMelSpectrogram, createMelFilterbank, hzToMel, melToHz }
