// examples/whisper/mel.js
// Mel spectrogram extraction for Whisper.
// Pure JS FFT + STFT + mel filterbank. No dependencies.
//
// Whisper expects: 80 mel bins (128 for large-v3), 30s chunks at 16kHz,
// hop_length=160, win_length=400, n_fft=400.

// --- Radix-2 Cooley-Tukey FFT ---

function fft(reIn, imIn) {
  const n = reIn.length
  if (n === 1) return { re: new Float64Array([reIn[0]]), im: new Float64Array([imIn[0]]) }

  // Bit-reversal permutation
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  const bits = Math.log2(n)
  for (let i = 0; i < n; i++) {
    let rev = 0
    for (let b = 0; b < bits; b++) rev |= ((i >> b) & 1) << (bits - 1 - b)
    re[rev] = reIn[i]
    im[rev] = imIn[i]
  }

  // Iterative butterfly
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2
    const angle = -2 * Math.PI / size
    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < half; j++) {
        const theta = angle * j
        const wr = Math.cos(theta)
        const wi = Math.sin(theta)
        const a = i + j
        const b = i + j + half
        const tr = wr * re[b] - wi * im[b]
        const ti = wr * im[b] + wi * re[b]
        re[b] = re[a] - tr
        im[b] = im[a] - ti
        re[a] += tr
        im[a] += ti
      }
    }
  }
  return { re, im }
}

// Next power of 2
function nextPow2(n) {
  let p = 1
  while (p < n) p <<= 1
  return p
}

// --- Hann window ---

function hannWindow(len) {
  const w = new Float32Array(len)
  for (let i = 0; i < len; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / len))
  return w
}

// --- STFT ---

function stft(samples, nFft, hopLength, winLength) {
  const window = hannWindow(winLength)
  const padded = nFft > winLength ? nFft : winLength
  const fftSize = nextPow2(padded)
  const numFrames = Math.floor((samples.length - winLength) / hopLength) + 1
  const freqBins = fftSize / 2 + 1

  // Output: magnitude squared, shape [freqBins, numFrames]
  const mag2 = new Float64Array(freqBins * numFrames)

  const reIn = new Float64Array(fftSize)
  const imIn = new Float64Array(fftSize)

  for (let t = 0; t < numFrames; t++) {
    const start = t * hopLength
    reIn.fill(0)
    imIn.fill(0)
    for (let i = 0; i < winLength; i++) {
      reIn[i] = (samples[start + i] || 0) * window[i]
    }

    const { re, im } = fft(reIn, imIn)

    for (let f = 0; f < freqBins; f++) {
      mag2[f * numFrames + t] = re[f] * re[f] + im[f] * im[f]
    }
  }

  return { mag2, freqBins, numFrames }
}

// --- Mel filterbank ---

function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700) }
function melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1) }

function createMelFilterbank(sampleRate, nFft, nMels) {
  const freqBins = nFft / 2 + 1
  const melLow = hzToMel(0)
  const melHigh = hzToMel(sampleRate / 2)

  // nMels + 2 equally spaced mel points
  const melPoints = new Float64Array(nMels + 2)
  for (let i = 0; i < nMels + 2; i++) {
    melPoints[i] = melLow + (melHigh - melLow) * i / (nMels + 1)
  }

  // Convert to Hz then to FFT bin indices
  const binPoints = melPoints.map(m => {
    const hz = melToHz(m)
    return Math.floor((nFft + 1) * hz / sampleRate)
  })

  // Build filter matrix [nMels, freqBins]
  const filters = new Float64Array(nMels * freqBins)
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

// --- Whisper mel spectrogram ---
// Matches whisper.cpp's log_mel_spectrogram:
// 1. STFT with Hann window
// 2. Apply mel filterbank
// 3. Log scale (clamped)
// 4. Normalize to [-1, 1] range

function melSpectrogram(samples, opts = {}) {
  const nFft = opts.nFft || 400
  const hopLength = opts.hopLength || 160
  const winLength = opts.winLength || 400
  const nMels = opts.nMels || 80
  const sampleRate = opts.sampleRate || 16000

  // Pad to 30 seconds if needed (Whisper expects fixed 30s chunks)
  const targetLen = 30 * sampleRate // 480000 samples
  let padded = samples
  if (samples.length < targetLen) {
    padded = new Float32Array(targetLen)
    padded.set(samples)
    // Rest is zero-padded
  } else if (samples.length > targetLen) {
    padded = samples.subarray(0, targetLen)
  }

  const { mag2, freqBins, numFrames } = stft(padded, nFft, hopLength, winLength)
  const { filters, nMels: nm } = createMelFilterbank(sampleRate, nextPow2(nFft > winLength ? nFft : winLength), nMels)

  // Apply mel filterbank: output [nMels, numFrames]
  const mel = new Float32Array(nMels * numFrames)
  for (let m = 0; m < nMels; m++) {
    for (let t = 0; t < numFrames; t++) {
      let sum = 0
      for (let f = 0; f < freqBins; f++) {
        sum += filters[m * freqBins + f] * mag2[f * numFrames + t]
      }
      mel[m * numFrames + t] = sum
    }
  }

  // Whisper normalization (matches whisper.cpp):
  // 1. log10(max(mel, 1e-10))
  // 2. clamp to max - 8.0
  // 3. (mel + 4.0) / 4.0
  let maxVal = -Infinity
  for (let i = 0; i < mel.length; i++) {
    mel[i] = Math.log10(Math.max(mel[i], 1e-10))
    if (mel[i] > maxVal) maxVal = mel[i]
  }

  for (let i = 0; i < mel.length; i++) {
    mel[i] = Math.max(mel[i], maxVal - 8.0)
    mel[i] = (mel[i] + 4.0) / 4.0
  }

  return { mel, nMels, numFrames }
}

export { fft, stft, hannWindow, createMelFilterbank, melSpectrogram, nextPow2 }
