// examples/whisper/audio.js
// Pure JS audio decoding — WAV files only, no dependencies.
// Decodes PCM (int16/int32/float32) and resamples to 16kHz mono as Whisper expects.

function readWav(buffer) {
  const view = new DataView(buffer instanceof ArrayBuffer ? buffer : buffer.buffer)
  if (view.getUint32(0, false) !== 0x52494646) throw new Error('Not a WAV file (missing RIFF header)')
  if (view.getUint32(8, false) !== 0x57415645) throw new Error('Not a WAV file (missing WAVE marker)')

  let offset = 12
  let fmt = null
  let dataChunk = null

  while (offset < view.byteLength - 8) {
    const id = String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3))
    const size = view.getUint32(offset + 4, true)
    if (id === 'fmt ') {
      fmt = {
        audioFormat: view.getUint16(offset + 8, true),
        numChannels: view.getUint16(offset + 10, true),
        sampleRate: view.getUint32(offset + 12, true),
        bitsPerSample: view.getUint16(offset + 22, true),
      }
    } else if (id === 'data') {
      dataChunk = { offset: offset + 8, size }
    }
    offset += 8 + size
    if (size % 2 !== 0) offset++ // WAV chunks are word-aligned
  }

  if (!fmt) throw new Error('WAV: missing fmt chunk')
  if (!dataChunk) throw new Error('WAV: missing data chunk')
  if (fmt.audioFormat !== 1 && fmt.audioFormat !== 3) {
    throw new Error(`WAV: unsupported format ${fmt.audioFormat} (only PCM=1 and IEEE float=3)`)
  }

  const { numChannels, sampleRate, bitsPerSample, audioFormat } = fmt
  const bytesPerSample = bitsPerSample / 8
  const numSamples = dataChunk.size / (bytesPerSample * numChannels)

  // Decode to float32 mono
  const samples = new Float32Array(numSamples)
  let pos = dataChunk.offset

  for (let i = 0; i < numSamples; i++) {
    let sum = 0
    for (let ch = 0; ch < numChannels; ch++) {
      if (audioFormat === 3) {
        // IEEE float
        sum += bitsPerSample === 32 ? view.getFloat32(pos, true) : view.getFloat64(pos, true)
      } else if (bitsPerSample === 16) {
        sum += view.getInt16(pos, true) / 32768
      } else if (bitsPerSample === 32) {
        sum += view.getInt32(pos, true) / 2147483648
      } else if (bitsPerSample === 8) {
        sum += (view.getUint8(pos) - 128) / 128
      }
      pos += bytesPerSample
    }
    samples[i] = sum / numChannels
  }

  return { samples, sampleRate, numChannels, numSamples }
}

// Linear interpolation resampler
function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples
  const ratio = fromRate / toRate
  const outLen = Math.ceil(samples.length / ratio)
  const out = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const srcPos = i * ratio
    const idx = Math.floor(srcPos)
    const frac = srcPos - idx
    const a = samples[idx] || 0
    const b = samples[Math.min(idx + 1, samples.length - 1)] || 0
    out[i] = a + frac * (b - a)
  }
  return out
}

// Load a WAV file and return 16kHz mono float32 samples
async function loadAudio(path) {
  const file = Bun.file(path)
  const buffer = await file.arrayBuffer()
  const { samples, sampleRate } = readWav(buffer)
  const resampled = resample(samples, sampleRate, 16000)
  return { samples: resampled, sampleRate: 16000 }
}

export { readWav, resample, loadAudio }
