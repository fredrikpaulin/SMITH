// examples/tts/audio.js
// WAV file encoding and macOS audio playback via afplay.

import { spawn } from 'node:child_process'

// Encode PCM float32 samples to WAV file format (16-bit PCM)
function encodeWAV(samples, sampleRate = 24000) {
  const numSamples = samples.length
  const bitsPerSample = 16
  const numChannels = 1
  const byteRate = sampleRate * numChannels * bitsPerSample / 8
  const blockAlign = numChannels * bitsPerSample / 8
  const dataSize = numSamples * blockAlign
  const headerSize = 44
  const buffer = new ArrayBuffer(headerSize + dataSize)
  const view = new DataView(buffer)

  // RIFF header
  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(view, 8, 'WAVE')

  // fmt chunk
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)         // chunk size
  view.setUint16(20, 1, true)          // PCM format
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitsPerSample, true)

  // data chunk
  writeString(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  // Convert float32 [-1, 1] to int16
  const offset = headerSize
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
  }

  return new Uint8Array(buffer)
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i))
  }
}

// Write WAV to file and play with afplay (macOS)
async function playAudio(samples, sampleRate = 24000) {
  const wav = encodeWAV(samples, sampleRate)
  const tmpPath = `/tmp/smith_tts_${Date.now()}.wav`
  await Bun.write(tmpPath, wav)

  return new Promise((resolve, reject) => {
    const proc = spawn('afplay', [tmpPath])
    proc.on('close', (code) => {
      Bun.file(tmpPath).exists().then(() => {
        // cleanup silently
        try { require('fs').unlinkSync(tmpPath) } catch {}
      })
      code === 0 ? resolve() : reject(new Error(`afplay exited ${code}`))
    })
    proc.on('error', reject)
  })
}

// Write WAV to file
async function saveWAV(samples, path, sampleRate = 24000) {
  const wav = encodeWAV(samples, sampleRate)
  await Bun.write(path, wav)
}

export { encodeWAV, playAudio, saveWAV }
