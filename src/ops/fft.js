// smith/src/ops/fft.js
// GPU dispatch for radix-2 FFT.
// Handles zero-padding to power of 2, interleaved complex layout,
// and forward/inverse transforms.

import * as T from '../tensor.js'
import { run } from '../dispatch.js'

function nextPow2(n) {
  let p = 1
  while (p < n) p <<= 1
  return p
}

function log2int(n) {
  let l = 0
  while ((1 << l) < n) l++
  return l
}

// FFT params: n, logN, batch, inverse
function fftParams(n, batch, inverse) {
  const params = new Uint32Array(4)
  params[0] = n
  params[1] = log2int(n)
  params[2] = batch
  params[3] = inverse ? 1 : 0
  return params
}

// Dispatch FFT kernel. All n threads must be in one threadgroup (shared memory).
// Grid: one threadgroup per batch element, each threadgroup has n threads.
// Max n = 1024 (Metal threadgroup size limit on Apple Silicon).
// For Whisper's n_fft=400 → padded to 512, well within limit.
function dispatchFFT(complexIn, complexOut, fftN, batch, inverse) {
  if (fftN > 1024) throw new Error(`GPU FFT max size is 1024 (got ${fftN}). Use multi-pass for larger.`)
  const params = fftParams(fftN, batch, inverse)
  run('fft_radix2', [
    { buffer: complexIn.buffer, index: 0 },
    { buffer: complexOut.buffer, index: 1 },
  ], { x: fftN, y: batch },
  { x: fftN, y: 1 },
  { data: params, index: 2 })
}

// GPU forward FFT on real input.
// input: tensor of shape [n] (real values)
// Returns { re, im } tensors of shape [n] (full complex output)
function gpuFFT(input, n) {
  const fftN = n || nextPow2(input.shape[0])
  const inputLen = input.shape[0]

  // Create interleaved complex input: [re0, im0, re1, im1, ...]
  const complexIn = T.create([fftN * 2], input.dtype)
  for (let i = 0; i < fftN; i++) {
    complexIn.data[i * 2] = i < inputLen ? input.data[i] : 0 // real
    complexIn.data[i * 2 + 1] = 0 // imaginary
  }

  const complexOut = T.create([fftN * 2], input.dtype)
  dispatchFFT(complexIn, complexOut, fftN, 1, false)

  // Deinterleave into separate re/im tensors
  const re = T.create([fftN], input.dtype)
  const im = T.create([fftN], input.dtype)
  for (let i = 0; i < fftN; i++) {
    re.data[i] = complexOut.data[i * 2]
    im.data[i] = complexOut.data[i * 2 + 1]
  }

  return { re, im }
}

// GPU inverse FFT from complex input.
// re, im: tensors of shape [n]
// Returns real tensor of shape [n]
function gpuIFFT(re, im, n) {
  const fftN = n || re.shape[0]

  // Create interleaved complex input
  const complexIn = T.create([fftN * 2], re.dtype)
  for (let i = 0; i < fftN; i++) {
    complexIn.data[i * 2] = re.data[i]
    complexIn.data[i * 2 + 1] = im.data[i]
  }

  const complexOut = T.create([fftN * 2], re.dtype)
  dispatchFFT(complexIn, complexOut, fftN, 1, true)

  // Extract real part
  const out = T.create([fftN], re.dtype)
  for (let i = 0; i < fftN; i++) {
    out.data[i] = complexOut.data[i * 2]
  }

  return out
}

// Batch FFT: run multiple independent FFTs at once.
// complexIn: interleaved complex tensor [batch * fftN * 2]
// Returns complexOut: interleaved complex tensor [batch * fftN * 2]
function gpuBatchFFT(complexIn, fftN, batch, inverse) {
  const complexOut = T.create([batch * fftN * 2], complexIn.dtype)
  dispatchFFT(complexIn, complexOut, fftN, batch, inverse || false)
  return complexOut
}

export { gpuFFT, gpuIFFT, gpuBatchFFT, nextPow2, log2int, fftParams }
