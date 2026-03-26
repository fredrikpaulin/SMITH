import { test, expect } from 'bun:test'
import smith from '../src/index.js'
import { fft as cpuFFT } from '../examples/whisper/mel.js'

const { tensor, zeros, variable, backward, noGrad, gpuFFT, gpuIFFT, gpuBatchFFT, fft } = smith

// --- Helper: compare CPU and GPU FFT ---
function cpuFFTReal(input) {
  const n = input.length
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  for (let i = 0; i < n; i++) re[i] = input[i]
  return cpuFFT(re, im)
}

// --- GPU FFT correctness ---

test('GPU FFT matches CPU FFT — impulse (delta at 0)', () => {
  const n = 8
  const input = tensor(Array.from({ length: n }, (_, i) => i === 0 ? 1 : 0), [n])
  const { re, im } = gpuFFT(input)

  // FFT of impulse = all ones (re) + all zeros (im)
  for (let i = 0; i < n; i++) {
    expect(re.data[i]).toBeCloseTo(1, 4)
    expect(im.data[i]).toBeCloseTo(0, 4)
  }
})

test('GPU FFT matches CPU FFT — DC signal (all ones)', () => {
  const n = 16
  const input = tensor(Array.from({ length: n }, () => 1), [n])
  const { re, im } = gpuFFT(input)

  // FFT of DC = [N, 0, 0, ..., 0]
  expect(re.data[0]).toBeCloseTo(n, 3)
  for (let i = 1; i < n; i++) {
    expect(re.data[i]).toBeCloseTo(0, 3)
    expect(im.data[i]).toBeCloseTo(0, 3)
  }
})

test('GPU FFT matches CPU FFT — sine wave', () => {
  const n = 64
  const freq = 3
  const vals = Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * freq * i / n))
  const input = tensor(vals, [n])

  const { re: gpuRe, im: gpuIm } = gpuFFT(input)
  const { re: cpuRe, im: cpuIm } = cpuFFTReal(vals)

  for (let i = 0; i < n; i++) {
    expect(gpuRe.data[i]).toBeCloseTo(cpuRe[i], 2)
    expect(gpuIm.data[i]).toBeCloseTo(cpuIm[i], 2)
  }
})

test('GPU FFT matches CPU FFT — random signal', () => {
  const n = 128
  const vals = Array.from({ length: n }, () => Math.random() * 2 - 1)
  const input = tensor(vals, [n])

  const { re: gpuRe, im: gpuIm } = gpuFFT(input)
  const { re: cpuRe, im: cpuIm } = cpuFFTReal(vals)

  for (let i = 0; i < n; i++) {
    expect(gpuRe.data[i]).toBeCloseTo(cpuRe[i], 2)
    expect(gpuIm.data[i]).toBeCloseTo(cpuIm[i], 2)
  }
})

// --- FFT round-trip ---

test('IFFT(FFT(x)) ≈ x — round trip', () => {
  const n = 64
  const vals = Array.from({ length: n }, () => Math.random() * 2 - 1)
  const input = tensor(vals, [n])

  const { re, im } = gpuFFT(input)
  const recovered = gpuIFFT(re, im)

  for (let i = 0; i < n; i++) {
    expect(recovered.data[i]).toBeCloseTo(vals[i], 3)
  }
})

test('IFFT(FFT(x)) round trip — large N=512', () => {
  const n = 512
  const vals = Array.from({ length: n }, () => Math.random() * 2 - 1)
  const input = tensor(vals, [n])

  const { re, im } = gpuFFT(input)
  const recovered = gpuIFFT(re, im)

  for (let i = 0; i < n; i++) {
    expect(recovered.data[i]).toBeCloseTo(vals[i], 2)
  }
})

// --- FFT zero-padding ---

test('GPU FFT handles non-power-of-2 via padding', () => {
  const n = 400  // Whisper's n_fft
  const vals = Array.from({ length: n }, () => Math.random() * 0.5)
  const input = tensor(vals, [n])

  // gpuFFT pads to 512
  const { re, im } = gpuFFT(input, 512)

  // Verify by CPU FFT with same padding
  const paddedVals = new Float64Array(512)
  for (let i = 0; i < n; i++) paddedVals[i] = vals[i]
  const cpuIm = new Float64Array(512)
  const { re: cpuRe, im: cpuImOut } = cpuFFT(paddedVals, cpuIm)

  for (let i = 0; i < 512; i++) {
    expect(re.data[i]).toBeCloseTo(cpuRe[i], 1)
    expect(im.data[i]).toBeCloseTo(cpuImOut[i], 1)
  }
})

// --- Batch FFT ---

test('batch FFT processes multiple independent FFTs', () => {
  const n = 32
  const batch = 4

  // Create batch of different signals
  const signals = Array.from({ length: batch }, () =>
    Array.from({ length: n }, () => Math.random() * 2 - 1)
  )

  // Interleaved complex input [batch * n * 2]
  const complexIn = tensor(
    signals.flatMap(sig => sig.flatMap(v => [v, 0])),
    [batch * n * 2]
  )

  const complexOut = gpuBatchFFT(complexIn, n, batch, false)

  // Compare each batch against individual GPU FFT
  for (let b = 0; b < batch; b++) {
    const singleIn = tensor(signals[b], [n])
    const { re, im } = gpuFFT(singleIn)

    for (let i = 0; i < n; i++) {
      const outIdx = (b * n + i) * 2
      expect(complexOut.data[outIdx]).toBeCloseTo(re.data[i], 2)
      expect(complexOut.data[outIdx + 1]).toBeCloseTo(im.data[i], 2)
    }
  }
})

// --- Edge cases ---

test('FFT of silence (all zeros)', () => {
  const n = 64
  const input = tensor(new Array(n).fill(0), [n])
  const { re, im } = gpuFFT(input)

  for (let i = 0; i < n; i++) {
    expect(re.data[i]).toBeCloseTo(0, 5)
    expect(im.data[i]).toBeCloseTo(0, 5)
  }
})

test('FFT of single sample', () => {
  const input = tensor([3.14], [1])
  const { re, im } = gpuFFT(input)
  // FFT of single value = that value
  expect(re.data[0]).toBeCloseTo(3.14, 3)
  expect(im.data[0]).toBeCloseTo(0, 5)
})

// --- Autograd ---

test('FFT autograd — gradient through real output', () => {
  const n = 8
  const vals = Array.from({ length: n }, () => Math.random())
  const x = variable(tensor(vals, [n]), { requiresGrad: true })

  const { re } = fft(x)
  // Sum of real part as loss
  let loss = variable(tensor([0], [1]), { requiresGrad: false })
  // Manual: loss = sum(re) = sum of FFT real parts
  // Gradient of sum(re(FFT(x))) w.r.t. x[i] = sum_k cos(2πki/N) = N if i=0, else 0... no
  // Actually: d(re(X_k))/d(x_n) = cos(2πkn/N)
  // d(sum_k re(X_k))/d(x_n) = sum_k cos(2πkn/N) = N*δ(n,0) via DFT of all-ones
  // So gradient should be [N, 0, 0, ..., 0]... wait that's the IFFT of all-ones times N
  // Let's verify with finite differences instead

  // Finite difference check
  const eps = 1e-4
  const baseRe = gpuFFT(tensor(vals, [n]))
  let baseLoss = 0
  for (let i = 0; i < n; i++) baseLoss += baseRe.re.data[i]

  const finiteDiffGrad = new Float32Array(n)
  for (let j = 0; j < n; j++) {
    const perturbed = vals.slice()
    perturbed[j] += eps
    const pRe = gpuFFT(tensor(perturbed, [n]))
    let pLoss = 0
    for (let i = 0; i < n; i++) pLoss += pRe.re.data[i]
    finiteDiffGrad[j] = (pLoss - baseLoss) / eps
  }

  // Autograd gradient
  backward(re)

  for (let i = 0; i < n; i++) {
    // re.data.data is all ones (grad of sum), so autograd should match finite diff
    expect(x.grad.data[i]).toBeCloseTo(finiteDiffGrad[i], 1)
  }
})

test('FFT autograd — gradient through imaginary output', () => {
  const n = 8
  const vals = Array.from({ length: n }, () => Math.random())
  const x = variable(tensor(vals, [n]), { requiresGrad: true })

  const { im } = fft(x)

  // Finite difference for sum(im(FFT(x)))
  const eps = 1e-4
  const baseIm = gpuFFT(tensor(vals, [n]))
  let baseLoss = 0
  for (let i = 0; i < n; i++) baseLoss += baseIm.im.data[i]

  const finiteDiffGrad = new Float32Array(n)
  for (let j = 0; j < n; j++) {
    const perturbed = vals.slice()
    perturbed[j] += eps
    const pIm = gpuFFT(tensor(perturbed, [n]))
    let pLoss = 0
    for (let i = 0; i < n; i++) pLoss += pIm.im.data[i]
    finiteDiffGrad[j] = (pLoss - baseLoss) / eps
  }

  backward(im)

  for (let i = 0; i < n; i++) {
    expect(x.grad.data[i]).toBeCloseTo(finiteDiffGrad[i], 1)
  }
})
