import { test, expect } from 'bun:test'
import smith from '../src/index.js'

const { variable, tensor, zeros, ones, conv1d, conv1dOutputSize, backward, noGrad, param } = smith

test('conv1dOutputSize basic', () => {
  expect(conv1dOutputSize(100, 3, 1, 1)).toBe(100) // same padding
  expect(conv1dOutputSize(100, 3, 2, 1)).toBe(50)  // stride 2
  expect(conv1dOutputSize(100, 3, 1, 0)).toBe(98)  // no padding
  expect(conv1dOutputSize(100, 5, 1, 2)).toBe(100) // kernel 5, pad 2
})

test('conv1d forward shape — stride 1 same padding', () => {
  noGrad(() => {
    const x = variable(tensor(Array.from({ length: 4 * 20 }, () => Math.random()), [4, 20]))
    const w = variable(tensor(Array.from({ length: 8 * 4 * 3 }, () => Math.random() * 0.1), [8, 4, 3]))
    const b = variable(zeros([8]))

    const out = conv1d(x, w, b, { stride: 1, padding: 1 })
    expect(out.data.shape).toEqual([8, 20])
  })
})

test('conv1d forward shape — stride 2', () => {
  noGrad(() => {
    const x = variable(tensor(Array.from({ length: 4 * 20 }, () => Math.random()), [4, 20]))
    const w = variable(tensor(Array.from({ length: 8 * 4 * 3 }, () => Math.random() * 0.1), [8, 4, 3]))
    const b = variable(zeros([8]))

    const out = conv1d(x, w, b, { stride: 2, padding: 1 })
    expect(out.data.shape).toEqual([8, 10])
  })
})

test('conv1d forward shape — no padding', () => {
  noGrad(() => {
    const x = variable(tensor(Array.from({ length: 4 * 20 }, () => Math.random()), [4, 20]))
    const w = variable(tensor(Array.from({ length: 8 * 4 * 5 }, () => Math.random() * 0.1), [8, 4, 5]))
    const out = conv1d(x, w, null, { stride: 1, padding: 0 })
    expect(out.data.shape).toEqual([8, 16])
  })
})

test('conv1d forward with known values', () => {
  noGrad(() => {
    // 1-channel, length 5 input: [1, 2, 3, 4, 5]
    // 1-output, kernel 3, weights all 1, no bias
    const x = variable(tensor([1, 2, 3, 4, 5], [1, 5]))
    const w = variable(tensor([1, 1, 1], [1, 1, 3]))
    const out = conv1d(x, w, null, { stride: 1, padding: 0 })
    expect(out.data.shape).toEqual([1, 3])
    // Expected: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    expect(out.data.data[0]).toBeCloseTo(6)
    expect(out.data.data[1]).toBeCloseTo(9)
    expect(out.data.data[2]).toBeCloseTo(12)
  })
})

test('conv1d forward with padding', () => {
  noGrad(() => {
    const x = variable(tensor([1, 2, 3], [1, 3]))
    const w = variable(tensor([1, 1, 1], [1, 1, 3]))
    const out = conv1d(x, w, null, { stride: 1, padding: 1 })
    expect(out.data.shape).toEqual([1, 3])
    // padding=1: [0,1,2], [1,2,3], [2,3,0] → [3, 6, 5]
    expect(out.data.data[0]).toBeCloseTo(3)
    expect(out.data.data[1]).toBeCloseTo(6)
    expect(out.data.data[2]).toBeCloseTo(5)
  })
})

test('conv1d forward with bias', () => {
  noGrad(() => {
    const x = variable(tensor([1, 2, 3], [1, 3]))
    const w = variable(tensor([1, 1, 1], [1, 1, 3]))
    const b = variable(tensor([10], [1]))
    const out = conv1d(x, w, b, { stride: 1, padding: 0 })
    expect(out.data.shape).toEqual([1, 1])
    expect(out.data.data[0]).toBeCloseTo(16) // 1+2+3 + 10
  })
})

test('conv1d backward — weight gradient', () => {
  const x = variable(tensor([1, 2, 3, 4, 5], [1, 5]), { requiresGrad: false })
  const w = variable(tensor([1, 1, 1], [1, 1, 3]), { requiresGrad: true })
  const out = conv1d(x, w, null, { stride: 1, padding: 0 })
  // out = [6, 9, 12] → sum = 27
  const loss = smith.sum(out)
  backward(loss)

  expect(w.grad).not.toBeNull()
  expect(w.grad.shape).toEqual([1, 1, 3])
  // dW = x-patches @ grad^T
  // patches: [[1,2,3],[2,3,4],[3,4,5]] → grad all 1 → dW = [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
  expect(w.grad.data[0]).toBeCloseTo(6)
  expect(w.grad.data[1]).toBeCloseTo(9)
  expect(w.grad.data[2]).toBeCloseTo(12)
})

test('conv1d backward — input gradient', () => {
  const x = variable(tensor([1, 2, 3, 4, 5], [1, 5]), { requiresGrad: true })
  const w = variable(tensor([1, 2, 3], [1, 1, 3]), { requiresGrad: false })
  const out = conv1d(x, w, null, { stride: 1, padding: 0 })
  const loss = smith.sum(out)
  backward(loss)

  expect(x.grad).not.toBeNull()
  expect(x.grad.shape).toEqual([1, 5])
  // col2im of W^T @ grad: for each position, accumulate weight contributions
  // pos 0: only patch 0, kernel pos 0 → w[0]=1
  // pos 1: patch 0 k1 + patch 1 k0 → w[1]+w[0] = 2+1 = 3
  // pos 2: patch 0 k2 + patch 1 k1 + patch 2 k0 → 3+2+1 = 6
  // pos 3: patch 1 k2 + patch 2 k1 → 3+2 = 5
  // pos 4: patch 2 k2 → 3
  expect(x.grad.data[0]).toBeCloseTo(1)
  expect(x.grad.data[1]).toBeCloseTo(3)
  expect(x.grad.data[2]).toBeCloseTo(6)
  expect(x.grad.data[3]).toBeCloseTo(5)
  expect(x.grad.data[4]).toBeCloseTo(3)
})

test('conv1d backward — bias gradient', () => {
  const x = variable(tensor([1, 2, 3, 4, 5], [1, 5]), { requiresGrad: false })
  const w = variable(tensor([1, 1, 1], [1, 1, 3]), { requiresGrad: false })
  const b = variable(zeros([1]), { requiresGrad: true })
  const out = conv1d(x, w, b, { stride: 1, padding: 0 })
  const loss = smith.sum(out)
  backward(loss)

  expect(b.grad).not.toBeNull()
  expect(b.grad.shape).toEqual([1])
  // bias grad = sum of grad over output length = 3 (three output positions)
  expect(b.grad.data[0]).toBeCloseTo(3)
})

test('conv1d backward — multi-channel with stride and padding', () => {
  // 2 input channels, 3 output channels, kernel=3, stride=2, padding=1
  const cIn = 2, cOut = 3, len = 8, k = 3
  const x = variable(tensor(Array.from({ length: cIn * len }, (_, i) => (i + 1) * 0.1), [cIn, len]), { requiresGrad: true })
  const w = variable(tensor(Array.from({ length: cOut * cIn * k }, (_, i) => (i % 5 - 2) * 0.1), [cOut, cIn, k]), { requiresGrad: true })
  const b = variable(tensor(Array.from({ length: cOut }, (_, i) => i * 0.5), [cOut]), { requiresGrad: true })
  const out = conv1d(x, w, b, { stride: 2, padding: 1 })

  expect(out.data.shape).toEqual([cOut, conv1dOutputSize(len, k, 2, 1)])

  const loss = smith.sum(out)
  backward(loss)

  // All three grads must exist and have correct shapes
  expect(x.grad.shape).toEqual([cIn, len])
  expect(w.grad.shape).toEqual([cOut, cIn, k])
  expect(b.grad.shape).toEqual([cOut])

  // bias grad = number of output positions per output channel
  const outLen = conv1dOutputSize(len, k, 2, 1)
  for (let oc = 0; oc < cOut; oc++) {
    expect(b.grad.data[oc]).toBeCloseTo(outLen)
  }
})

test('conv1d GPU col2im matches expected — stride 1 no padding', () => {
  // Verify the col2im scatter-add is correct by checking input grad analytically
  // x = [1], len=4: [1, 2, 3, 4], w = [1, 1, 2]: [0.5, 1.0], kernel=2, stride=1, pad=0
  const x = variable(tensor([1, 2, 3, 4], [1, 4]), { requiresGrad: true })
  const w = variable(tensor([0.5, 1.0], [1, 1, 2]), { requiresGrad: false })
  const out = conv1d(x, w, null, { stride: 1, padding: 0 })
  // out = [0.5*1+1.0*2, 0.5*2+1.0*3, 0.5*3+1.0*4] = [2.5, 4.0, 5.5]
  expect(out.data.shape).toEqual([1, 3])
  expect(out.data.data[0]).toBeCloseTo(2.5)
  expect(out.data.data[1]).toBeCloseTo(4.0)
  expect(out.data.data[2]).toBeCloseTo(5.5)

  const loss = smith.sum(out)
  backward(loss)

  // dX: col2im of W^T @ ones
  // pos 0: only contributes to out[0] via k=0 → w[0]=0.5
  // pos 1: out[0] via k=1 → w[1]=1.0, out[1] via k=0 → w[0]=0.5 → total 1.5
  // pos 2: out[1] via k=1 → w[1]=1.0, out[2] via k=0 → w[0]=0.5 → total 1.5
  // pos 3: out[2] via k=1 → w[1]=1.0
  expect(x.grad.data[0]).toBeCloseTo(0.5)
  expect(x.grad.data[1]).toBeCloseTo(1.5)
  expect(x.grad.data[2]).toBeCloseTo(1.5)
  expect(x.grad.data[3]).toBeCloseTo(1.0)
})

test('finite-diff: conv1d multi-channel weight gradient', () => {
  const cIn = 3, cOut = 4, len = 12, k = 3
  const eps = 1e-4

  const xData = Array.from({ length: cIn * len }, () => Math.random() - 0.5)
  const wData = Array.from({ length: cOut * cIn * k }, () => (Math.random() - 0.5) * 0.2)
  const x = variable(tensor(xData, [cIn, len]), { requiresGrad: false })
  const w = variable(tensor(wData, [cOut, cIn, k]), { requiresGrad: true })

  // Analytical gradient
  smith.zeroGrad([w])
  const loss = smith.sum(conv1d(x, w, null, { stride: 1, padding: 1 }))
  backward(loss)
  const analyticGrad = Array.from(w.grad.data)

  // Spot-check 15 random weight positions
  const wd = w.data.data
  for (let trial = 0; trial < 15; trial++) {
    const i = Math.floor(Math.random() * wd.length)
    const orig = wd[i]
    wd[i] = orig + eps
    const lPlus = smith.noGrad(() => smith.toArray(smith.sum(conv1d(x, w, null, { stride: 1, padding: 1 })).data))
    wd[i] = orig - eps
    const lMinus = smith.noGrad(() => smith.toArray(smith.sum(conv1d(x, w, null, { stride: 1, padding: 1 })).data))
    wd[i] = orig
    const numGrad = (lPlus - lMinus) / (2 * eps)
    expect(Math.abs(analyticGrad[i] - numGrad)).toBeLessThan(5e-3)
  }
})

test('finite-diff: conv1d multi-channel input gradient (stride 2, padding 1)', () => {
  const cIn = 2, cOut = 3, len = 10, k = 3
  const eps = 1e-4

  const xData = Array.from({ length: cIn * len }, () => Math.random() - 0.5)
  const wData = Array.from({ length: cOut * cIn * k }, () => (Math.random() - 0.5) * 0.2)
  const x = variable(tensor(xData, [cIn, len]), { requiresGrad: true })
  const w = variable(tensor(wData, [cOut, cIn, k]), { requiresGrad: false })

  smith.zeroGrad([x])
  const loss = smith.sum(conv1d(x, w, null, { stride: 2, padding: 1 }))
  backward(loss)
  const analyticGrad = Array.from(x.grad.data)

  const xd = x.data.data
  for (let trial = 0; trial < 15; trial++) {
    const i = Math.floor(Math.random() * xd.length)
    const orig = xd[i]
    xd[i] = orig + eps
    const lPlus = smith.noGrad(() => smith.toArray(smith.sum(conv1d(x, w, null, { stride: 2, padding: 1 })).data))
    xd[i] = orig - eps
    const lMinus = smith.noGrad(() => smith.toArray(smith.sum(conv1d(x, w, null, { stride: 2, padding: 1 })).data))
    xd[i] = orig
    const numGrad = (lPlus - lMinus) / (2 * eps)
    expect(Math.abs(analyticGrad[i] - numGrad)).toBeLessThan(5e-3)
  }
})

test('conv1d Whisper encoder shapes', () => {
  noGrad(() => {
    // Whisper tiny: conv1(80→384, k=3, s=1, p=1) then conv2(384→384, k=3, s=2, p=1)
    const nMels = 80, dim = 384, audioLen = 3000
    const mel = variable(tensor(Array.from({ length: nMels * audioLen }, () => Math.random() * 0.1), [nMels, audioLen]))
    const w1 = variable(tensor(Array.from({ length: dim * nMels * 3 }, () => Math.random() * 0.01), [dim, nMels, 3]))
    const b1 = variable(zeros([dim]))

    const out1 = smith.gelu(conv1d(mel, w1, b1, { stride: 1, padding: 1 }))
    expect(out1.data.shape).toEqual([dim, audioLen])

    const w2 = variable(tensor(Array.from({ length: dim * dim * 3 }, () => Math.random() * 0.01), [dim, dim, 3]))
    const b2 = variable(zeros([dim]))
    const out2 = smith.gelu(conv1d(out1, w2, b2, { stride: 2, padding: 1 }))
    expect(out2.data.shape).toEqual([dim, 1500])
  })
})
