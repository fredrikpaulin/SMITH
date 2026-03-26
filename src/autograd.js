// smith/src/autograd.js
// DAG-based reverse-mode automatic differentiation.
// Ported from TinyFormer's autograd.js, with GPU tensor ops replacing CPU ops.
// Variables wrap tensors with gradients and backward functions.

import * as T from './tensor.js'
import { add as gpuAdd } from './ops/add.js'
import { sub as gpuSub } from './ops/sub.js'
import { mul as gpuMul, scale as gpuScale, neg as gpuNeg } from './ops/mul.js'
import { div as gpuDiv } from './ops/div.js'
import { matmul as gpuMatmul } from './ops/matmul.js'
import { relu as gpuRelu, reluBackward as gpuReluBackward } from './ops/relu.js'
import { gelu as gpuGelu, geluBackward as gpuGeluBackward } from './ops/gelu.js'
import { sum as gpuSum, mean as gpuMean } from './ops/reduce.js'
import { softmax as gpuSoftmax } from './ops/softmax.js'
import { layernormForward as gpuLayernormFwd, layernormBackward as gpuLayernormBwd } from './ops/layernorm.js'
import { transpose as gpuTranspose, inverseAxes } from './ops/transpose.js'
import { reshape as gpuReshape } from './ops/reshape.js'
import { flashAttentionForward as gpuFlashFwd, flashAttentionBackward as gpuFlashBwd } from './ops/flash_attention.js'
import { conv2dForward as gpuConv2dFwd, conv2dBackwardInput as gpuConv2dBwdInput, conv2dBackwardWeight as gpuConv2dBwdWeight, conv2dBackwardBias as gpuConv2dBwdBias, convOutputSize } from './ops/conv2d.js'
import { canUseWinograd, winogradForward as gpuWinogradFwd, winogradBackwardInput as gpuWinogradBwdInput } from './ops/conv2d_winograd.js'
import { shouldUseIm2col, im2colForward as gpuIm2colFwd, im2colBackwardInput as gpuIm2colBwdInput, im2colBackwardWeight as gpuIm2colBwdWeight } from './ops/conv2d_im2col.js'
import { maxPool2dForward as gpuMaxPool2dFwd, maxPool2dBackward as gpuMaxPool2dBwd, avgPool2dForward as gpuAvgPool2dFwd, avgPool2dBackward as gpuAvgPool2dBwd } from './ops/pool2d.js'
import { batchnormForward as gpuBnFwd, batchnormInference as gpuBnInfer, batchnormBackward as gpuBnBwd, createBatchNorm } from './ops/batchnorm.js'
import { ropeForward as gpuRopeFwd, ropeBackward as gpuRopeBwd, precomputeRoPE } from './ops/rope.js'
import { rmsnormForward as gpuRmsnormFwd, rmsnormBackward as gpuRmsnormBwd } from './ops/rmsnorm.js'
import { swigluForward as gpuSwigluFwd, swigluBackward as gpuSwigluBwd } from './ops/swiglu.js'
import * as conv1dGpu from './ops/conv1d.js'
import { gpuFFT as gpuFFTOp, gpuIFFT as gpuIFFTOp, gpuBatchFFT as gpuBatchFFTOp } from './ops/fft.js'

let _noGrad = false

// --- Variable creation ---

function variable(tensor, opts = {}) {
  return {
    data: tensor,
    grad: null,
    _backward: opts._backward || null,
    _deps: opts._deps || [],
    requiresGrad: opts.requiresGrad !== undefined ? opts.requiresGrad : false,
  }
}

function param(shape, initFn, dtype = 'f32') {
  const data = initFn ? initFn(shape) : T.randn(shape, dtype)
  return variable(data, { requiresGrad: true })
}

// --- Backward pass (topological sort, ported from TinyFormer) ---

function backward(v) {
  if (!v.grad) {
    v.grad = v.data.shape.length === 0 ? T.scalar(1) : T.ones(v.data.shape, v.data.dtype)
  }

  const order = []
  const visited = new Set()

  function topo(node) {
    if (visited.has(node)) return
    visited.add(node)
    for (const dep of node._deps) topo(dep)
    order.push(node)
  }
  topo(v)
  order.reverse()

  for (const node of order) {
    if (node._backward) node._backward(node.grad)
  }
}

// --- Gradient utilities ---

function zeroGrad(params) {
  for (const p of params) p.grad = null
}

function noGrad(fn) {
  const prev = _noGrad
  _noGrad = true
  try { return fn() }
  finally { _noGrad = prev }
}

// Accumulate gradient into a variable (broadcast-aware, ported from TinyFormer)
function addGrad(v, g) {
  if (!v.requiresGrad && !v._backward) return
  let grad = g
  // Sum away extra leading dims from broadcasting
  if (v.data.shape.length < grad.shape.length) {
    const extra = grad.shape.length - v.data.shape.length
    for (let i = 0; i < extra; i++) grad = gpuSum(grad, 0)
  }
  // Sum any dim that was broadcast (size 1 in v, size > 1 in grad)
  for (let i = 0; i < v.data.shape.length; i++) {
    if (v.data.shape[i] === 1 && grad.shape[i] !== 1) {
      grad = gpuReshape(gpuSum(grad, i), v.data.shape)
    }
  }
  // Ensure contiguous layout — transposed views have physical data in wrong order
  grad = T.contiguous(grad)
  v.grad = v.grad ? gpuAdd(v.grad, grad) : grad
}

// --- Autograd operations ---
// Each returns a new Variable with _backward and _deps.
// Forward compute uses GPU ops. Backward closures capture context.

function add(a, b) {
  return variable(gpuAdd(a.data, b.data), {
    _deps: [a, b],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, grad)
      addGrad(b, grad)
    },
  })
}

function sub(a, b) {
  return variable(gpuSub(a.data, b.data), {
    _deps: [a, b],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, grad)
      addGrad(b, gpuNeg(grad))
    },
  })
}

function mul(a, b) {
  return variable(gpuMul(a.data, b.data), {
    _deps: [a, b],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, gpuMul(grad, b.data))
      addGrad(b, gpuMul(grad, a.data))
    },
  })
}

function matmul(a, b) {
  return variable(gpuMatmul(a.data, b.data), {
    _deps: [a, b],
    _backward: _noGrad ? null : (grad) => {
      // dL/dA = grad @ B^T,  dL/dB = A^T @ grad
      const aNdim = a.data.shape.length
      const bNdim = b.data.shape.length

      const bAxes = []
      for (let i = 0; i < bNdim - 2; i++) bAxes.push(i)
      bAxes.push(bNdim - 1, bNdim - 2)

      const aAxes = []
      for (let i = 0; i < aNdim - 2; i++) aAxes.push(i)
      aAxes.push(aNdim - 1, aNdim - 2)

      addGrad(a, gpuMatmul(grad, gpuTranspose(b.data, bAxes)))
      addGrad(b, gpuMatmul(gpuTranspose(a.data, aAxes), grad))
    },
  })
}

function scale(a, s) {
  return variable(gpuScale(a.data, s), {
    _deps: [a],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, gpuScale(grad, s))
    },
  })
}

function neg(a) {
  return variable(gpuNeg(a.data), {
    _deps: [a],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, gpuNeg(grad))
    },
  })
}

function relu(a) {
  return variable(gpuRelu(a.data), {
    _deps: [a],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, gpuReluBackward(a.data, grad))
    },
  })
}

function gelu(a) {
  return variable(gpuGelu(a.data), {
    _deps: [a],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, gpuGeluBackward(a.data, grad))
    },
  })
}

function sum(a, axis) {
  const outData = gpuSum(a.data, axis)
  return variable(outData, {
    _deps: [a],
    _backward: _noGrad ? null : (grad) => {
      if (axis === undefined || axis === null) {
        // Scalar sum — broadcast grad to all elements
        addGrad(a, T.full(a.data.shape, T.getValue(grad, 0), a.data.dtype))
      } else {
        const ax = axis < 0 ? axis + a.data.shape.length : axis
        const expandedShape = a.data.shape.slice()
        expandedShape[ax] = 1
        const reshaped = gpuReshape(grad, expandedShape)
        // Broadcast back by adding to zeros
        addGrad(a, gpuAdd(reshaped, T.zeros(a.data.shape, a.data.dtype)))
      }
    },
  })
}

function reshape(a, newShape) {
  return variable(gpuReshape(a.data, newShape), {
    _deps: [a],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, gpuReshape(grad, a.data.shape))
    },
  })
}

function transposeVar(a, axes) {
  // Resolve default axes (reverse) so backward closure has concrete values
  if (!axes) {
    axes = []
    for (let i = a.data.shape.length - 1; i >= 0; i--) axes.push(i)
  }
  return variable(gpuTranspose(a.data, axes), {
    _deps: [a],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, gpuTranspose(grad, inverseAxes(axes)))
    },
  })
}

// Embedding lookup (ported from TinyFormer)
function embedding(indices, weight) {
  const vocab = weight.data.shape[0]
  const dim = weight.data.shape[1]
  const seqLen = indices.length
  // CPU-side gather (indices are small, this is fine)
  const out = T.create([seqLen, dim], weight.data.dtype)
  for (let i = 0; i < seqLen; i++) {
    const idx = indices[i]
    for (let j = 0; j < dim; j++) {
      out.data[i * dim + j] = weight.data.data[idx * dim + j]
    }
  }

  return variable(out, {
    _deps: [weight],
    _backward: _noGrad ? null : (grad) => {
      // Scatter-add gradients back to embedding rows
      const wGrad = weight.grad ? weight.grad : T.zeros(weight.data.shape, weight.data.dtype)
      // CPU scatter (grad is small, no GPU needed)
      for (let i = 0; i < seqLen; i++) {
        const idx = indices[i]
        for (let j = 0; j < dim; j++) {
          wGrad.data[idx * dim + j] += grad.data[i * dim + j]
        }
      }
      weight.grad = wGrad
    },
  })
}

// --- Softmax ---

function softmax(a, axis = -1) {
  const outData = gpuSoftmax(a.data, axis)
  return variable(outData, {
    _deps: [a],
    _backward: _noGrad ? null : (grad) => {
      // d(softmax)/dx = s * (grad - sum(grad * s, axis))
      const ax = axis < 0 ? axis + a.data.shape.length : axis
      const gs = gpuMul(grad, outData)
      const gsSum = gpuSum(gs, ax)
      const expandedShape = a.data.shape.slice()
      expandedShape[ax] = 1
      const gsSumExpanded = gpuReshape(gsSum, expandedShape)
      // Broadcast back
      const gsSumBroad = gpuAdd(gsSumExpanded, T.zeros(a.data.shape, a.data.dtype))
      addGrad(a, gpuMul(outData, gpuSub(grad, gsSumBroad)))
    },
  })
}

// --- Layer Normalization ---

function layernorm(a, gamma, beta, eps = 1e-5) {
  const { out: outData, xhat } = gpuLayernormFwd(a.data, gamma.data, beta.data, eps)
  return variable(outData, {
    _deps: [a, gamma, beta],
    _backward: _noGrad ? null : (grad) => {
      const { gradInput, gradGamma, gradBeta } = gpuLayernormBwd(grad, xhat, gamma.data, a.data, eps)
      addGrad(a, gradInput)
      addGrad(gamma, gradGamma)
      addGrad(beta, gradBeta)
    },
  })
}

// --- Flash Attention ---
// Fused attention: Q, K, V → O in O(n) memory using tiled online softmax.
// Saves log-sum-exp stats (L, M) for the backward pass instead of the full attention matrix.

function flashAttention(q, k, v, causal = true) {
  const { O, L, M } = gpuFlashFwd(q.data, k.data, v.data, causal)
  return variable(O, {
    _deps: [q, k, v],
    _backward: _noGrad ? null : (grad) => {
      const { dQ, dK, dV } = gpuFlashBwd(
        q.data, k.data, v.data, O, grad, L, M, causal
      )
      addGrad(q, dQ)
      addGrad(k, dK)
      addGrad(v, dV)
    },
  })
}

// --- Cross-entropy loss ---

function crossEntropy(logits, targets) {
  // logits: Variable [batch, vocab], targets: plain array of int indices
  // Numerically stable: log-softmax then pick target
  const logitsData = logits.data
  const batch = logitsData.shape[0]
  const vocab = logitsData.shape[1]

  // CPU-side cross-entropy forward (small compared to matmul)
  // log-softmax: x - max(x) - log(sum(exp(x - max(x))))
  let totalLoss = 0
  for (let i = 0; i < batch; i++) {
    let rowMax = -Infinity
    for (let j = 0; j < vocab; j++) {
      const v = logitsData.data[i * vocab + j]
      if (v > rowMax) rowMax = v
    }
    let sumExp = 0
    for (let j = 0; j < vocab; j++) {
      sumExp += Math.exp(logitsData.data[i * vocab + j] - rowMax)
    }
    const logSumExp = Math.log(sumExp)
    const logit = logitsData.data[i * vocab + targets[i]]
    totalLoss += -(logit - rowMax - logSumExp)
  }
  totalLoss /= batch

  return variable(T.scalar(totalLoss), {
    _deps: [logits],
    _backward: _noGrad ? null : (grad) => {
      // Gradient = softmax(logits) - one_hot(targets), scaled by 1/batch * grad
      const probs = gpuSoftmax(logitsData, 1)
      // Modify probs in-place (subtract 1 from target positions, scale)
      const g = grad.data ? grad.data[0] : 1
      const s = g / batch
      for (let i = 0; i < batch; i++) {
        probs.data[i * vocab + targets[i]] -= 1
      }
      addGrad(logits, gpuScale(probs, s))
    },
  })
}

// --- Conv2d ---

function conv2d(input, weight, bias, opts = {}) {
  const useWinograd = canUseWinograd(weight.data, opts)
  const useIm2col = !useWinograd && shouldUseIm2col(weight.data, opts)

  let fwd, bwdInput, bwdWeight
  if (useWinograd) {
    fwd = gpuWinogradFwd
    bwdInput = gpuWinogradBwdInput
    bwdWeight = gpuConv2dBwdWeight  // Winograd weight grad uses direct
  } else if (useIm2col) {
    fwd = gpuIm2colFwd
    bwdInput = gpuIm2colBwdInput
    bwdWeight = gpuIm2colBwdWeight
  } else {
    fwd = gpuConv2dFwd
    bwdInput = gpuConv2dBwdInput
    bwdWeight = gpuConv2dBwdWeight
  }

  const outData = fwd(input.data, weight.data, bias ? bias.data : null, opts)
  return variable(outData, {
    _deps: bias ? [input, weight, bias] : [input, weight],
    _backward: _noGrad ? null : (grad) => {
      addGrad(input, bwdInput(grad, weight.data, input.data.shape, opts))
      addGrad(weight, bwdWeight(input.data, grad, weight.data.shape, opts))
      if (bias) addGrad(bias, gpuConv2dBwdBias(grad))
    },
  })
}

// --- Conv1d ---
// CPU-side im2col + GPU matmul. Covers 1D sequence models (Whisper encoder,
// WaveNet, temporal convolutions). Layout: [C_in, L] → [C_out, L_out].
// im2col extracts [C_in * K, L_out] patches, weight is [C_out, C_in * K].

function conv1d(input, weight, bias, opts = {}) {
  const stride = opts.stride || 1
  const padding = opts.padding || 0
  const [cIn, length] = input.data.shape
  const [cOut, wCIn, kernelSize] = weight.data.shape

  // GPU im2col forward: extracts patches on GPU, then GEMM
  const { output: outData, patches } = conv1dGpu.conv1dForward(
    T.contiguous(input.data), T.contiguous(weight.data),
    bias ? T.contiguous(bias.data) : null,
    stride, padding
  )

  return variable(outData, {
    _deps: bias ? [input, weight, bias] : [input, weight],
    _backward: _noGrad ? null : (grad) => {
      // dW: grad @ patches^T → [C_out, C_in*K] → reshape to [C_out, C_in, K]
      const dW = conv1dGpu.conv1dBackwardWeight(grad, patches, weight.data.shape)
      addGrad(weight, dW)

      // dX: W^T @ grad → col2im on GPU
      const dX = conv1dGpu.conv1dBackwardInput(
        grad, T.contiguous(weight.data), length, stride, padding
      )
      addGrad(input, dX)

      if (bias) {
        // dBias: sum grad over L_out dimension
        addGrad(bias, gpuSum(grad, 1))
      }
    },
  })
}

function conv1dOutputSize(length, kernelSize, stride = 1, padding = 0) {
  return Math.floor((length + 2 * padding - kernelSize) / stride) + 1
}

// --- Max Pool 2d ---

function maxPool2d(input, opts = {}) {
  const { out, indices } = gpuMaxPool2dFwd(input.data, opts)
  return variable(out, {
    _deps: [input],
    _backward: _noGrad ? null : (grad) => {
      addGrad(input, gpuMaxPool2dBwd(grad, indices, input.data.shape))
    },
  })
}

// --- Avg Pool 2d ---

function avgPool2d(input, opts = {}) {
  const outData = gpuAvgPool2dFwd(input.data, opts)
  return variable(outData, {
    _deps: [input],
    _backward: _noGrad ? null : (grad) => {
      addGrad(input, gpuAvgPool2dBwd(grad, input.data.shape, opts))
    },
  })
}

// --- Batch Normalization ---

function batchnorm(input, layer, training = true) {
  if (!training) {
    return variable(gpuBnInfer(input.data, layer), {
      _deps: [input],
      _backward: null,
    })
  }
  const { out, savedMean, savedInvStd } = gpuBnFwd(input.data, layer)
  return variable(out, {
    _deps: [input],
    _backward: _noGrad ? null : (grad) => {
      const { gradInput, gradGamma, gradBeta } = gpuBnBwd(grad, input.data, savedMean, savedInvStd, layer)
      addGrad(input, gradInput)
      // Accumulate gamma/beta gradients directly (they're plain tensors in layer, not Variables)
      if (!layer.gamma.grad) layer.gamma.grad = gradGamma
      else layer.gamma.grad = gpuAdd(layer.gamma.grad, gradGamma)
      if (!layer.beta.grad) layer.beta.grad = gradBeta
      else layer.beta.grad = gpuAdd(layer.beta.grad, gradBeta)
    },
  })
}

// --- RoPE ---

function rope(a, ropeTable, startPos = 0) {
  const outData = gpuRopeFwd(a.data, ropeTable, startPos)
  return variable(outData, {
    _deps: [a],
    _backward: _noGrad ? null : (grad) => {
      addGrad(a, gpuRopeBwd(grad, ropeTable, startPos))
    },
  })
}

// --- RMSNorm ---

function rmsNorm(a, gamma, eps = 1e-5) {
  const outData = gpuRmsnormFwd(a.data, gamma.data, eps)
  return variable(outData, {
    _deps: [a, gamma],
    _backward: _noGrad ? null : (grad) => {
      const { gradInput, gradGamma } = gpuRmsnormBwd(grad, a.data, gamma.data, eps)
      addGrad(a, gradInput)
      addGrad(gamma, gradGamma)
    },
  })
}

// --- SwiGLU ---
// Fused silu(gate) * up — gate and up are Variables

function swiglu(gate, up) {
  const outData = gpuSwigluFwd(gate.data, up.data)
  return variable(outData, {
    _deps: [gate, up],
    _backward: _noGrad ? null : (grad) => {
      const { gradGate, gradUp } = gpuSwigluBwd(grad, gate.data, up.data)
      addGrad(gate, gradGate)
      addGrad(up, gradUp)
    },
  })
}

// --- FFT (differentiable) ---
// Forward: FFT of real input → { re: Variable [n], im: Variable [n] }
// Backward: DFT is a linear transform W. For real input x, X = Wx.
//   d(loss)/d(x) = real( W^H @ (grad_re + i * grad_im) )
//                = real( N * IFFT(grad_re + i * grad_im) )
// We track re and im as separate Variables. Each backward contributes its part.
// re backward: real(N * IFFT(grad_re + 0i)) = N * IFFT(grad_re).real
// im backward: real(N * IFFT(0 + i*grad_im)) = N * (-IFFT(grad_im).im ... no)
// Actually, for im backward: IFFT(i * g)[n] = (1/N) sum_k (i*g_k) * e^{2πikn/N}
//   real part = (1/N) sum_k g_k * (-sin(2πkn/N)) = -Im(IFFT(g))
//   So real(N * IFFT(i*g)) = -N * Im(IFFT(g))... but IFFT only returns real for real input.
// Cleaner: use gpuBatchFFT for complex IFFT directly.

function fft(input) {
  const n = input.data.shape[0]
  const { re, im } = gpuFFTOp(T.contiguous(input.data))

  const reVar = variable(re, {
    _deps: [input],
    _backward: _noGrad ? null : (grad) => {
      // IFFT of (grad_re + 0i), take real part, scale by N
      const complexGrad = T.create([n * 2], input.data.dtype)
      for (let i = 0; i < n; i++) {
        complexGrad.data[i * 2] = grad.data[i]
        complexGrad.data[i * 2 + 1] = 0
      }
      const ifftOut = gpuBatchFFTOp(complexGrad, n, 1, true) // inverse
      // IFFT result real part * N (IFFT already divides by N, so multiply back)
      const gradInput = T.create([n], input.data.dtype)
      for (let i = 0; i < n; i++) gradInput.data[i] = ifftOut.data[i * 2] * n
      addGrad(input, gradInput)
    },
  })

  const imVar = variable(im, {
    _deps: [input],
    _backward: _noGrad ? null : (grad) => {
      // IFFT of (0 + i*grad_im), take real part, scale by N
      const complexGrad = T.create([n * 2], input.data.dtype)
      for (let i = 0; i < n; i++) {
        complexGrad.data[i * 2] = 0
        complexGrad.data[i * 2 + 1] = grad.data[i]
      }
      const ifftOut = gpuBatchFFTOp(complexGrad, n, 1, true) // inverse
      const gradInput = T.create([n], input.data.dtype)
      for (let i = 0; i < n; i++) gradInput.data[i] = ifftOut.data[i * 2] * n
      addGrad(input, gradInput)
    },
  })

  return { re: reVar, im: imVar }
}

export {
  variable, param,
  backward, zeroGrad, noGrad,
  add, sub, mul, matmul, scale, neg,
  relu, gelu,
  softmax, layernorm, crossEntropy,
  flashAttention,
  sum, reshape,
  transposeVar as transpose,
  embedding,
  addGrad,
  fft,
  conv1d, conv1dOutputSize,
  conv2d, maxPool2d, avgPool2d, batchnorm,
  createBatchNorm, convOutputSize,
  rope, rmsNorm, swiglu, precomputeRoPE,
}
