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

export {
  variable, param,
  backward, zeroGrad, noGrad,
  add, sub, mul, matmul, scale, neg,
  relu, gelu,
  softmax, layernorm, crossEntropy,
  sum, reshape,
  transposeVar as transpose,
  embedding,
  addGrad,
}
