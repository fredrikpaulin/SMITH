// smith/src/tensor.js
// GPU-backed tensor creation and shape utilities.
// Shape logic (computeStrides, broadcastShapes, etc.) ported from TinyFormer's tensor.js.
// Storage is Metal buffers accessed via bun:ffi.

import * as device from './device.js'
import { dtypeBytes, dtypeArray, toFloat16, fromFloat16 } from './dtype.js'
import { poolAlloc, poolFree } from './pool.js'

// --- Shape utilities (ported from TinyFormer) ---

function computeStrides(shape) {
  const strides = new Array(shape.length)
  let stride = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride
    stride *= shape[i]
  }
  return strides
}

function shapeSize(shape) {
  let s = 1
  for (let i = 0; i < shape.length; i++) s *= shape[i]
  return s
}

function broadcastShapes(a, b) {
  const maxLen = Math.max(a.length, b.length)
  const result = new Array(maxLen)
  for (let i = 0; i < maxLen; i++) {
    const da = i < maxLen - a.length ? 1 : a[i - (maxLen - a.length)]
    const db = i < maxLen - b.length ? 1 : b[i - (maxLen - b.length)]
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`)
    }
    result[i] = Math.max(da, db)
  }
  return result
}

function shapesEqual(a, b) {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

// --- Tensor creation ---

// Core factory: allocate GPU buffer + typed array view
function create(shape, dtype = 'f32', mode = device.SHARED) {
  const size = shapeSize(shape)
  const bytes = size * dtypeBytes(dtype)
  const buffer = poolAlloc(bytes, mode)
  const data = mode === device.SHARED
    ? device.viewBuffer(buffer, bytes, dtypeArray(dtype))
    : null // private buffers can't be read from CPU
  return {
    buffer,
    data,
    shape: shape.slice(),
    strides: computeStrides(shape),
    dtype,
    size,
    offset: 0,
  }
}

// Create tensor from JS array data
function tensor(values, shape, dtype = 'f32') {
  const size = shapeSize(shape)
  const t = create(shape, dtype)
  if (dtype === 'f16') {
    for (let i = 0; i < size; i++) t.data[i] = toFloat16(values[i] ?? 0)
  } else {
    for (let i = 0; i < size; i++) t.data[i] = values[i] ?? 0
  }
  return t
}

function zeros(shape, dtype = 'f32') {
  return create(shape, dtype)
  // data is already zeroed by Metal buffer allocation
}

function ones(shape, dtype = 'f32') {
  const t = create(shape, dtype)
  if (dtype === 'f16') {
    const one = toFloat16(1.0)
    t.data.fill(one)
  } else {
    t.data.fill(1)
  }
  return t
}

function full(shape, value, dtype = 'f32') {
  const t = create(shape, dtype)
  if (dtype === 'f16') {
    t.data.fill(toFloat16(value))
  } else {
    t.data.fill(value)
  }
  return t
}

function rand(shape, dtype = 'f32') {
  const t = create(shape, dtype)
  // rand runs on CPU — for small tensors this is fine
  if (dtype === 'f16') {
    for (let i = 0; i < t.size; i++) t.data[i] = toFloat16(Math.random())
  } else {
    for (let i = 0; i < t.size; i++) t.data[i] = Math.random()
  }
  return t
}

function randn(shape, dtype = 'f32') {
  const t = create(shape, dtype)
  // Box-Muller transform (ported from TinyFormer)
  for (let i = 0; i < t.size; i += 2) {
    const u1 = Math.random() || 1e-10
    const u2 = Math.random()
    const r = Math.sqrt(-2 * Math.log(u1))
    const theta = 2 * Math.PI * u2
    const v0 = r * Math.cos(theta)
    const v1 = r * Math.sin(theta)
    if (dtype === 'f16') {
      t.data[i] = toFloat16(v0)
      if (i + 1 < t.size) t.data[i + 1] = toFloat16(v1)
    } else {
      t.data[i] = v0
      if (i + 1 < t.size) t.data[i + 1] = v1
    }
  }
  return t
}

function scalar(value, dtype = 'f32') {
  return tensor([value], [], dtype)
}

// --- Read values from tensor (handles f16 decode) ---

function getValue(t, index) {
  if (t.dtype === 'f16') return fromFloat16(t.data[index])
  return t.data[index]
}

function toArray(t) {
  if (!t.data) throw new Error('Cannot read from private (GPU-only) tensor')
  const flat = new Array(t.size)
  if (t.dtype === 'f16') {
    for (let i = 0; i < t.size; i++) flat[i] = fromFloat16(t.data[i])
  } else {
    for (let i = 0; i < t.size; i++) flat[i] = t.data[i]
  }
  if (t.shape.length === 0) return flat[0]
  if (t.shape.length === 1) return flat
  // Reshape into nested array
  function build(dim, offset) {
    if (dim === t.shape.length - 1) return flat.slice(offset, offset + t.shape[dim])
    const result = []
    const stride = t.strides[dim]
    for (let i = 0; i < t.shape[dim]; i++) result.push(build(dim + 1, offset + i * stride))
    return result
  }
  return build(0, 0)
}

function toString(t) {
  return `tensor(${JSON.stringify(toArray(t))}, shape=[${t.shape}], dtype=${t.dtype})`
}

// --- Cleanup ---

function release(t) {
  if (t.buffer) {
    poolFree(t.buffer, t.size * dtypeBytes(t.dtype))
    t.buffer = null
    t.data = null
  }
}

// --- Byte size ---

function byteSize(t) {
  return t.size * dtypeBytes(t.dtype)
}

// --- Contiguous copy ---
// Virtual ops (transpose, reshape) produce views with non-standard strides.
// Shaders assume contiguous memory, so we need to copy before dispatch.

function isContiguous(t) {
  const expected = computeStrides(t.shape)
  if (t.strides.length !== expected.length) return false
  for (let i = 0; i < expected.length; i++) {
    if (t.strides[i] !== expected[i]) return false
  }
  return true
}

function contiguous(t) {
  if (isContiguous(t)) return t
  const out = create(t.shape, t.dtype)
  const n = t.size
  const ndim = t.shape.length
  const outStrides = computeStrides(t.shape)
  for (let i = 0; i < n; i++) {
    let remaining = i
    let srcIdx = (t.offset || 0)
    for (let d = 0; d < ndim; d++) {
      const coord = (remaining / outStrides[d]) | 0
      remaining %= outStrides[d]
      srcIdx += coord * t.strides[d]
    }
    out.data[i] = t.data[srcIdx]
  }
  return out
}

export {
  // Shape utilities
  computeStrides,
  shapeSize,
  broadcastShapes,
  shapesEqual,

  // Creation
  create,
  tensor,
  zeros,
  ones,
  full,
  rand,
  randn,
  scalar,

  // Access
  getValue,
  toArray,
  toString,
  byteSize,

  // Contiguous
  isContiguous,
  contiguous,

  // Cleanup
  release,
}
