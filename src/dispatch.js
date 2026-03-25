// smith/src/dispatch.js
// Shader dispatch helper. Translates op-level calls into Metal command encoding.
// One function, ~40 lines of real logic. Everything else in Smith dispatches through here.

import * as device from './device.js'

// Default threadgroup size for 1D kernels
const GROUP_1D = 256

// Dispatch a compute kernel synchronously.
// kernel: string name of the Metal kernel function
// buffers: array of { buffer: ptr, index: number }
// grid: { x, y?, z? } — total thread count
// group: { x, y?, z? } — threadgroup size (optional, defaults sensible)
// params: { data: TypedArray|Buffer, index: number } — inline constant data (optional)
function run(kernel, buffers, grid, group, params) {
  const pso = device.pipeline(kernel)
  const enc = device.begin()

  device.setPipeline(enc, pso)

  for (const b of buffers) {
    device.setBuffer(enc, b.buffer, b.index)
  }

  if (params) {
    const data = params.data
    const byteLength = data.byteLength || data.length
    device.setBytes(enc, data, byteLength, params.index)
  }

  const gx = grid.x || 1
  const gy = grid.y || 1
  const gz = grid.z || 1
  const grpX = group?.x || Math.min(gx, GROUP_1D)
  const grpY = group?.y || 1
  const grpZ = group?.z || 1

  device.dispatch(enc, gx, gy, gz, grpX, grpY, grpZ)
  device.endSync(enc)
}

// Convenience: dispatch an elementwise op on a flat buffer
// kernel: shader name, a/b/out: tensor objects, size: element count
function runElementwise(kernel, buffers, size) {
  run(kernel, buffers, { x: size }, { x: Math.min(size, GROUP_1D) })
}

// Build a Uint32Array of matmul params for the shader
function matmulParams(M, N, K) {
  return new Uint32Array([M, N, K])
}

function batchMatmulParams(M, N, K, batch) {
  return new Uint32Array([M, N, K, batch])
}

// Build axis reduce params
function axisReduceParams(outer, axisSize, inner) {
  return new Uint32Array([outer, axisSize, inner])
}

// Build broadcast params for elementwise broadcast ops
// out_shape: array, a_strides: array (broadcast strides), b_strides: array
function broadcastParams(outShape, aStrides, bStrides, outSize) {
  const ndim = outShape.length
  // Pad to 8 dims
  const buf = new Uint32Array(1 + 8 + 8 + 8 + 1) // ndim + shapes + a_strides + b_strides + out_size
  buf[0] = ndim
  for (let i = 0; i < ndim; i++) {
    buf[1 + i] = outShape[i]
    buf[9 + i] = aStrides[i]
    buf[17 + i] = bStrides[i]
  }
  buf[25] = outSize
  return buf
}

// Scale params (single float, passed as bytes)
function scaleParams(value) {
  return new Float32Array([value])
}

export {
  run,
  runElementwise,
  matmulParams,
  batchMatmulParams,
  axisReduceParams,
  broadcastParams,
  scaleParams,
  GROUP_1D,
}
