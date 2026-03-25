// smith/src/device.js
// Thin bun:ffi wrapper around libsmith.dylib.
// Loads the native library once at import time.

import { dlopen, FFIType, ptr, toArrayBuffer, CString } from 'bun:ffi'
import { resolve, dirname } from 'path'

const LIB_PATH = resolve(dirname(import.meta.dir), 'native', 'libsmith.dylib')

const { symbols: lib } = dlopen(LIB_PATH, {
  // Lifecycle
  smith_init:                       { returns: FFIType.ptr, args: [] },
  smith_destroy:                    { returns: FFIType.void, args: [FFIType.ptr] },

  // Device info
  smith_device_name:                { returns: FFIType.ptr, args: [FFIType.ptr] },
  smith_max_threadgroup_memory:     { returns: FFIType.u64, args: [FFIType.ptr] },
  smith_max_threads_per_threadgroup:{ returns: FFIType.u64, args: [FFIType.ptr] },

  // Buffers
  smith_alloc:                      { returns: FFIType.ptr, args: [FFIType.ptr, FFIType.u64, FFIType.u32] },
  smith_buffer_contents:            { returns: FFIType.ptr, args: [FFIType.ptr] },
  smith_buffer_length:              { returns: FFIType.u64, args: [FFIType.ptr] },
  smith_release_buffer:             { returns: FFIType.void, args: [FFIType.ptr] },

  // Shader library
  smith_load_library:               { returns: FFIType.ptr, args: [FFIType.ptr, FFIType.cstring] },
  smith_compile_source:             { returns: FFIType.ptr, args: [FFIType.ptr, FFIType.cstring, FFIType.ptr] },
  smith_create_pipeline:            { returns: FFIType.ptr, args: [FFIType.ptr, FFIType.ptr, FFIType.cstring] },

  // Compute — granular
  smith_begin:                      { returns: FFIType.ptr, args: [FFIType.ptr] },
  smith_set_buffer:                 { returns: FFIType.void, args: [FFIType.ptr, FFIType.ptr, FFIType.u32] },
  smith_set_bytes:                  { returns: FFIType.void, args: [FFIType.ptr, FFIType.ptr, FFIType.u32, FFIType.u32] },
  smith_set_pipeline:               { returns: FFIType.void, args: [FFIType.ptr, FFIType.ptr] },
  smith_dispatch:                   { returns: FFIType.void, args: [FFIType.ptr, FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64] },
  smith_end_sync:                   { returns: FFIType.void, args: [FFIType.ptr] },
  smith_end_async:                  { returns: FFIType.ptr, args: [FFIType.ptr] },
  smith_wait:                       { returns: FFIType.void, args: [FFIType.ptr] },
})

// Storage mode constants
const SHARED = 0
const PRIVATE = 1

// Initialize Metal device once
const ctx = lib.smith_init()
if (!ctx) throw new Error('smith: failed to initialize Metal device. Apple Silicon GPU required.')

// Resolve shader library path
const METALLIB_PATH = resolve(dirname(import.meta.dir), 'shaders', 'smith.metallib')

// Load precompiled shader library (lazy — loaded on first pipeline request)
let _shaderLib = null
function shaderLib() {
  if (!_shaderLib) {
    _shaderLib = lib.smith_load_library(ctx, Buffer.from(METALLIB_PATH + '\0'))
    if (!_shaderLib) throw new Error(`smith: failed to load shader library at ${METALLIB_PATH}`)
  }
  return _shaderLib
}

// Pipeline cache: kernel name → pipeline pointer
const _pipelines = new Map()

function pipeline(kernelName) {
  let p = _pipelines.get(kernelName)
  if (!p) {
    p = lib.smith_create_pipeline(ctx, shaderLib(), Buffer.from(kernelName + '\0'))
    if (!p) throw new Error(`smith: kernel '${kernelName}' not found in shader library`)
    _pipelines.set(kernelName, p)
  }
  return p
}

// --- Device info ---

function deviceName() {
  const namePtr = lib.smith_device_name(ctx)
  const name = new CString(namePtr)
  // Note: we should free namePtr, but CString copies it
  return name.toString()
}

function maxThreadgroupMemory() {
  return Number(lib.smith_max_threadgroup_memory(ctx))
}

function maxThreadsPerThreadgroup() {
  return Number(lib.smith_max_threads_per_threadgroup(ctx))
}

// --- Buffer operations ---

function alloc(bytes, mode = SHARED) {
  const buf = lib.smith_alloc(ctx, bytes, mode)
  if (!buf) throw new Error(`smith: failed to allocate ${bytes} bytes (mode=${mode})`)
  return buf
}

function bufferContents(buffer) {
  return lib.smith_buffer_contents(buffer)
}

function bufferLength(buffer) {
  return Number(lib.smith_buffer_length(buffer))
}

function releaseBuffer(buffer) {
  lib.smith_release_buffer(buffer)
}

// Create a typed array view into a shared Metal buffer's memory.
// This is the zero-copy bridge: JS reads/writes the same bytes the GPU sees.
function viewBuffer(buffer, byteLength, TypedArray = Float32Array) {
  const rawPtr = lib.smith_buffer_contents(buffer)
  const ab = toArrayBuffer(rawPtr, 0, byteLength)
  return new TypedArray(ab)
}

// --- Compute dispatch ---

function begin() {
  return lib.smith_begin(ctx)
}

function setBuffer(enc, buffer, index) {
  lib.smith_set_buffer(enc, buffer, index)
}

function setBytes(enc, data, length, index) {
  lib.smith_set_bytes(enc, data, length, index)
}

function setPipeline(enc, pipelinePtr) {
  lib.smith_set_pipeline(enc, pipelinePtr)
}

function dispatch(enc, gridX, gridY, gridZ, groupX, groupY, groupZ) {
  lib.smith_dispatch(enc, gridX, gridY, gridZ, groupX, groupY, groupZ)
}

function endSync(enc) {
  lib.smith_end_sync(enc)
}

function endAsync(enc) {
  return lib.smith_end_async(enc)
}

function wait(token) {
  lib.smith_wait(token)
}

// Compile shader from source string (for development / runtime kernels)
function compileSource(source) {
  const errBuf = new BigInt64Array(1) // pointer to error string
  const library = lib.smith_compile_source(ctx, Buffer.from(source + '\0'), ptr(errBuf))
  if (!library) {
    const errPtr = errBuf[0]
    const msg = errPtr ? new CString(Number(errPtr)).toString() : 'unknown error'
    throw new Error(`smith: shader compilation failed: ${msg}`)
  }
  return library
}

function createPipeline(library, fnName) {
  const p = lib.smith_create_pipeline(ctx, library, Buffer.from(fnName + '\0'))
  if (!p) throw new Error(`smith: kernel '${fnName}' not found`)
  return p
}

export {
  // Constants
  SHARED,
  PRIVATE,

  // Device
  ctx,
  deviceName,
  maxThreadgroupMemory,
  maxThreadsPerThreadgroup,

  // Buffers
  alloc,
  bufferContents,
  bufferLength,
  releaseBuffer,
  viewBuffer,

  // Pipelines
  shaderLib,
  pipeline,
  compileSource,
  createPipeline,

  // Compute
  begin,
  setBuffer,
  setBytes,
  setPipeline,
  dispatch,
  endSync,
  endAsync,
  wait,
}
