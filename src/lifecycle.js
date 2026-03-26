// smith/src/lifecycle.js
// Tensor lifecycle management: dispose, retain, scoped cleanup, allocation guards.
// No direct dependency on device.js or pool.js — uses a pluggable release callback.

// --- Release callback ---
// Set by tensor.js to wire into poolFree without creating a circular import
// through device.js. Defaults to a no-op for testing without GPU.
let _releaseFn = () => {}

function setReleaseFn(fn) {
  _releaseFn = fn
}

// --- Scope stack ---
// Each scope tracks tensors allocated within it.
// When the scope exits, all non-retained tensors are released.

const _scopeStack = []
let _noAllocMode = false

// --- Dispose / Retain ---

// Mark a tensor as disposed. Returns its buffer to the pool immediately.
// Accessing a disposed tensor's data afterwards is undefined.
function dispose(t) {
  if (!t || t._disposed) return
  if (t._refCount !== undefined && t._refCount > 1) {
    t._refCount--
    return
  }
  t._disposed = true
  t._refCount = 0
  if (t.buffer) {
    _releaseFn(t)
    t.buffer = null
    t.data = null
  }
}

// Increment ref count. Tensor survives scope exit.
// Call dispose() to decrement. Buffer freed when count hits 0.
function retain(t) {
  if (!t) return t
  if (t._refCount === undefined) t._refCount = 1
  t._refCount++
  return t
}

// Check if a tensor has been disposed
function isDisposed(t) {
  return !!(t && t._disposed)
}

// --- Scoped cleanup: using(fn) ---

function _cleanupScope(scope) {
  for (const t of scope) {
    if (t._disposed) continue
    // Scope always calls dispose, which decrements refCount.
    // Retained tensors survive because dispose only frees at refCount 0.
    dispose(t)
  }
}

function using(fn) {
  const scope = []
  _scopeStack.push(scope)
  let result
  try {
    result = fn()
  } finally {
    _scopeStack.pop()
    _cleanupScope(scope)
  }
  return result
}

// Async version of using()
async function usingAsync(fn) {
  const scope = []
  _scopeStack.push(scope)
  let result
  try {
    result = await fn()
  } finally {
    _scopeStack.pop()
    _cleanupScope(scope)
  }
  return result
}

// --- Allocation guard: withNoAlloc(fn) ---

function withNoAlloc(fn) {
  _noAllocMode = true
  try {
    return fn()
  } finally {
    _noAllocMode = false
  }
}

// --- Hook: called by tensor.create() to register new tensors ---

function trackAllocation(t) {
  if (_noAllocMode) {
    throw new Error('Unexpected tensor allocation inside withNoAlloc()')
  }
  // Initialize lifecycle fields
  if (t._refCount === undefined) t._refCount = 1
  t._disposed = false
  // Register with current scope if one is active
  if (_scopeStack.length > 0) {
    _scopeStack[_scopeStack.length - 1].push(t)
  }
}

// --- Query ---

function activeScopeDepth() {
  return _scopeStack.length
}

export {
  dispose,
  retain,
  isDisposed,
  using,
  usingAsync,
  withNoAlloc,
  trackAllocation,
  activeScopeDepth,
  setReleaseFn,
}
