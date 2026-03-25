// smith/src/f16mode.js
// Global f16 mode toggle and mixed precision training utilities.
// When f16 mode is enabled, tensor creation defaults to f16 dtype
// and ops dispatch f16 shaders.

let _f16Enabled = false

function f16Mode(enabled) {
  if (enabled !== undefined) _f16Enabled = !!enabled
  return _f16Enabled
}

function defaultDtype() {
  return _f16Enabled ? 'f16' : 'f32'
}

// --- Dynamic Loss Scaler ---
// Prevents gradient underflow in f16 training.
// Starts at a high scale factor (e.g. 2^16) and halves on NaN/Inf,
// doubles when consecutive steps succeed.

function createLossScaler(opts = {}) {
  const initScale = opts.initScale ?? 65536  // 2^16
  const growthInterval = opts.growthInterval ?? 2000
  const growthFactor = opts.growthFactor ?? 2
  const backoffFactor = opts.backoffFactor ?? 0.5
  const minScale = opts.minScale ?? 1

  let scale = initScale
  let goodSteps = 0

  function getScale() { return scale }

  // Scale the loss before backward pass
  function scaleUp(loss) { return loss * scale }

  // Unscale gradients after backward pass, returns false if NaN/Inf detected
  function unscale(gradArrays) {
    const invScale = 1 / scale
    let hasNaN = false

    for (const grad of gradArrays) {
      if (!grad || !grad.data) continue
      for (let i = 0; i < grad.data.length; i++) {
        const v = grad.data[i] * invScale
        if (!isFinite(v)) hasNaN = true
        grad.data[i] = v
      }
    }

    return !hasNaN
  }

  // Update scale factor after a step
  // ok: true if gradients were finite, false if NaN/Inf
  function update(ok) {
    if (ok) {
      goodSteps++
      if (goodSteps >= growthInterval) {
        scale *= growthFactor
        goodSteps = 0
      }
    } else {
      scale = Math.max(minScale, scale * backoffFactor)
      goodSteps = 0
    }
  }

  return { getScale, scaleUp, unscale, update }
}

export { f16Mode, defaultDtype, createLossScaler }
