// smith/src/dtype.js
// Data type definitions and f16 encode/decode.
// f16 conversion ported from TinyFormer's optimized.js.

// --- Dtype metadata ---

const DTYPES = {
  f32: { bytes: 4, TypedArray: Float32Array },
  f16: { bytes: 2, TypedArray: Uint16Array },
  i32: { bytes: 4, TypedArray: Int32Array },
  u8:  { bytes: 1, TypedArray: Uint8Array },
}

function dtypeBytes(dtype) {
  return DTYPES[dtype]?.bytes ?? 4
}

function dtypeArray(dtype) {
  return DTYPES[dtype]?.TypedArray ?? Float32Array
}

// --- Float16 encode/decode (IEEE 754 half-precision) ---
// Ported from TinyFormer optimized.js

// Shared scratch buffers for bit manipulation (allocated once)
const _f32 = new Float32Array(1)
const _u32 = new Uint32Array(_f32.buffer)

function toFloat16(val) {
  _f32[0] = val
  const bits = _u32[0]

  const sign = (bits >> 31) & 1
  let exp = (bits >> 23) & 0xFF
  let frac = bits & 0x7FFFFF

  if (exp === 0xFF) {
    // Inf or NaN
    return (sign << 15) | 0x7C00 | (frac ? 0x200 : 0)
  }

  exp -= 127 // unbias from float32

  if (exp > 15) {
    return (sign << 15) | 0x7C00 // overflow to inf
  }

  if (exp < -14) {
    // Subnormal or zero
    if (exp < -24) return sign << 15 // too small
    frac |= 0x800000 // add implicit 1
    const shift = -1 - exp
    frac >>= shift
    return (sign << 15) | (frac >> 13)
  }

  return (sign << 15) | ((exp + 15) << 10) | (frac >> 13)
}

function fromFloat16(h) {
  const sign = (h >> 15) & 1
  const exp = (h >> 10) & 0x1F
  const frac = h & 0x3FF

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0
    // Subnormal
    const f = frac / 1024
    const val = Math.pow(2, -14) * f
    return sign ? -val : val
  }

  if (exp === 31) {
    return frac ? NaN : (sign ? -Infinity : Infinity)
  }

  const val = Math.pow(2, exp - 15) * (1 + frac / 1024)
  return sign ? -val : val
}

// Bulk conversion: Float32Array → Uint16Array (f16)
function float32ToFloat16(float32Array) {
  const out = new Uint16Array(float32Array.length)
  for (let i = 0; i < float32Array.length; i++) {
    out[i] = toFloat16(float32Array[i])
  }
  return out
}

// Bulk conversion: Uint16Array (f16) → Float32Array
function float16ToFloat32(uint16Array) {
  const out = new Float32Array(uint16Array.length)
  for (let i = 0; i < uint16Array.length; i++) {
    out[i] = fromFloat16(uint16Array[i])
  }
  return out
}

export {
  DTYPES,
  dtypeBytes,
  dtypeArray,
  toFloat16,
  fromFloat16,
  float32ToFloat16,
  float16ToFloat32,
}
