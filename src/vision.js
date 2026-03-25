// smith/src/vision.js
// Image preprocessing for vision models.
// CPU-side: resize, center crop, normalize.
// No dependencies — raw pixel manipulation.

import * as T from './tensor.js'

// ImageNet normalization constants
const IMAGENET_MEAN = [0.485, 0.456, 0.406]
const IMAGENET_STD = [0.229, 0.224, 0.225]

// CLIP normalization (same as ImageNet)
const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

// --- Resize (bilinear interpolation) ---

function resizeBilinear(pixels, srcW, srcH, dstW, dstH, channels = 3) {
  const out = new Float32Array(channels * dstH * dstW)
  const xRatio = srcW / dstW
  const yRatio = srcH / dstH

  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < dstH; y++) {
      const srcY = y * yRatio
      const y0 = Math.floor(srcY)
      const y1 = Math.min(y0 + 1, srcH - 1)
      const fy = srcY - y0

      for (let x = 0; x < dstW; x++) {
        const srcX = x * xRatio
        const x0 = Math.floor(srcX)
        const x1 = Math.min(x0 + 1, srcW - 1)
        const fx = srcX - x0

        // Bilinear interpolation
        const v00 = pixels[c * srcH * srcW + y0 * srcW + x0]
        const v01 = pixels[c * srcH * srcW + y0 * srcW + x1]
        const v10 = pixels[c * srcH * srcW + y1 * srcW + x0]
        const v11 = pixels[c * srcH * srcW + y1 * srcW + x1]

        const v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                  v10 * (1 - fx) * fy + v11 * fx * fy
        out[c * dstH * dstW + y * dstW + x] = v
      }
    }
  }
  return out
}

// --- Center crop ---

function centerCrop(pixels, srcW, srcH, cropW, cropH, channels = 3) {
  const offsetX = Math.floor((srcW - cropW) / 2)
  const offsetY = Math.floor((srcH - cropH) / 2)
  const out = new Float32Array(channels * cropH * cropW)

  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < cropH; y++) {
      for (let x = 0; x < cropW; x++) {
        out[c * cropH * cropW + y * cropW + x] =
          pixels[c * srcH * srcW + (y + offsetY) * srcW + (x + offsetX)]
      }
    }
  }
  return out
}

// --- Normalize ---

function normalize(pixels, h, w, mean, std) {
  const out = new Float32Array(3 * h * w)
  for (let c = 0; c < 3; c++) {
    const channelOffset = c * h * w
    for (let i = 0; i < h * w; i++) {
      out[channelOffset + i] = (pixels[channelOffset + i] - mean[c]) / std[c]
    }
  }
  return out
}

// --- Convert raw RGBA/RGB bytes to CHW float [0, 1] ---

function rgbaToChw(rgba, w, h) {
  const out = new Float32Array(3 * h * w)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const srcIdx = (y * w + x) * 4
      out[0 * h * w + y * w + x] = rgba[srcIdx] / 255
      out[1 * h * w + y * w + x] = rgba[srcIdx + 1] / 255
      out[2 * h * w + y * w + x] = rgba[srcIdx + 2] / 255
    }
  }
  return out
}

function rgbToChw(rgb, w, h) {
  const out = new Float32Array(3 * h * w)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const srcIdx = (y * w + x) * 3
      out[0 * h * w + y * w + x] = rgb[srcIdx] / 255
      out[1 * h * w + y * w + x] = rgb[srcIdx + 1] / 255
      out[2 * h * w + y * w + x] = rgb[srcIdx + 2] / 255
    }
  }
  return out
}

// --- Preprocessing pipelines ---

// ResNet preprocessing: resize shorter edge to 256, center crop 224, normalize
function preprocessResNet(pixels, srcW, srcH, opts = {}) {
  const { cropSize = 224, resizeShort = 256 } = opts

  // Convert to CHW float [0,1] if needed
  let chw = pixels
  if (!(pixels instanceof Float32Array) || pixels.length === srcW * srcH * 4) {
    chw = rgbaToChw(pixels, srcW, srcH)
  }

  // Resize shorter edge to resizeShort
  const scale = resizeShort / Math.min(srcW, srcH)
  const newW = Math.round(srcW * scale)
  const newH = Math.round(srcH * scale)
  const resized = resizeBilinear(chw, srcW, srcH, newW, newH)

  // Center crop
  const cropped = centerCrop(resized, newW, newH, cropSize, cropSize)

  // Normalize
  const normalized = normalize(cropped, cropSize, cropSize, IMAGENET_MEAN, IMAGENET_STD)

  // Return as tensor [1, 3, cropSize, cropSize]
  return T.tensor(Array.from(normalized), [1, 3, cropSize, cropSize])
}

// CLIP preprocessing: resize to 224, center crop 224, normalize
function preprocessCLIP(pixels, srcW, srcH, opts = {}) {
  const { imageSize = 224 } = opts

  let chw = pixels
  if (!(pixels instanceof Float32Array) || pixels.length === srcW * srcH * 4) {
    chw = rgbaToChw(pixels, srcW, srcH)
  }

  // Resize shorter edge to imageSize
  const scale = imageSize / Math.min(srcW, srcH)
  const newW = Math.round(srcW * scale)
  const newH = Math.round(srcH * scale)
  const resized = resizeBilinear(chw, srcW, srcH, newW, newH)

  // Center crop
  const cropped = centerCrop(resized, newW, newH, imageSize, imageSize)

  // Normalize with CLIP constants
  const normalized = normalize(cropped, imageSize, imageSize, CLIP_MEAN, CLIP_STD)

  return T.tensor(Array.from(normalized), [1, 3, imageSize, imageSize])
}

// --- PPM image loader (simple, no dependencies) ---
// PPM P6 format: binary RGB, common in ML pipelines

function loadPPM(buffer) {
  const bytes = new Uint8Array(buffer)
  let idx = 0

  function skipWhitespace() {
    while (idx < bytes.length && (bytes[idx] === 32 || bytes[idx] === 10 || bytes[idx] === 13 || bytes[idx] === 9)) idx++
    // Skip comments
    while (idx < bytes.length && bytes[idx] === 35) { // '#'
      while (idx < bytes.length && bytes[idx] !== 10) idx++
      idx++
    }
  }

  function readToken() {
    skipWhitespace()
    let tok = ''
    while (idx < bytes.length && bytes[idx] !== 32 && bytes[idx] !== 10 && bytes[idx] !== 13 && bytes[idx] !== 9) {
      tok += String.fromCharCode(bytes[idx++])
    }
    return tok
  }

  const magic = readToken()
  if (magic !== 'P6') throw new Error(`Unsupported PPM format: ${magic} (expected P6)`)

  const width = parseInt(readToken())
  const height = parseInt(readToken())
  const maxVal = parseInt(readToken())
  idx++ // skip single whitespace byte after maxVal

  const rgb = bytes.subarray(idx, idx + width * height * 3)
  const chw = rgbToChw(rgb, width, height)
  return { pixels: chw, width, height, maxVal }
}

export {
  resizeBilinear, centerCrop, normalize,
  rgbaToChw, rgbToChw,
  preprocessResNet, preprocessCLIP,
  loadPPM,
  IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD,
}
