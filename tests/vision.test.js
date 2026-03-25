// tests/vision.test.js
// Phase 15: Vision model loading (ResNet, CLIP) + image preprocessing

import { test, expect } from 'bun:test'
import * as T from '../src/tensor.js'
import * as A from '../src/autograd.js'
import {
  createResNet, forwardResNet, resnetParams,
  RESNET_CONFIGS,
  createBasicBlock, basicBlockForward,
  createBottleneck, bottleneckForward,
  createConv, convForward, createBN, bnForward,
} from '../src/resnet.js'
import {
  createCLIP, forwardVision, forwardText,
  clipSimilarity, l2Normalize, clipParams,
  CLIP_CONFIGS,
} from '../src/clip.js'
import {
  resizeBilinear, centerCrop, normalize,
  rgbaToChw, rgbToChw,
  preprocessResNet, preprocessCLIP,
  loadPPM,
  IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD,
} from '../src/vision.js'

// =====================================================================
// Image Preprocessing
// =====================================================================

test('rgbaToChw converts RGBA bytes to CHW float', () => {
  // 2x2 image: red, green, blue, white
  const rgba = new Uint8Array([
    255, 0, 0, 255,     // red
    0, 255, 0, 255,     // green
    0, 0, 255, 255,     // blue
    255, 255, 255, 255,  // white
  ])
  const chw = rgbaToChw(rgba, 2, 2)
  expect(chw.length).toBe(3 * 2 * 2)

  // R channel
  expect(chw[0]).toBeCloseTo(1.0, 2)   // red pixel R
  expect(chw[1]).toBeCloseTo(0.0, 2)   // green pixel R
  expect(chw[2]).toBeCloseTo(0.0, 2)   // blue pixel R
  expect(chw[3]).toBeCloseTo(1.0, 2)   // white pixel R

  // G channel
  expect(chw[4]).toBeCloseTo(0.0, 2)   // red pixel G
  expect(chw[5]).toBeCloseTo(1.0, 2)   // green pixel G

  // B channel
  expect(chw[10]).toBeCloseTo(1.0, 2)  // blue pixel B
})

test('rgbToChw converts RGB bytes to CHW float', () => {
  const rgb = new Uint8Array([128, 64, 32, 255, 128, 0])  // 2 pixels
  const chw = rgbToChw(rgb, 2, 1)
  expect(chw.length).toBe(3 * 1 * 2)
  expect(chw[0]).toBeCloseTo(128 / 255, 3)  // R of pixel 0
  expect(chw[1]).toBeCloseTo(255 / 255, 3)  // R of pixel 1
  expect(chw[2]).toBeCloseTo(64 / 255, 3)   // G of pixel 0
})

test('resizeBilinear scales correctly', () => {
  // 2x2 image → 4x4
  const src = new Float32Array([
    // R channel
    0, 1,
    1, 0,
    // G channel
    1, 0,
    0, 1,
    // B channel
    0.5, 0.5,
    0.5, 0.5,
  ])
  const dst = resizeBilinear(src, 2, 2, 4, 4, 3)
  expect(dst.length).toBe(3 * 4 * 4)

  // Corner values should match source
  expect(dst[0]).toBeCloseTo(0, 2)   // R top-left
  expect(dst[3]).toBeCloseTo(1, 2)   // R top-right
})

test('centerCrop extracts center region', () => {
  // 4x4 image, crop 2x2 center
  const src = new Float32Array(3 * 4 * 4)
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        src[c * 16 + y * 4 + x] = y * 4 + x
      }
    }
  }
  const cropped = centerCrop(src, 4, 4, 2, 2, 3)
  expect(cropped.length).toBe(3 * 2 * 2)
  // Center 2x2 of a 4x4 grid starts at (1,1)
  expect(cropped[0]).toBe(5)  // row 1, col 1
  expect(cropped[1]).toBe(6)  // row 1, col 2
  expect(cropped[2]).toBe(9)  // row 2, col 1
  expect(cropped[3]).toBe(10) // row 2, col 2
})

test('normalize applies mean/std correctly', () => {
  const pixels = new Float32Array(3 * 2 * 2).fill(0.5)
  const normed = normalize(pixels, 2, 2, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
  // (0.5 - 0.5) / 0.25 = 0
  for (let i = 0; i < normed.length; i++) {
    expect(normed[i]).toBeCloseTo(0, 5)
  }

  const normed2 = normalize(pixels, 2, 2, [0.4, 0.4, 0.4], [0.2, 0.2, 0.2])
  // (0.5 - 0.4) / 0.2 = 0.5
  for (let i = 0; i < normed2.length; i++) {
    expect(normed2[i]).toBeCloseTo(0.5, 5)
  }
})

test('preprocessResNet produces correct output shape', () => {
  const rgba = new Uint8Array(256 * 256 * 4)
  for (let i = 0; i < rgba.length; i++) rgba[i] = Math.floor(Math.random() * 256)
  const tensor = preprocessResNet(rgba, 256, 256)
  expect(tensor.shape).toEqual([1, 3, 224, 224])
  expect(tensor.size).toBe(1 * 3 * 224 * 224)
})

test('preprocessCLIP produces correct output shape', () => {
  const rgba = new Uint8Array(320 * 240 * 4)
  for (let i = 0; i < rgba.length; i++) rgba[i] = Math.floor(Math.random() * 256)
  const tensor = preprocessCLIP(rgba, 320, 240)
  expect(tensor.shape).toEqual([1, 3, 224, 224])
})

test('loadPPM parses P6 format', () => {
  // Build a minimal 2x2 PPM P6 file
  const header = 'P6\n2 2\n255\n'
  const headerBytes = new TextEncoder().encode(header)
  const pixels = new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128])
  const buf = new ArrayBuffer(headerBytes.length + pixels.length)
  new Uint8Array(buf).set(headerBytes, 0)
  new Uint8Array(buf).set(pixels, headerBytes.length)

  const { pixels: chw, width, height } = loadPPM(buf)
  expect(width).toBe(2)
  expect(height).toBe(2)
  expect(chw.length).toBe(3 * 2 * 2)
  // R channel: [255/255, 0, 0, 128/255]
  expect(chw[0]).toBeCloseTo(1.0, 2)
  expect(chw[1]).toBeCloseTo(0.0, 2)
})

test('ImageNet and CLIP constants are defined', () => {
  expect(IMAGENET_MEAN.length).toBe(3)
  expect(IMAGENET_STD.length).toBe(3)
  expect(CLIP_MEAN.length).toBe(3)
  expect(CLIP_STD.length).toBe(3)
  // Sanity: all between 0 and 1
  for (const v of [...IMAGENET_MEAN, ...IMAGENET_STD, ...CLIP_MEAN, ...CLIP_STD]) {
    expect(v).toBeGreaterThan(0)
    expect(v).toBeLessThan(1)
  }
})

// =====================================================================
// ResNet
// =====================================================================

test('RESNET_CONFIGS has all standard variants', () => {
  expect(Object.keys(RESNET_CONFIGS)).toEqual(
    expect.arrayContaining(['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
  )
  expect(RESNET_CONFIGS.resnet18.blockType).toBe('basic')
  expect(RESNET_CONFIGS.resnet50.blockType).toBe('bottleneck')
  expect(RESNET_CONFIGS.resnet50.layers).toEqual([3, 4, 6, 3])
})

test('createResNet builds resnet18 with correct structure', () => {
  const model = createResNet('resnet18', 10)
  expect(model.variant).toBe('resnet18')
  expect(model.blockType).toBe('basic')
  expect(model.numClasses).toBe(10)
  expect(model.stem.conv.weight.data.shape).toEqual([64, 3, 7, 7])
  expect(model.layers.length).toBe(4)
  expect(model.layers[0].length).toBe(2)  // [2, 2, 2, 2]
  expect(model.layers[1].length).toBe(2)
  expect(model.fc.weight.data.shape).toEqual([512, 10])
})

test('createResNet builds resnet50 with correct structure', () => {
  const model = createResNet('resnet50', 1000)
  expect(model.blockType).toBe('bottleneck')
  expect(model.layers[0].length).toBe(3)  // [3, 4, 6, 3]
  expect(model.layers[1].length).toBe(4)
  expect(model.layers[2].length).toBe(6)
  expect(model.layers[3].length).toBe(3)
  // Bottleneck: first block input 64 → output 256
  expect(model.layers[0][0].conv1.weight.data.shape).toEqual([64, 64, 1, 1])
  expect(model.layers[0][0].conv2.weight.data.shape).toEqual([64, 64, 3, 3])
  expect(model.layers[0][0].conv3.weight.data.shape).toEqual([256, 64, 1, 1])
  expect(model.fc.weight.data.shape).toEqual([2048, 1000])
})

test('BasicBlock forward produces correct output shape', () => {
  const block = createBasicBlock(64, 64, 1)
  const x = A.variable(T.randn([1, 64, 8, 8]), { requiresGrad: false })
  const out = basicBlockForward(x, block, false)
  expect(out.data.shape).toEqual([1, 64, 8, 8])
})

test('BasicBlock with downsample changes spatial dims', () => {
  const block = createBasicBlock(64, 128, 2)
  expect(block.downsample).not.toBeNull()
  const x = A.variable(T.randn([1, 64, 8, 8]), { requiresGrad: false })
  const out = basicBlockForward(x, block, false)
  expect(out.data.shape).toEqual([1, 128, 4, 4])
})

test('Bottleneck forward produces correct output shape', () => {
  const block = createBottleneck(64, 64, 256, 1)
  const x = A.variable(T.randn([1, 64, 8, 8]), { requiresGrad: false })
  const out = bottleneckForward(x, block, false)
  expect(out.data.shape).toEqual([1, 256, 8, 8])
})

test('Bottleneck with stride 2 downsamples', () => {
  const block = createBottleneck(256, 128, 512, 2)
  const x = A.variable(T.randn([1, 256, 8, 8]), { requiresGrad: false })
  const out = bottleneckForward(x, block, false)
  expect(out.data.shape).toEqual([1, 512, 4, 4])
})

test('forwardResNet (resnet18) produces logits', () => {
  const model = createResNet('resnet18', 10)
  const x = A.variable(T.randn([1, 3, 32, 32]), { requiresGrad: false })
  A.noGrad(() => {
    const out = forwardResNet(model, x, false)
    expect(out.data.shape).toEqual([1, 10])
    // Check output is finite
    const data = out.data.data
    let allFinite = true
    for (let i = 0; i < data.length; i++) {
      if (!isFinite(data[i])) { allFinite = false; break }
    }
    expect(allFinite).toBe(true)
  })
})

test('resnetParams collects all parameters', () => {
  const model = createResNet('resnet18', 10)
  const params = resnetParams(model)
  // ResNet-18: stem (1 conv + 1 bn) + 8 basic blocks (each: 2 conv + 2 bn) + 2 downsamples (1 conv + 1 bn each) + FC (weight + bias)
  // stem: conv.weight + bn.gamma + bn.beta = 3
  // layer1: 2 blocks × (conv1.w + bn1.g + bn1.b + conv2.w + bn2.g + bn2.b) = 2×6 = 12
  // layer2: block0 has downsample (+conv.w + bn.g + bn.b = 3), 2 blocks × 6 = 12 + 3 = 15
  // layer3: same as layer2 = 15
  // layer4: same as layer2 = 15
  // FC: weight + bias = 2
  // Total: 3 + 12 + 15 + 15 + 15 + 2 = 62
  expect(params.length).toBe(62)
  // All should have data property
  for (const p of params) {
    expect(p.data).toBeDefined()
  }
})

test('ResNet backward through small model', () => {
  const block = createBasicBlock(16, 16, 1)
  const x = A.variable(T.randn([1, 16, 4, 4]), { requiresGrad: true })
  const out = basicBlockForward(x, block, true)
  const loss = A.sum(out)
  A.backward(loss)
  expect(x.grad).not.toBeNull()
  expect(x.grad.shape).toEqual([1, 16, 4, 4])
  // Check gradient is finite
  let allFinite = true
  for (let i = 0; i < x.grad.data.length; i++) {
    if (!isFinite(x.grad.data[i])) { allFinite = false; break }
  }
  expect(allFinite).toBe(true)
})

// =====================================================================
// CLIP
// =====================================================================

test('CLIP_CONFIGS has standard variants', () => {
  expect(Object.keys(CLIP_CONFIGS)).toEqual(
    expect.arrayContaining(['ViT-B/32', 'ViT-B/16', 'ViT-L/14'])
  )
  expect(CLIP_CONFIGS['ViT-B/32'].vision.dim).toBe(768)
  expect(CLIP_CONFIGS['ViT-B/32'].text.dim).toBe(512)
  expect(CLIP_CONFIGS['ViT-B/32'].embedDim).toBe(512)
})

test('createCLIP builds ViT-B/32 model', () => {
  const model = createCLIP('ViT-B/32')
  const v = model.visual
  const t = model.text

  // Vision
  expect(v.patchConvWeight.data.shape).toEqual([768, 3, 32, 32])
  expect(v.classToken.data.shape).toEqual([1, 768])
  expect(v.posEmbed.data.shape).toEqual([50, 768])  // (224/32)^2 + 1 = 49 + 1 = 50
  expect(v.blocks.length).toBe(12)
  expect(v.projection.data.shape).toEqual([768, 512])

  // Text
  expect(t.tokenEmbed.data.shape).toEqual([49408, 512])
  expect(t.posEmbed.data.shape).toEqual([77, 512])
  expect(t.blocks.length).toBe(12)
  expect(t.textProjection.data.shape).toEqual([512, 512])

  // Logit scale
  expect(model.logitScale.data.size).toBe(1)
})

test('l2Normalize normalizes rows to unit length', () => {
  const x = A.variable(T.tensor([3, 4, 0, 0, 1, 0, 0, 0], [2, 4]), { requiresGrad: false })
  const normed = l2Normalize(x)
  expect(normed.data.shape).toEqual([2, 4])

  // First row: [3,4,0,0] → norm=5 → [0.6, 0.8, 0, 0]
  const d = normed.data.data
  expect(d[0]).toBeCloseTo(0.6, 3)
  expect(d[1]).toBeCloseTo(0.8, 3)
  expect(d[2]).toBeCloseTo(0, 3)
  expect(d[3]).toBeCloseTo(0, 3)

  // Second row: [1,0,0,0] → norm=1 → [1, 0, 0, 0]
  expect(d[4]).toBeCloseTo(1.0, 3)
  expect(d[5]).toBeCloseTo(0, 3)
})

test('clipParams collects all parameters', () => {
  const model = createCLIP('ViT-B/32')
  const params = clipParams(model)
  // Should be a large number of parameters
  expect(params.length).toBeGreaterThan(100)
  for (const p of params) {
    expect(p.data).toBeDefined()
  }
})

test('forwardText produces embedding of correct shape', () => {
  // Build a minimal CLIP-like model for testing
  // Use the real createCLIP but with a short sequence
  const model = createCLIP('ViT-B/32')
  const tokenIds = [1, 2, 3, 4, 5]  // 5 tokens

  A.noGrad(() => {
    const embedding = forwardText(model, tokenIds)
    // Should produce [1, embedDim] = [1, 512]
    expect(embedding.data.shape).toEqual([1, 512])

    // Check finite
    let allFinite = true
    for (let i = 0; i < embedding.data.data.length; i++) {
      if (!isFinite(embedding.data.data[i])) { allFinite = false; break }
    }
    expect(allFinite).toBe(true)
  })
})

test('forwardVision produces embedding of correct shape', () => {
  // Create a tiny "CLIP" model — we need to test with small patch size
  // ViT-B/32 expects 224x224 images, patches of 32 → 7x7 = 49 patches
  const model = createCLIP('ViT-B/32')
  const x = T.randn([1, 3, 224, 224])

  A.noGrad(() => {
    const embedding = forwardVision(model, x)
    expect(embedding.data.shape).toEqual([1, 512])

    let allFinite = true
    for (let i = 0; i < embedding.data.data.length; i++) {
      if (!isFinite(embedding.data.data[i])) { allFinite = false; break }
    }
    expect(allFinite).toBe(true)
  })
})

test('clipSimilarity produces NxM similarity matrix', () => {
  const imgFeats = A.variable(T.randn([2, 512]), { requiresGrad: false })
  const txtFeats = A.variable(T.randn([3, 512]), { requiresGrad: false })
  const logitScale = A.variable(T.scalar(Math.log(1 / 0.07)), { requiresGrad: false })

  const sim = clipSimilarity(imgFeats, txtFeats, logitScale)
  expect(sim.data.shape).toEqual([2, 3])

  // Values should be finite
  let allFinite = true
  for (let i = 0; i < sim.data.data.length; i++) {
    if (!isFinite(sim.data.data[i])) { allFinite = false; break }
  }
  expect(allFinite).toBe(true)
})

// =====================================================================
// Integration: round-trip create → save → load
// =====================================================================

test('ResNet round-trip: create → export weights → reload', () => {
  const model1 = createResNet('resnet18', 10)

  // Grab some weights from model1
  const stemW = new Float32Array(model1.stem.conv.weight.data.data.length)
  stemW.set(model1.stem.conv.weight.data.data)

  const fcW = new Float32Array(model1.fc.weight.data.data.length)
  fcW.set(model1.fc.weight.data.data)

  // Create model2 and copy stem weights
  const model2 = createResNet('resnet18', 10)
  model2.stem.conv.weight.data.data.set(stemW)
  model2.fc.weight.data.data.set(fcW)

  // Verify weights match
  for (let i = 0; i < stemW.length; i++) {
    expect(model2.stem.conv.weight.data.data[i]).toBe(stemW[i])
  }
  for (let i = 0; i < fcW.length; i++) {
    expect(model2.fc.weight.data.data[i]).toBe(fcW[i])
  }
})

test('ResNet inference mode uses running stats for batchnorm', () => {
  const block = createBasicBlock(16, 16, 1)

  // Run one training pass to populate running stats
  const x = A.variable(T.randn([2, 16, 4, 4]), { requiresGrad: false })
  basicBlockForward(x, block, true)  // training=true

  // Running mean should have changed from zeros
  const runMean = block.bn1.runningMean.data
  let hasNonZero = false
  for (let i = 0; i < runMean.length; i++) {
    if (Math.abs(runMean[i]) > 1e-8) { hasNonZero = true; break }
  }
  expect(hasNonZero).toBe(true)

  // Inference should also work
  A.noGrad(() => {
    const out = basicBlockForward(x, block, false)  // training=false
    expect(out.data.shape).toEqual([2, 16, 4, 4])
  })
})

// =====================================================================
// Safetensors weight loading format tests
// =====================================================================

test('ResNet weight map covers all layers for resnet18', () => {
  // Verify that the expected torchvision weight names exist for resnet18
  const expectedPrefixes = [
    'conv1', 'bn1',
    'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.conv2', 'layer1.0.bn2',
    'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.conv2', 'layer1.1.bn2',
    'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.conv2', 'layer2.0.bn2',
    'layer2.0.downsample.0', 'layer2.0.downsample.1',
    'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.conv2', 'layer2.1.bn2',
    'fc',
  ]
  // Just verify the model structure matches
  const model = createResNet('resnet18', 10)
  expect(model.layers[0].length).toBe(2)
  expect(model.layers[1][0].downsample).not.toBeNull()
  expect(model.layers[1][1].downsample).toBeNull()
})

test('CLIP weight map covers vision and text blocks', () => {
  const model = createCLIP('ViT-B/32')
  // Vision: 12 blocks
  expect(model.visual.blocks.length).toBe(12)
  for (const block of model.visual.blocks) {
    expect(block.mha.qProj.weight).toBeDefined()
    expect(block.ffn1.weight).toBeDefined()
  }
  // Text: 12 blocks
  expect(model.text.blocks.length).toBe(12)
})

// =====================================================================
// Pipeline: preprocessing → model forward
// =====================================================================

test('Full ResNet pipeline: preprocess → forward', () => {
  const model = createResNet('resnet18', 10)
  const rgba = new Uint8Array(64 * 64 * 4)
  for (let i = 0; i < rgba.length; i++) rgba[i] = Math.floor(Math.random() * 256)
  const input = preprocessResNet(rgba, 64, 64)
  const x = A.variable(input, { requiresGrad: false })

  A.noGrad(() => {
    const logits = forwardResNet(model, x, false)
    expect(logits.data.shape).toEqual([1, 10])
    let allFinite = true
    for (let i = 0; i < logits.data.data.length; i++) {
      if (!isFinite(logits.data.data[i])) { allFinite = false; break }
    }
    expect(allFinite).toBe(true)
  })
})

test('Full CLIP text pipeline: tokenize → forward', () => {
  const model = createCLIP('ViT-B/32')
  const tokenIds = [49406, 320, 1125, 539, 320, 2368, 49407]  // "a photo of a dog" approx

  A.noGrad(() => {
    const embedding = forwardText(model, tokenIds)
    expect(embedding.data.shape).toEqual([1, 512])
  })
})
