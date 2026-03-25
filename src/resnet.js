// smith/src/resnet.js
// ResNet builder and weight loader from safetensors.
// Supports ResNet-18/34 (BasicBlock) and ResNet-50/101/152 (Bottleneck).
// Uses conv2d, batchnorm, relu, avgPool2d, linear from autograd.

import * as T from './tensor.js'
import * as A from './autograd.js'
import { createBatchNorm } from './ops/batchnorm.js'
import { createLinear } from './nn.js'
import { parseSafetensors, readTensor } from './safetensors.js'

// --- Weight loading helpers ---

function f16ToF32(h) {
  const sign = (h >> 15) & 1
  const exp = (h >> 10) & 0x1f
  const mant = h & 0x3ff
  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024)
  }
  if (exp === 31) return mant === 0 ? (sign ? -Infinity : Infinity) : NaN
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024)
}

function getF32(parsed, name) {
  const t = readTensor(parsed, name)
  if (t.dtype === 'F32') return { data: t.data, shape: t.shape }
  if (t.dtype === 'F16') {
    const out = new Float32Array(t.data.length)
    for (let i = 0; i < t.data.length; i++) out[i] = f16ToF32(t.data[i])
    return { data: out, shape: t.shape }
  }
  throw new Error(`Cannot convert ${t.dtype} to f32 for tensor ${name}`)
}

function loadIntoParam(param, srcData) {
  if (param.data.size !== srcData.length) {
    throw new Error(`Size mismatch: param has ${param.data.size}, source has ${srcData.length}`)
  }
  if (srcData instanceof Float32Array) {
    param.data.data.set(srcData)
  } else {
    for (let i = 0; i < srcData.length; i++) param.data.data[i] = srcData[i]
  }
}

function loadIntoTensor(tensor, srcData) {
  if (tensor.size !== srcData.length) {
    throw new Error(`Size mismatch: tensor has ${tensor.size}, source has ${srcData.length}`)
  }
  if (srcData instanceof Float32Array) {
    tensor.data.set(srcData)
  } else {
    for (let i = 0; i < srcData.length; i++) tensor.data[i] = srcData[i]
  }
}

// --- Layer constructors ---

function createConv(inC, outC, kSize, stride = 1, padding = 0, bias = false) {
  const s = Math.sqrt(2 / (inC * kSize * kSize))
  const weight = A.variable(T.tensor(
    Array.from({ length: outC * inC * kSize * kSize }, () => (Math.random() * 2 - 1) * s),
    [outC, inC, kSize, kSize]
  ), { requiresGrad: true })
  const b = bias
    ? A.variable(T.zeros([outC]), { requiresGrad: true })
    : null
  return { weight, bias: b, stride, padding }
}

function convForward(x, layer) {
  return A.conv2d(x, layer.weight, layer.bias, {
    stride: layer.stride,
    padding: layer.padding,
  })
}

function createBN(channels) {
  const bn = createBatchNorm(channels)
  // Wrap gamma/beta as Variables for gradient tracking
  bn.gammaVar = A.variable(bn.gamma, { requiresGrad: true })
  bn.betaVar = A.variable(bn.beta, { requiresGrad: true })
  return bn
}

function bnForward(x, layer, training) {
  // Sync gamma/beta from Variables back to layer before forward
  layer.gamma = layer.gammaVar.data
  layer.beta = layer.betaVar.data
  return A.batchnorm(x, layer, training)
}

// --- ResNet blocks ---

// BasicBlock: used in ResNet-18, ResNet-34
// conv3x3 → bn → relu → conv3x3 → bn → (+shortcut) → relu
function createBasicBlock(inC, outC, stride = 1) {
  const block = {
    conv1: createConv(inC, outC, 3, stride, 1),
    bn1: createBN(outC),
    conv2: createConv(outC, outC, 3, 1, 1),
    bn2: createBN(outC),
    downsample: null,
  }
  if (stride !== 1 || inC !== outC) {
    block.downsample = {
      conv: createConv(inC, outC, 1, stride, 0),
      bn: createBN(outC),
    }
  }
  return block
}

function basicBlockForward(x, block, training) {
  let out = convForward(x, block.conv1)
  out = bnForward(out, block.bn1, training)
  out = A.relu(out)
  out = convForward(out, block.conv2)
  out = bnForward(out, block.bn2, training)

  let shortcut = x
  if (block.downsample) {
    shortcut = convForward(x, block.downsample.conv)
    shortcut = bnForward(shortcut, block.downsample.bn, training)
  }
  out = A.add(out, shortcut)
  out = A.relu(out)
  return out
}

// Bottleneck: used in ResNet-50/101/152
// conv1x1 → bn → relu → conv3x3 → bn → relu → conv1x1 → bn → (+shortcut) → relu
function createBottleneck(inC, midC, outC, stride = 1) {
  const block = {
    conv1: createConv(inC, midC, 1, 1, 0),
    bn1: createBN(midC),
    conv2: createConv(midC, midC, 3, stride, 1),
    bn2: createBN(midC),
    conv3: createConv(midC, outC, 1, 1, 0),
    bn3: createBN(outC),
    downsample: null,
  }
  if (stride !== 1 || inC !== outC) {
    block.downsample = {
      conv: createConv(inC, outC, 1, stride, 0),
      bn: createBN(outC),
    }
  }
  return block
}

function bottleneckForward(x, block, training) {
  let out = convForward(x, block.conv1)
  out = bnForward(out, block.bn1, training)
  out = A.relu(out)
  out = convForward(out, block.conv2)
  out = bnForward(out, block.bn2, training)
  out = A.relu(out)
  out = convForward(out, block.conv3)
  out = bnForward(out, block.bn3, training)

  let shortcut = x
  if (block.downsample) {
    shortcut = convForward(x, block.downsample.conv)
    shortcut = bnForward(shortcut, block.downsample.bn, training)
  }
  out = A.add(out, shortcut)
  out = A.relu(out)
  return out
}

// --- ResNet configurations ---

const RESNET_CONFIGS = {
  'resnet18':  { blockType: 'basic',      layers: [2, 2, 2, 2],  channels: [64, 128, 256, 512] },
  'resnet34':  { blockType: 'basic',      layers: [3, 4, 6, 3],  channels: [64, 128, 256, 512] },
  'resnet50':  { blockType: 'bottleneck', layers: [3, 4, 6, 3],  channels: [256, 512, 1024, 2048] },
  'resnet101': { blockType: 'bottleneck', layers: [3, 4, 23, 3], channels: [256, 512, 1024, 2048] },
  'resnet152': { blockType: 'bottleneck', layers: [3, 8, 36, 3], channels: [256, 512, 1024, 2048] },
}

// Bottleneck mid-channels (1/4 of output for standard ResNets)
const BOTTLENECK_MID = { 256: 64, 512: 128, 1024: 256, 2048: 512 }

// --- Build ResNet model ---

function createResNet(variant = 'resnet50', numClasses = 1000) {
  const cfg = RESNET_CONFIGS[variant]
  if (!cfg) throw new Error(`Unknown variant: ${variant}. Use: ${Object.keys(RESNET_CONFIGS).join(', ')}`)

  const { blockType, layers, channels } = cfg

  // Stem: 7×7 conv, stride 2, pad 3 → BN → ReLU → MaxPool 3×3 stride 2 pad 1
  const stem = {
    conv: createConv(3, 64, 7, 2, 3),
    bn: createBN(64),
  }

  // Build residual layers
  const resLayers = []
  let inC = 64

  for (let layerIdx = 0; layerIdx < 4; layerIdx++) {
    const outC = channels[layerIdx]
    const numBlocks = layers[layerIdx]
    const stride = layerIdx === 0 ? 1 : 2
    const blocks = []

    for (let b = 0; b < numBlocks; b++) {
      const blockStride = b === 0 ? stride : 1
      if (blockType === 'basic') {
        blocks.push(createBasicBlock(inC, outC, blockStride))
        inC = outC
      } else {
        const midC = BOTTLENECK_MID[outC]
        blocks.push(createBottleneck(inC, midC, outC, blockStride))
        inC = outC
      }
    }
    resLayers.push(blocks)
  }

  // Classifier head: global avg pool → FC
  const fc = createLinear(channels[3], numClasses, true)

  return {
    variant,
    blockType,
    numClasses,
    stem,
    layers: resLayers,
    fc,
    config: { variant, blockType, layers, channels, numClasses },
  }
}

// --- Forward pass ---

function forwardResNet(model, x, training = false) {
  const blockFwd = model.blockType === 'basic' ? basicBlockForward : bottleneckForward

  // Stem
  let out = convForward(x, model.stem.conv)
  out = bnForward(out, model.stem.bn, training)
  out = A.relu(out)
  out = A.maxPool2d(out, { kernelSize: 3, stride: 2, padding: 1 })

  // Residual layers
  for (const layerBlocks of model.layers) {
    for (const block of layerBlocks) {
      out = blockFwd(out, block, training)
    }
  }

  // Global average pool: [N, C, H, W] → [N, C, 1, 1] → [N, C]
  const [n, c, h, w] = out.data.shape
  out = A.avgPool2d(out, { kernelSize: [h, w] })
  out = A.reshape(out, [n, c])

  // Classifier
  out = A.matmul(out, model.fc.weight)
  if (model.fc.bias) out = A.add(out, model.fc.bias)

  return out
}

// --- Collect all parameters ---

function resnetParams(model) {
  const params = []

  function addConvParams(conv) {
    params.push(conv.weight)
    if (conv.bias) params.push(conv.bias)
  }

  function addBNParams(bn) {
    params.push(bn.gammaVar)
    params.push(bn.betaVar)
  }

  function addBlockParams(block) {
    addConvParams(block.conv1)
    addBNParams(block.bn1)
    addConvParams(block.conv2)
    addBNParams(block.bn2)
    if (block.conv3) {
      addConvParams(block.conv3)
      addBNParams(block.bn3)
    }
    if (block.downsample) {
      addConvParams(block.downsample.conv)
      addBNParams(block.downsample.bn)
    }
  }

  // Stem
  addConvParams(model.stem.conv)
  addBNParams(model.stem.bn)

  // Residual layers
  for (const layerBlocks of model.layers) {
    for (const block of layerBlocks) {
      addBlockParams(block)
    }
  }

  // FC
  params.push(model.fc.weight)
  if (model.fc.bias) params.push(model.fc.bias)

  return params
}

// --- Weight name mapping (torchvision → Smith) ---

const LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']

function loadConvWeights(parsed, prefix, conv) {
  const w = getF32(parsed, `${prefix}.weight`)
  loadIntoParam(conv.weight, w.data)
  if (conv.bias && parsed.tensors[`${prefix}.bias`]) {
    const b = getF32(parsed, `${prefix}.bias`)
    loadIntoParam(conv.bias, b.data)
  }
}

function loadBNWeights(parsed, prefix, bn) {
  const gamma = getF32(parsed, `${prefix}.weight`)
  const beta = getF32(parsed, `${prefix}.bias`)
  const runMean = getF32(parsed, `${prefix}.running_mean`)
  const runVar = getF32(parsed, `${prefix}.running_var`)
  loadIntoTensor(bn.gamma, gamma.data)
  loadIntoTensor(bn.beta, beta.data)
  loadIntoTensor(bn.runningMean, runMean.data)
  loadIntoTensor(bn.runningVar, runVar.data)
  // Sync to Variables
  bn.gammaVar.data = bn.gamma
  bn.betaVar.data = bn.beta
}

function loadBlockWeights(parsed, prefix, block, blockType) {
  loadConvWeights(parsed, `${prefix}.conv1`, block.conv1)
  loadBNWeights(parsed, `${prefix}.bn1`, block.bn1)
  loadConvWeights(parsed, `${prefix}.conv2`, block.conv2)
  loadBNWeights(parsed, `${prefix}.bn2`, block.bn2)
  if (blockType === 'bottleneck') {
    loadConvWeights(parsed, `${prefix}.conv3`, block.conv3)
    loadBNWeights(parsed, `${prefix}.bn3`, block.bn3)
  }
  if (block.downsample) {
    loadConvWeights(parsed, `${prefix}.downsample.0`, block.downsample.conv)
    loadBNWeights(parsed, `${prefix}.downsample.1`, block.downsample.bn)
  }
}

function mapResNetWeights(parsed, model) {
  // Stem
  loadConvWeights(parsed, 'conv1', model.stem.conv)
  loadBNWeights(parsed, 'bn1', model.stem.bn)

  // Residual layers
  for (let i = 0; i < 4; i++) {
    const blocks = model.layers[i]
    for (let b = 0; b < blocks.length; b++) {
      loadBlockWeights(parsed, `${LAYER_NAMES[i]}.${b}`, blocks[b], model.blockType)
    }
  }

  // FC head
  const fcW = getF32(parsed, 'fc.weight')
  // torchvision FC is [outDim, inDim], Smith linear is [inDim, outDim]
  const [outDim, inDim] = fcW.shape
  const transposed = new Float32Array(inDim * outDim)
  for (let r = 0; r < outDim; r++) {
    for (let c = 0; c < inDim; c++) {
      transposed[c * outDim + r] = fcW.data[r * inDim + c]
    }
  }
  loadIntoParam(model.fc.weight, transposed)

  const fcB = getF32(parsed, 'fc.bias')
  loadIntoParam(model.fc.bias, fcB.data)
}

// --- High-level loader ---

async function loadResNet(path, opts = {}) {
  const { variant = 'resnet50', numClasses = 1000 } = opts
  const buf = await Bun.file(path).arrayBuffer()
  const parsed = parseSafetensors(buf)

  const model = createResNet(variant, numClasses)
  mapResNetWeights(parsed, model)

  return {
    model,
    config: model.config,
    forward: (x, training) => forwardResNet(model, x, training),
    params: () => resnetParams(model),
  }
}

export {
  createResNet, forwardResNet, resnetParams,
  mapResNetWeights, loadResNet,
  RESNET_CONFIGS,
  // Building blocks (for custom architectures)
  createBasicBlock, basicBlockForward,
  createBottleneck, bottleneckForward,
  createConv, convForward, createBN, bnForward,
}
