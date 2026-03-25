// smith/src/checkpoint.js
// Model serialization: save/load weights as binary + JSON manifest.
// Ported from TinyFormer's checkpoint.js.

import * as T from './tensor.js'
import { createModel, modelParams } from './model.js'

// --- Save ---

async function saveCheckpoint(model, path) {
  const params = modelParams(model)

  let totalFloats = 0
  const paramMeta = []
  for (const p of params) {
    const size = T.shapeSize(p.data.shape)
    paramMeta.push({ shape: p.data.shape, size, offset: totalFloats })
    totalFloats += size
  }

  // Pack all weights into a single Float32Array
  const weights = new Float32Array(totalFloats)
  for (let i = 0; i < params.length; i++) {
    const c = T.contiguous(params[i].data)
    weights.set(c.data, paramMeta[i].offset)
  }

  const manifest = {
    config: model.config,
    paramMeta,
    totalFloats,
  }

  await Bun.write(path + '.json', JSON.stringify(manifest, null, 2))
  await Bun.write(path + '.bin', weights.buffer)
}

// --- Load ---

async function loadCheckpoint(path) {
  const configText = await Bun.file(path + '.json').text()
  const manifest = JSON.parse(configText)
  const weightsBuf = await Bun.file(path + '.bin').arrayBuffer()
  const weights = new Float32Array(weightsBuf)

  const model = createModel(manifest.config)
  const params = modelParams(model)

  if (params.length !== manifest.paramMeta.length) {
    throw new Error(`Param count mismatch: model has ${params.length}, checkpoint has ${manifest.paramMeta.length}`)
  }

  // Copy weights from the flat buffer into each parameter's GPU tensor
  for (let i = 0; i < params.length; i++) {
    const meta = manifest.paramMeta[i]
    const src = weights.subarray(meta.offset, meta.offset + meta.size)
    // Write directly into the existing GPU-backed typed array
    params[i].data.data.set(src)
  }

  return model
}

export { saveCheckpoint, loadCheckpoint }
