#!/usr/bin/env bun
// Quick utility: list all tensor names in the model safetensors
// Usage: bun examples/tts/verify_weights.js <model-dir>

import smith from '../../src/index.js'
const { parseSafetensors, listTensors } = smith

const modelDir = process.argv[2] || './models/Qwen3-TTS-12Hz-1.7B-Base'

async function check(path, label) {
  try {
    const buf = await Bun.file(path).arrayBuffer()
    const parsed = parseSafetensors(buf)
    const names = Object.keys(parsed.tensors).sort()
    console.log(`\n=== ${label} (${names.length} tensors) ===`)
    for (const n of names) {
      const info = parsed.tensors[n]
      console.log(`  ${n}  ${info.dtype}  [${info.shape}]`)
    }
    return names
  } catch (e) {
    console.log(`\n=== ${label}: ${e.message} ===`)
    return []
  }
}

const mainNames = await check(`${modelDir}/model.safetensors`, 'Main model')
const tokNames = await check(`${modelDir}/speech_tokenizer/model.safetensors`, 'Speech tokenizer')

// Check what model.js expects
console.log('\n=== Expected prefixes ===')
const expectedPrefixes = [
  'talker.model.codec_embedding',
  'talker.model.text_embedding',
  'talker.text_projection',
  'talker.model.layers.',
  'talker.model.norm',
  'talker.codec_head',
  'talker.code_predictor.model.codec_embedding',
  'talker.code_predictor.small_to_mtp_projection',
  'talker.code_predictor.model.layers.',
  'talker.code_predictor.model.norm',
  'talker.code_predictor.lm_head',
]
for (const prefix of expectedPrefixes) {
  const matches = mainNames.filter(n => n.startsWith(prefix))
  console.log(`  ${prefix}*  → ${matches.length} matches`)
  if (matches.length === 0) console.log(`    ⚠ NO MATCHES`)
}

const tokExpected = [
  'decoder.quantizer.rvq_first',
  'decoder.quantizer.rvq_rest',
  'decoder.pre_conv',
  'decoder.pre_transformer',
  'decoder.upsample',
  'decoder.decoder.0.conv',  // conv_in
  'decoder.decoder.1.block', // first decoder block
]
for (const prefix of tokExpected) {
  const matches = tokNames.filter(n => n.startsWith(prefix))
  console.log(`  ${prefix}*  → ${matches.length} matches`)
  if (matches.length === 0) console.log(`    ⚠ NO MATCHES`)
}
