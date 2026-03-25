// smith/src/tokenizer.js
// BPE tokenizer ported directly from TinyFormer. CPU-only, no GPU needed.

// --- Training ---

function train(text, vocabSize = 512, onMerge) {
  const bytes = new Uint8Array(Buffer.from(text, 'utf-8'))
  let ids = Array.from(bytes)

  const vocab = new Array(256)
  for (let i = 0; i < 256; i++) vocab[i] = new Uint8Array([i])

  const merges = []
  const numMerges = vocabSize - 256

  for (let m = 0; m < numMerges; m++) {
    const counts = new Map()
    for (let i = 0; i < ids.length - 1; i++) {
      const key = ids[i] << 16 | ids[i + 1]
      counts.set(key, (counts.get(key) || 0) + 1)
    }
    if (counts.size === 0) break

    let bestKey = -1, bestCount = -1
    for (const [key, count] of counts) {
      if (count > bestCount) { bestCount = count; bestKey = key }
    }

    const a = bestKey >> 16
    const b = bestKey & 0xFFFF
    const newId = 256 + m

    merges.push([a, b])
    const aBytes = vocab[a]
    const bBytes = vocab[b]
    const merged = new Uint8Array(aBytes.length + bBytes.length)
    merged.set(aBytes)
    merged.set(bBytes, aBytes.length)
    vocab[newId] = merged

    if (onMerge) {
      const token = Buffer.from(merged).toString('utf-8')
      onMerge(m + 1, numMerges, a, b, token, bestCount)
    }

    const newIds = []
    let i = 0
    while (i < ids.length) {
      if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
        newIds.push(newId)
        i += 2
      } else {
        newIds.push(ids[i])
        i++
      }
    }
    ids = newIds
  }

  return { vocab, merges, vocabSize: vocab.length }
}

// --- Encode ---

function encode(text, merges) {
  const bytes = new Uint8Array(Buffer.from(text, 'utf-8'))
  let ids = Array.from(bytes)

  for (let m = 0; m < merges.length; m++) {
    const [a, b] = merges[m]
    const newId = 256 + m
    const newIds = []
    let i = 0
    while (i < ids.length) {
      if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
        newIds.push(newId)
        i += 2
      } else {
        newIds.push(ids[i])
        i++
      }
    }
    ids = newIds
  }
  return ids
}

// --- Decode ---

function decode(ids, vocab) {
  const chunks = []
  for (const id of ids) {
    if (id < vocab.length && vocab[id]) {
      chunks.push(vocab[id])
    } else {
      chunks.push(new Uint8Array([0xEF, 0xBF, 0xBD]))
    }
  }
  let totalLen = 0
  for (const c of chunks) totalLen += c.length
  const buf = new Uint8Array(totalLen)
  let offset = 0
  for (const c of chunks) { buf.set(c, offset); offset += c.length }
  return Buffer.from(buf).toString('utf-8')
}

// --- Save / Load ---

function save(path, tokenizer) {
  const { vocab, merges, vocabSize } = tokenizer
  const vocabB64 = []
  for (let i = 0; i < vocab.length; i++) {
    vocabB64.push(vocab[i] ? Buffer.from(vocab[i]).toString('base64') : null)
  }
  return Bun.write(path, JSON.stringify({ vocabSize, merges, vocab: vocabB64 }))
}

async function load(path) {
  const text = await Bun.file(path).text()
  const obj = JSON.parse(text)
  const vocab = obj.vocab.map(b64 => b64 ? new Uint8Array(Buffer.from(b64, 'base64')) : null)
  return { vocab, merges: obj.merges, vocabSize: obj.vocabSize }
}

function getTokenStr(id, vocab) {
  if (!vocab[id]) return `<unk:${id}>`
  try { return Buffer.from(vocab[id]).toString('utf-8') }
  catch { return `<bytes:${Array.from(vocab[id]).map(b => b.toString(16)).join('')}>` }
}

export { train, encode, decode, save, load, getTokenStr }
