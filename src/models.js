// smith/src/models.js
// Model registry, fetcher, and resolver.
// Downloads model files from Hugging Face Hub or direct URLs,
// verifies SHA-256 checksums, and caches them in models/<id>/.

import { resolve, dirname, join } from 'path'
import { existsSync, mkdirSync, readdirSync, unlinkSync, readFileSync, rmdirSync, renameSync } from 'fs'
import { createHash } from 'crypto'

const MODELS_DIR = resolve(dirname(import.meta.dir), 'models')
const REGISTRY_PATH = join(MODELS_DIR, 'registry.json')

const HF_BASE = 'https://huggingface.co'

// --- Registry ---

let _registry = null

function loadRegistry() {
  if (_registry) return _registry
  const file = Bun.file(REGISTRY_PATH)
  if (!file.size) {
    _registry = { models: {} }
    return _registry
  }
  _registry = JSON.parse(readFileSync(REGISTRY_PATH, 'utf8'))
  return _registry
}

function reloadRegistry() {
  _registry = null
  return loadRegistry()
}

/** List all models in the registry with their status (cached or not). */
function listModels() {
  const reg = loadRegistry()
  return Object.entries(reg.models).map(([id, entry]) => {
    const dir = join(MODELS_DIR, id)
    const cached = entry.files.every(f => existsSync(join(dir, f.name)))
    return { id, ...entry, cached }
  })
}

/** Get a single model entry from the registry. */
function getModel(id) {
  const reg = loadRegistry()
  const entry = reg.models[id]
  if (!entry) return null
  const dir = join(MODELS_DIR, id)
  const cached = entry.files.every(f => existsSync(join(dir, f.name)))
  return { id, ...entry, cached }
}

/** Resolve the local path for a model file. Returns null if not cached. */
function modelPath(id, filename) {
  const reg = loadRegistry()
  const entry = reg.models[id]
  if (!entry) return null
  const file = filename || entry.files[0]?.name
  if (!file) return null
  const p = join(MODELS_DIR, id, file)
  return existsSync(p) ? p : null
}

/** Get all local paths for a model's files. Returns null for uncached files. */
function modelPaths(id) {
  const reg = loadRegistry()
  const entry = reg.models[id]
  if (!entry) return null
  return entry.files.map(f => {
    const p = join(MODELS_DIR, id, f.name)
    return { name: f.name, path: existsSync(p) ? p : null }
  })
}

// --- Fetching ---

/** Build the download URL for a file in an HF repo. */
function hfUrl(repo, filename, revision = 'main') {
  return `${HF_BASE}/${repo}/resolve/${revision}/${filename}`
}

/** SHA-256 hash a file, streaming. Returns hex string. */
async function hashFile(path) {
  const file = Bun.file(path)
  const stream = file.stream()
  const hash = createHash('sha256')
  for await (const chunk of stream) {
    hash.update(chunk)
  }
  return hash.digest('hex')
}

/**
 * Download a single file with progress.
 * @param {string} url - URL to download from
 * @param {string} dest - Local destination path
 * @param {object} opts
 * @param {function} opts.onProgress - (downloaded, total) => void
 * @returns {Promise<void>}
 */
async function downloadFile(url, dest, opts = {}) {
  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`)

  const total = parseInt(res.headers.get('content-length') || '0', 10)
  let downloaded = 0

  // Ensure parent dir
  const dir = dirname(dest)
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })

  // Stream to disk
  const writer = Bun.file(dest).writer()
  for await (const chunk of res.body) {
    writer.write(chunk)
    downloaded += chunk.byteLength
    if (opts.onProgress) opts.onProgress(downloaded, total)
  }
  await writer.end()
}

/**
 * Fetch a model by registry ID.
 * Downloads all files, verifies checksums, stores in models/<id>/.
 *
 * @param {string} id - Model ID from registry
 * @param {object} opts
 * @param {function} opts.onProgress - (file, downloaded, total) => void
 * @param {boolean} opts.force - Re-download even if cached
 * @param {string} opts.revision - HF revision/branch (default: from registry or 'main')
 * @returns {Promise<{id: string, path: string, files: string[]}>}
 */
async function fetchModel(id, opts = {}) {
  const reg = loadRegistry()
  const entry = reg.models[id]
  if (!entry) throw new Error(`Unknown model "${id}". Use fetchUrl() for direct downloads or add to registry.`)

  const dir = join(MODELS_DIR, id)
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })

  const fetched = []

  for (const file of entry.files) {
    const dest = join(dir, file.name)

    // Skip if cached and not forced
    if (!opts.force && existsSync(dest)) {
      // Verify checksum if present
      if (file.sha256) {
        const hash = await hashFile(dest)
        if (hash === file.sha256) {
          fetched.push(file.name)
          continue
        }
        // Hash mismatch — re-download
      } else {
        fetched.push(file.name)
        continue
      }
    }

    // Build URL
    const revision = opts.revision || entry.revision || 'main'
    const url = file.url || hfUrl(entry.repo, file.name, revision)

    const tmpDest = dest + '.partial'
    await downloadFile(url, tmpDest, {
      onProgress: opts.onProgress
        ? (dl, tot) => opts.onProgress(file.name, dl, tot)
        : null,
    })

    // Verify checksum
    if (file.sha256) {
      const hash = await hashFile(tmpDest)
      if (hash !== file.sha256) {
        unlinkSync(tmpDest)
        throw new Error(`Checksum mismatch for ${file.name}: expected ${file.sha256}, got ${hash}`)
      }
    }

    // Atomic rename
    renameSync(tmpDest, dest)
    fetched.push(file.name)
  }

  return { id, path: dir, files: fetched }
}

/**
 * Fetch a file from a direct URL into the models directory.
 * Does not require a registry entry.
 *
 * @param {string} url - Direct download URL
 * @param {object} opts
 * @param {string} opts.id - Model ID / directory name (derived from URL if omitted)
 * @param {string} opts.filename - Override filename (derived from URL if omitted)
 * @param {string} opts.sha256 - Expected checksum
 * @param {function} opts.onProgress - (downloaded, total) => void
 * @param {boolean} opts.force - Re-download even if cached
 * @returns {Promise<{path: string, filename: string}>}
 */
async function fetchUrl(url, opts = {}) {
  const urlObj = new URL(url)
  const pathParts = urlObj.pathname.split('/').filter(Boolean)
  const filename = opts.filename || pathParts[pathParts.length - 1]
  const id = opts.id || filename.replace(/\.[^.]+$/, '')

  const dir = join(MODELS_DIR, id)
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })

  const dest = join(dir, filename)

  if (!opts.force && existsSync(dest)) {
    if (opts.sha256) {
      const hash = await hashFile(dest)
      if (hash === opts.sha256) return { path: dest, filename }
    } else {
      return { path: dest, filename }
    }
  }

  const tmpDest = dest + '.partial'
  await downloadFile(url, tmpDest, { onProgress: opts.onProgress })

  if (opts.sha256) {
    const hash = await hashFile(tmpDest)
    if (hash !== opts.sha256) {
      unlinkSync(tmpDest)
      throw new Error(`Checksum mismatch: expected ${opts.sha256}, got ${hash}`)
    }
  }

  renameSync(tmpDest, dest)
  return { path: dest, filename }
}

/**
 * Register a new model in the registry (appends to registry.json).
 *
 * @param {string} id - Model ID
 * @param {object} entry - { repo, format, description, files: [{name, sha256?, url?}], revision? }
 */
async function registerModel(id, entry) {
  const reg = loadRegistry()
  reg.models[id] = entry
  await Bun.write(REGISTRY_PATH, JSON.stringify(reg, null, 2) + '\n')
  _registry = reg
}

/**
 * Remove a model from disk cache. Does not remove from registry.
 * @param {string} id - Model ID
 */
function removeModel(id) {
  const dir = join(MODELS_DIR, id)
  if (!existsSync(dir)) return
  const files = readdirSync(dir)
  for (const f of files) {
    unlinkSync(join(dir, f))
  }
  rmdirSync(dir)
}

/** Get the models directory path. */
function modelsDir() {
  return MODELS_DIR
}

export {
  listModels,
  getModel,
  modelPath,
  modelPaths,
  fetchModel,
  fetchUrl,
  registerModel,
  removeModel,
  modelsDir,
  reloadRegistry,
  hashFile,
  hfUrl,
}
