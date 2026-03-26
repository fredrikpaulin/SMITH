# Whisper Speech-to-Text Example

Smith's first example project: a complete Whisper speech recognition implementation running on Apple Silicon GPU via Metal. Transcribes audio files using OpenAI's Whisper models with zero dependencies beyond Smith and Bun.

This document covers setup, usage, architecture, the module API for each component, and how the example uses Smith's core ops.

## Requirements

- macOS with Apple Silicon (M1–M5)
- Bun runtime
- Smith built (`bash build.sh`)

## Quick Start

```bash
# Build Smith
cd /path/to/smith && bash build.sh

# Transcribe (auto-downloads whisper-tiny on first run)
bun examples/whisper/cli.js --model whisper-tiny --file recording.wav
```

The model is fetched from Hugging Face and cached in `models/whisper-tiny/` automatically.

You can also pass a direct file path:

```bash
bun examples/whisper/cli.js --model path/to/ggml-tiny.bin --file recording.wav
```

## CLI Reference

```
bun examples/whisper/cli.js [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <id\|path>` | Registry ID or path to GGML model file | required |
| `-f, --file <path>` | Path to WAV audio file | required |
| `-l, --language <code>` | Language code (e.g. `en`, `de`, `fr`) | `en` |
| `--max-tokens <n>` | Maximum tokens to generate | `224` |
| `--temperature <t>` | Sampling temperature (0 = greedy) | `0` |
| `-o, --output <path>` | Write output to file instead of stdout | — |
| `--format <fmt>` | Output format: `text`, `json`, `srt`, `vtt` | `text` |
| `-v, --verbose` | Show model info and timing breakdown | `false` |
| `--list-models` | List available models from registry | — |

The `json` format includes the raw token IDs, language, audio duration, and (with `--verbose`) per-stage timing. The `srt` and `vtt` formats produce a single subtitle segment spanning the full audio duration — word-level timestamps are not yet supported.

## Models

Models are managed through Smith's model registry. Pass a registry ID to `--model` and it auto-downloads from [huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) on first use. Use `--list-models` to see available models.

| Registry ID | Size | Notes |
|-------------|------|-------|
| `whisper-tiny` | 75 MB | Fastest, good for clear audio |
| `whisper-tiny-en` | 75 MB | English only |
| `whisper-base` | 142 MB | Better accuracy, multilingual |
| `whisper-base-en` | 142 MB | English-only, slightly better on English |
| `whisper-small` | 466 MB | High quality, 99 languages |
| `whisper-medium` | 1.5 GB | Very high quality |

The loader supports F32, F16, and Q8_0 weight types. All models are dequantized to f32 at load time.

## Architecture

The pipeline has six stages, each implemented in a dedicated module:

```
audio.wav → WAV decode → 16kHz mono → mel spectrogram → encoder → decoder → text
             audio.js     audio.js      mel.js           model.js   model.js   tokenizer.js
```

Audio preprocessing runs on CPU. Everything inside the transformer (matmul, attention, layernorm, gelu, softmax, embedding, conv1d) runs on GPU via Smith's autograd ops.

### Whisper Model Structure

The encoder takes mel spectrogram frames and produces a sequence of hidden states. The decoder takes token IDs and encoder output, producing next-token logits.

**Encoder:** Two 1D convolutions (kernel=3) form the front-end — the first with stride=1, the second with stride=2 to halve the time resolution. Sinusoidal positional embeddings are added, then the signal passes through N transformer blocks (self-attention + FFN with GELU activation), finishing with layer normalization.

**Decoder:** Token embeddings plus learned positional embeddings feed into N transformer blocks. Each block has three sub-layers: causal self-attention, cross-attention to the encoder output, and a GELU FFN. A final layer norm feeds into a weight-tied linear projection (sharing the token embedding matrix) to produce logits over the vocabulary.

**Decoding:** Greedy (temperature=0) or temperature-based sampling. The initial token sequence is `[SOT, language, TRANSCRIBE, NOTIMESTAMPS]`. Generation stops at the EOT token or `maxTokens`.

## Smith Core Ops Used

The Whisper example drove the creation of three ops that were promoted into Smith's core library:

| Op | Source | Used for |
|----|--------|----------|
| `conv1d(input, weight, bias, { stride, padding })` | `src/autograd.js` | Encoder front-end (2 conv layers) |
| `multiHeadCrossAttention(x, kv, layer, mask?)` | `src/nn.js` | Decoder cross-attention to encoder output |
| `sinusoidalPE(maxLen, dim)` | `src/nn.js` | Encoder positional embeddings |

Other Smith ops used throughout: `matmul`, `add`, `gelu`, `layernorm`, `softmax`, `embedding`, `transpose`, `reshape`, `createMultiHeadAttention`, `multiHeadAttention`, `createLinear`, `linear`, `createCausalMask`, `noGrad`.

## Module API

### audio.js — WAV Decoder and Resampler

Pure JavaScript WAV file reader. Parses RIFF chunks, decodes PCM sample data, mixes stereo to mono, and resamples to any target rate via linear interpolation.

```js
import { loadAudio, readWav, resample } from './audio.js'
```

**`loadAudio(path)`** — High-level function. Reads a WAV file from disk, decodes it, resamples to 16kHz mono. Returns `{ samples: Float32Array, sampleRate: 16000 }`.

```js
const { samples, sampleRate } = await loadAudio('recording.wav')
// samples is a Float32Array of 16kHz mono audio in [-1, 1]
```

**`readWav(buffer)`** — Parse a WAV ArrayBuffer. Returns `{ samples: Float32Array, sampleRate, channels, bitsPerSample }`. Supports PCM int8, int16, int32, and IEEE float32. Stereo files are mixed to mono by averaging channels.

**`resample(samples, fromRate, toRate)`** — Linear interpolation resampler. Returns a new Float32Array at the target sample rate.

### mel.js — Mel Spectrogram

Pure JavaScript FFT, STFT, and mel filterbank. Matches whisper.cpp's normalization so the model sees the same input it was trained on.

```js
import { melSpectrogram, fft, stft, hannWindow, createMelFilterbank } from './mel.js'
```

**`melSpectrogram(samples, opts?)`** — End-to-end mel extraction. Takes a Float32Array of audio samples, returns `{ mel: Float32Array, nMels, numFrames }`.

```js
const { mel, nMels, numFrames } = melSpectrogram(samples, {
  nMels: 80,         // 80 for tiny/base/small/medium, 128 for large
  sampleRate: 16000,
  nFft: 400,         // FFT window size
  hopLength: 160,    // hop between frames (10ms at 16kHz)
  winLength: 400,    // analysis window length
})
// mel is a flat Float32Array of shape [nMels, numFrames]
```

Normalization follows whisper.cpp: `log10(max(mel, 1e-10))` → clamp to `(maxVal - 8)` → `(mel + 4) / 4`.

**`fft(re, im)`** — Radix-2 Cooley-Tukey FFT. Input arrays must be power-of-2 length. Returns `{ re, im }` as Float64Arrays.

**`stft(samples, nFft, hopLength, winLength)`** — Short-time Fourier transform with Hann window. Returns a 2D array of magnitude spectra, one per frame.

**`hannWindow(length)`** — Returns a Float64Array Hann window.

**`createMelFilterbank(sampleRate, nFft, nMels)`** — Builds triangular mel-scale filter bank. Returns a 2D array `[nMels][nFft/2 + 1]`.

### model.js — Whisper Encoder-Decoder

The Whisper transformer, built entirely from Smith's core ops. No custom GPU shaders — everything goes through `conv1d`, `matmul`, `multiHeadAttention`, `multiHeadCrossAttention`, etc.

```js
import {
  WHISPER_CONFIGS, createWhisperModel,
  whisperEncode, whisperDecode, whisperTranscribe,
  whisperParams,
} from './model.js'
```

**`WHISPER_CONFIGS`** — Preset configurations for each model size:

```js
WHISPER_CONFIGS.tiny
// { dim: 384, encoderLayers: 4, decoderLayers: 4, numHeads: 6,
//   ffnDim: 1536, nMels: 80, vocabSize: 51865, maxTextCtx: 448 }
```

Available sizes: `tiny`, `base`, `small`, `medium`, `large`.

**`createWhisperModel(config)`** — Allocates all model parameters (conv weights, embedding matrices, transformer blocks). Returns a model object. Weights are initialized randomly — use `loadWhisperGGML` to load real weights.

**`whisperEncode(model, melInput)`** — Encoder forward pass. Takes a flat `Float32Array` of mel spectrogram data (shape `[nMels, numFrames]`). Returns a Variable of shape `[audioCtx, dim]` where `audioCtx = numFrames / 2` (due to stride-2 conv).

**`whisperDecode(model, encoderOut, tokens)`** — Decoder forward pass. Takes the encoder output Variable and an array of token IDs. Returns logits Variable of shape `[seqLen, vocabSize]`.

**`whisperTranscribe(model, melInput, opts?)`** — Full transcription loop. Encodes audio, then autoregressively decodes tokens until EOT or `maxTokens`. Returns an array of generated token IDs (excluding the prompt tokens).

```js
const tokens = whisperTranscribe(model, mel, {
  maxTokens: 224,
  temperature: 0,           // 0 = greedy, >0 = sampling
  eotToken: 50257,
  sotToken: 50258,
  langToken: 50259,         // en
  transcribeToken: 50359,
  noTimestamps: 50363,
  onToken: (token, step) => { /* progress callback */ },
})
```

**`whisperParams(model)`** — Collects all trainable parameters for optimizer use. Returns a flat array of Variables.

#### KV-Cached Decoding

For efficient generation, the cached API avoids recomputing the full decoder sequence each step. Encoder cross-attention K/V are projected once; self-attention KV caches grow by one position per token.

```js
import {
  whisperDecodePrefill, whisperDecodeStep, whisperTranscribeCached,
  precomputeEncoderKV,
} from './model.js'
```

**`precomputeEncoderKV(model, encoderOut)`** — Projects encoder output through each decoder block's cross-attention K/V weights. Returns `[{ k, v }]` per block, where k/v are `[numHeads, audioCtx, headDim]`. Called once after encoding.

**`whisperDecodePrefill(model, encoderOut, tokens)`** — Processes the full prompt through the decoder in one pass. Returns `{ logits, selfCaches, encoderKV }`. The `selfCaches` are `[{ k, v }]` per block with shape `[numHeads, promptLen, headDim]`.

**`whisperDecodeStep(model, encoderKV, tokenId, position, selfCaches)`** — Single-token cached decode. Returns `{ logits, selfCaches }` with updated self-attention caches (grown by 1 position).

**`whisperTranscribeCached(model, melInput, opts?)`** — Drop-in replacement for `whisperTranscribe` using the prefill + step pattern. Same options, same return value. Reduces total self-attention work from O(n²) to O(n).

```js
// Same API as whisperTranscribe
const tokens = whisperTranscribeCached(model, mel, {
  maxTokens: 224,
  temperature: 0,
  onToken: (token, step) => { /* progress callback */ },
})
```

### loader.js — GGML Model Loader

Loads whisper.cpp GGML binary files and populates a Smith model with real weights.

```js
import { loadWhisperGGML } from './loader.js'
```

**`loadWhisperGGML(path)`** — Reads a `.bin` file, parses headers and tensors, creates a `createWhisperModel` with the detected config, and loads all weights. Returns `{ model, config, vocab, melFilters, hparams }`.

```js
const { model, config, vocab, melFilters, hparams } = await loadWhisperGGML('ggml-tiny.bin')
// model is ready for whisperEncode / whisperDecode / whisperTranscribe
// vocab is the BPE token list (for createTokenizer)
// melFilters is { nMel, nFft, data } — the pre-trained filterbank from the model file
//   (not used by the example — we compute our own via createMelFilterbank)
// hparams has the raw GGML header values
```

The loader handles weight name mapping from whisper.cpp's PyTorch-style names (`encoder.blocks.0.attn.query.weight`) to Smith's MHA structure (`encoderBlocks[0].mha.qProj.weight`). Weights stored as F16 or Q8_0 are dequantized to f32 at load time.

### ggml_parser.js — GGML Binary Format Parser

Pure JavaScript parser for whisper.cpp's GGML binary format. Has no dependency on Smith — can be used standalone or tested without a GPU.

```js
import { parseWhisperGGML, GGML_FILE_MAGIC, GGML_TYPE } from './ggml_parser.js'
```

**`parseWhisperGGML(buffer)`** — Parses an ArrayBuffer containing a GGML file. Returns `{ hparams, vocab, melFilters, tensors, buffer }`.

```js
const buf = await Bun.file('ggml-tiny.bin').arrayBuffer()
const parsed = parseWhisperGGML(buf)

parsed.hparams
// { nAudioCtx, nAudioState, nAudioHead, nAudioLayer,
//   nTextCtx, nTextState, nTextHead, nTextLayer,
//   nMels, ftype, nVocab }

parsed.vocab
// ['!', '"', '#', ...] — 51865 BPE tokens

parsed.tensors
// [{ name, dims, ttype, nElements, dataOffset }, ...]
```

The GGML format starts with magic bytes `0x67676d6c`, followed by 11 int32 hyperparameters, mel filterbank data, a BPE vocabulary, and then tensor data with 32-byte alignment.

**`GGML_TYPE`** — Enum of tensor types: `F32 = 0`, `F16 = 1`, `Q4_0 = 2`, `Q4_1 = 3`, `Q8_0 = 8`.

### tokenizer.js — GPT-2 BPE Decoder

Decodes Whisper's GPT-2 byte-level BPE token IDs back to text. Handles Whisper's special tokens for language, task, and timestamps.

```js
import { createTokenizer, SPECIAL_TOKENS, LANGUAGES, languageToken } from './tokenizer.js'
```

**`createTokenizer(vocab)`** — Creates a tokenizer from the vocab array loaded from a GGML file. Returns `{ decode, isTimestamp, timestampToSeconds, vocab }`.

```js
const tokenizer = createTokenizer(vocab)

tokenizer.decode([2902, 553, 262, 995])
// " and the world"

tokenizer.isTimestamp(50364)  // true — <|0.00|>
tokenizer.isTimestamp(50257)  // false — EOT

tokenizer.timestampToSeconds(50364)  // 0
tokenizer.timestampToSeconds(50414)  // 1.0 (each step = 0.02s)
```

**`SPECIAL_TOKENS`** — Constants for Whisper's control tokens:

| Token | ID | Meaning |
|-------|----|---------|
| `EOT` | 50257 | End of text |
| `SOT` | 50258 | Start of transcript |
| `TRANSLATE` | 50358 | Translation task |
| `TRANSCRIBE` | 50359 | Transcription task |
| `NOSP` | 50362 | No speech detected |
| `NOT` | 50363 | No timestamps |
| `BEG` | 50364 | First timestamp (0.00s) |

**`LANGUAGES`** — Array of 99 language codes (ISO 639-1). Index position maps to token offset: language token ID = `50259 + index`.

**`languageToken(lang)`** — Returns the token ID for a language code. Defaults to English if the code is not found.

### chunk.js — Audio Chunking and Stitching

Pure JavaScript audio chunking for long audio files. Splits audio into overlapping 30-second windows and stitches per-chunk transcriptions back together with text-based deduplication.

```js
import { chunkAudio, stitchTranscriptions, transcribeChunk } from './chunk.js'
```

**`chunkAudio(samples, opts?)`** — Split a Float32Array of 16kHz audio into overlapping chunks. Returns `[{ samples, offsetSamples }]`. Options: `chunkSamples` (default 480000 = 30s), `overlapSamples` (default 16000 = 1s). Audio shorter than 30s returns a single chunk with no copy.

**`stitchTranscriptions(chunkResults, tokenizer, opts?)`** — Merge per-chunk token arrays. Deduplicates tokens in overlap regions by checking if decoded text from the start of each chunk matches the end of accumulated text. Returns a flat token array.

**`transcribeChunk(model, chunk, opts?)`** — Transcribe a single chunk. Computes mel spectrogram and runs cached Whisper inference. Returns `{ tokens, offsetMs }`. Async (lazily loads model.js).

**`transcribeChunked(model, samples, opts?)`** — Full pipeline: chunk → transcribe each → return per-chunk results. Async. Options include `onChunk(i, total)` callback for progress.

**`WHISPER_CHUNK_SAMPLES`** — Constant: `480000` (30 seconds at 16kHz).

## Audio Format

The WAV decoder supports PCM int8, int16, int32, and IEEE float32. Any sample rate is accepted — audio is automatically resampled to 16kHz. Stereo files are mixed to mono by averaging channels.

MP3, FLAC, OGG, and other formats are not supported. Convert with FFmpeg first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Tests

The example includes 26 tests across 5 files. Non-GPU tests run anywhere (including Linux CI). GPU tests require macOS + Apple Silicon.

```bash
# Non-GPU tests (run anywhere)
bun test examples/whisper/tests/audio.test.js      # 5 tests — WAV parsing, resampling
bun test examples/whisper/tests/mel.test.js         # 8 tests — FFT, STFT, mel filterbank (260k assertions)
bun test examples/whisper/tests/tokenizer.test.js   # 7 tests — BPE decode, special tokens
bun test examples/whisper/tests/loader.test.js      # 5 tests — GGML header/tensor parsing

# GPU tests (macOS + Apple Silicon only)
bun test examples/whisper/tests/model.test.js       # encoder shapes, decoder shapes, transcribe loop

# All at once
bun test examples/whisper/tests/
```

## Limitations

- **WAV only.** No MP3, FLAC, or OGG decoding. Convert with FFmpeg.
- **No word-level timestamps.** Output is a single text string. Whisper supports timestamps via special tokens, but decoding them requires token suppression logic not yet implemented.
- **No voice activity detection.** Long audio is split at fixed 30-second boundaries with 1-second overlap. Smarter splitting at silence boundaries would improve accuracy at chunk edges.
- **CPU mel spectrogram in example.** The example's `mel.js` uses CPU FFT. Smith now provides `gpuMelSpectrogram()` as a drop-in GPU alternative for batch processing.

## Extending

To add a new audio format (e.g. MP3), implement a decoder in `audio.js` that returns `{ samples: Float32Array, sampleRate, channels }` and call `resample()` to convert to 16kHz.

To add word-level timestamps, modify `whisperTranscribe` in `model.js` to handle timestamp tokens (`tokenizer.isTimestamp(token)`) and suppress non-timestamp tokens at appropriate positions in the decoding loop. The `timestampToSeconds` function in `tokenizer.js` converts timestamp token IDs to seconds.

To use the encoder/decoder as building blocks in your own pipeline (e.g. for translation or custom decoding), import them directly:

```js
import { whisperEncode, whisperDecode } from './examples/whisper/model.js'
import { loadWhisperGGML } from './examples/whisper/loader.js'
import smith from './src/index.js'

const { model, vocab } = await loadWhisperGGML('ggml-tiny.bin')
const encoderOut = smith.noGrad(() => whisperEncode(model, melData))

// Custom decoding loop
let tokens = [50258, 50259, 50359, 50363]
for (let i = 0; i < 100; i++) {
  const logits = smith.noGrad(() => whisperDecode(model, encoderOut, tokens))
  // ... your sampling logic
}
```
