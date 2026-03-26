# Whisper — Speech-to-Text on Apple Silicon GPU

A Whisper speech recognition implementation powered by Smith. Transcribes audio files using OpenAI's Whisper model running entirely on Apple Silicon GPU via Metal compute shaders.

Zero dependencies beyond Smith and Bun.

## Quick Start

```bash
# 1. Build Smith (if not already done)
cd /path/to/smith && bash build.sh

# 2. Download a Whisper model
curl -L -o ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

# 3. Transcribe
bun examples/whisper/cli.js --model ggml-tiny.bin --file recording.wav
```

## CLI Options

```
-m, --model <path>       Path to whisper.cpp GGML model (.bin)
-f, --file <path>        Path to audio file (.wav, 16-bit PCM)
-l, --language <code>    Language code (default: en)
--max-tokens <n>         Maximum tokens to generate (default: 224)
--temperature <t>        Sampling temperature, 0 = greedy (default: 0)
-o, --output <path>      Write output to file
--format <fmt>           Output format: text, json, srt, vtt (default: text)
-v, --verbose            Show timing and model info
```

## Architecture

```
audio.wav ──► WAV decoder ──► 16kHz mono ──► mel spectrogram ──► Whisper encoder ──► Whisper decoder ──► text
              (audio.js)      (resample)     (mel.js)           (model.js)          (model.js)         (tokenizer.js)
```

### Components

| File | Purpose |
|------|---------|
| `cli.js` | CLI entry point |
| `audio.js` | WAV file decoder + resampler (pure JS) |
| `mel.js` | FFT, STFT, mel filterbank (pure JS) |
| `model.js` | Whisper encoder-decoder transformer (Smith GPU ops) |
| `loader.js` | Load weights from whisper.cpp GGML format |
| `ggml_parser.js` | Pure JS GGML binary parser (no GPU dependency) |
| `tokenizer.js` | GPT-2 BPE token decoder with Whisper special tokens |

### What runs on GPU

Everything inside the transformer — matmul, attention, layernorm, gelu, softmax, embedding lookup. The audio preprocessing (WAV decode, FFT, mel filterbank) runs on CPU since it's I/O-bound and only happens once per file.

Conv1d in the encoder uses Smith's matmul via im2col — no dedicated 1D conv shader yet.

## Models

Download GGML models from [huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp):

| Model | Size | Quality |
|-------|------|---------|
| `ggml-tiny.bin` | 75 MB | Good for short, clear audio |
| `ggml-base.en.bin` | 142 MB | Better accuracy, English only |
| `ggml-small.bin` | 466 MB | High quality, multilingual |
| `ggml-medium.bin` | 1.5 GB | Very high quality |

## Audio Format

Currently supports WAV files (PCM int16, int32, or float32). Any sample rate — automatically resampled to 16kHz. Mono or stereo (stereo is mixed to mono).

## Tests

```bash
# Run non-GPU tests (parser, audio, mel, tokenizer)
bun test examples/whisper/tests/audio.test.js
bun test examples/whisper/tests/mel.test.js
bun test examples/whisper/tests/tokenizer.test.js
bun test examples/whisper/tests/loader.test.js

# Run GPU tests (requires macOS + Apple Silicon)
bun test examples/whisper/tests/model.test.js
```

## Limitations

- WAV only (no MP3/FLAC/OGG decoding yet)
- No word-level timestamps
- No voice activity detection
- Conv1d uses CPU-side im2col (no dedicated Metal shader)
- Mel spectrogram computed on CPU (could be GPU-accelerated)
