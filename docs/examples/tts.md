# Qwen3-TTS Text-to-Speech

Zero-dependency TTS using Qwen3-TTS-12Hz-1.7B-Base on Apple Silicon via Smith's Metal compute pipeline.

## Prerequisites

Download the model (~4.5GB):

```sh
# Requires huggingface-cli
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./models/Qwen3-TTS-12Hz-1.7B-Base
```

Build Smith (compiles Metal shaders + native bridge):

```sh
./build.sh
```

## Usage

Interactive TUI — type text, hear speech:

```sh
bun examples/tts/cli.js ./models/Qwen3-TTS-12Hz-1.7B-Base
```

Commands inside the TUI:

- `/config` — show current generation parameters
- `/temp <n>` — set temperature (default 0.9)
- `/topk <n>` — set top-k (default 50)
- `/help` — list commands

## Architecture

Three-stage pipeline, all running on-device:

1. **Talker** (28-layer transformer, 2048 dim) — text tokens → group-0 codec codes via autoregressive decoding with KV cache
2. **Code Predictor** (5-layer transformer, 1024 dim) — group-0 code + talker hidden state → groups 1-15 codes
3. **Speech Decoder** — 16-group codec codes → 24kHz PCM waveform via dequantization, transformer, upsampling, and vocoder

## Files

- `cli.js` — TUI entry point
- `model.js` — safetensors weight loader, maps Python weight names to JS model structs
- `tokenizer.js` — byte-level BPE tokenizer (GPT-2 style)
- `talker.js` — Talker forward pass with KV cache, QK norm, SwiGLU FFN
- `predictor.js` — Code Predictor with per-group LM heads and codec embeddings
- `decoder.js` — SplitRVQ dequantization, transformer, ConvNeXt upsampling, SnakeBeta vocoder
- `audio.js` — WAV encoding and macOS `afplay` playback

## Generation pipeline

The talker and predictor run interleaved, not sequentially. At each decode step:

1. **Predictor** receives the current talker hidden state (projected 2048→1024) as position 0, and the group-0 code embedding (also projected) as position 1. It autoregressively generates groups 1-15 at positions 2-16.
2. All 16 group embeddings are summed: group-0 from the talker's codec embedding, groups 1-15 from the predictor's codec embeddings. The `tts_pad_embed` vector is added to the sum.
3. **Talker** decodes one step using this summed embedding as input, producing the next group-0 code and a new hidden state.

This feedback loop is critical — without it, the talker generates codes with no acoustic context.

The decoder converts 16-group codes to audio through: RVQ dequantization → pre-conv → 8-layer transformer → 2× ConvNeXt upsampling → 4 vocoder blocks (8×5×4×3 upsampling) with SnakeBeta activations and dilated residual convolutions → final 1-channel output at 24kHz.

## New Smith ops

This example required two new ops added to Smith core:

- `src/ops/conv1d_transpose.js` + `shaders/conv1d_transpose.metal` — transposed (fractionally-strided) 1D convolution for upsampling
- `src/ops/snake.js` + `shaders/activation.metal` (snake_forward kernel) — Snake activation function
