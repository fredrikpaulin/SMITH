// smith/shaders/mel.metal
// STFT magnitude and mel filterbank kernels for mel spectrogram extraction.
// Designed for Whisper-style audio processing but general-purpose.

#include <metal_stdlib>
using namespace metal;

// --- STFT windowing + magnitude ---
// Applies Hann window to a frame of audio, computes FFT magnitude squared.
// This kernel does NOT do FFT itself — it prepares windowed frames as interleaved
// complex (for feeding to fft_radix2) and separately computes |FFT|² from FFT output.

struct STFTParams {
  uint nFft;       // FFT length (after zero-padding to pow2)
  uint winLength;  // window length (e.g. 400)
  uint hopLength;  // hop length (e.g. 160)
  uint numFrames;  // total number of STFT frames
  uint numSamples; // total number of audio samples
};

// Apply Hann window to audio frames and write interleaved complex output.
// One thread per (frame, sample position within fftSize).
// gid.x = position within FFT, gid.y = frame index
kernel void stft_window(
  device const float *samples  [[buffer(0)]],
  device float *windowed       [[buffer(1)]],  // interleaved complex output [batch * fftSize * 2]
  constant STFTParams &p       [[buffer(2)]],
  uint2 gid                    [[thread_position_in_grid]]
) {
  uint pos = gid.x;   // position within FFT
  uint frame = gid.y;  // frame index

  if (pos >= p.nFft || frame >= p.numFrames) return;

  uint outIdx = (frame * p.nFft + pos) * 2; // interleaved complex

  float val = 0.0f;
  if (pos < p.winLength) {
    uint sampleIdx = frame * p.hopLength + pos;
    if (sampleIdx < p.numSamples) {
      // Hann window: 0.5 * (1 - cos(2π * i / winLength))
      float w = 0.5f * (1.0f - cos(2.0f * M_PI_F * float(pos) / float(p.winLength)));
      val = samples[sampleIdx] * w;
    }
  }
  // Zero-padding for positions beyond winLength is implicit (val = 0)

  windowed[outIdx]     = val;  // real
  windowed[outIdx + 1] = 0.0f; // imaginary
}

// Compute magnitude squared from interleaved complex FFT output.
// Only the first nFft/2 + 1 frequency bins (positive frequencies).
// Output: [freqBins, numFrames] layout (freq-major, matching CPU mel.js)
// gid.x = frequency bin, gid.y = frame index

struct MagParams {
  uint nFft;       // FFT length
  uint freqBins;   // nFft / 2 + 1
  uint numFrames;  // number of frames
};

kernel void stft_magnitude(
  device const float *fftOut   [[buffer(0)]],  // interleaved complex [numFrames * nFft * 2]
  device float *mag2           [[buffer(1)]],  // [freqBins, numFrames]
  constant MagParams &p        [[buffer(2)]],
  uint2 gid                    [[thread_position_in_grid]]
) {
  uint f = gid.x;     // frequency bin
  uint frame = gid.y;  // frame index

  if (f >= p.freqBins || frame >= p.numFrames) return;

  uint fftIdx = (frame * p.nFft + f) * 2;
  float re = fftOut[fftIdx];
  float im = fftOut[fftIdx + 1];

  // Output in [freqBins, numFrames] layout (freq-major)
  mag2[f * p.numFrames + frame] = re * re + im * im;
}

// --- Mel filterbank ---
// Matrix multiply: filters [nMels, freqBins] × mag2 [freqBins, numFrames] → mel [nMels, numFrames]
// gid.x = frame, gid.y = mel bin

struct MelParams {
  uint nMels;
  uint freqBins;
  uint numFrames;
};

kernel void mel_filterbank(
  device const float *filters  [[buffer(0)]],  // [nMels, freqBins]
  device const float *mag2     [[buffer(1)]],  // [freqBins, numFrames]
  device float *mel            [[buffer(2)]],  // [nMels, numFrames]
  constant MelParams &p        [[buffer(3)]],
  uint2 gid                    [[thread_position_in_grid]]
) {
  uint frame = gid.x;
  uint m = gid.y;  // mel bin

  if (frame >= p.numFrames || m >= p.nMels) return;

  float sum = 0.0f;
  for (uint f = 0; f < p.freqBins; f++) {
    sum += filters[m * p.freqBins + f] * mag2[f * p.numFrames + frame];
  }

  mel[m * p.numFrames + frame] = sum;
}

// --- Whisper log-mel normalization ---
// 1. log10(max(mel, 1e-10))
// 2. clamp to (maxVal - 8)
// 3. (mel + 4) / 4
// Two-pass: first compute log + find max, then normalize.
// Pass 1: log transform (per-element)

kernel void mel_log(
  device float *mel            [[buffer(0)]],  // in-place [nMels * numFrames]
  constant uint &totalSize     [[buffer(1)]],
  uint gid                     [[thread_position_in_grid]]
) {
  if (gid >= totalSize) return;
  mel[gid] = log10(max(mel[gid], 1e-10f));
}

// Pass 2: clamp and normalize (needs maxVal computed on CPU between passes)
struct NormParams {
  float maxVal;
  uint totalSize;
};

kernel void mel_normalize(
  device float *mel            [[buffer(0)]],
  constant NormParams &p       [[buffer(1)]],
  uint gid                     [[thread_position_in_grid]]
) {
  if (gid >= p.totalSize) return;
  float v = max(mel[gid], p.maxVal - 8.0f);
  mel[gid] = (v + 4.0f) / 4.0f;
}
