# canora

**High-performance audio analysis library for Python, written in Rust.**

A drop-in replacement for [librosa](https://librosa.org/) with significantly faster feature extraction and batch analysis.

> *canora* (Latin): melodious, tuneful — from *canor*, "song"

## Installation

```bash
pip install canora
```

Requires Python 3.9+. Pre-built wheels available for Linux, macOS (Intel & Apple Silicon), and Windows.

Build from source:

```bash
git clone https://github.com/kkollsga/canora.git
cd canora
pip install maturin
maturin develop --release
```

## Quick Start

```python
import canora
import numpy as np

# Load audio
y, sr = canora.load("track.mp3", sr=22050)

# STFT
D = canora.stft(y)
S_db = canora.amplitude_to_db(np.abs(D))

# Mel spectrogram + MFCC
mel = canora.melspectrogram(y=y, sr=22050.0)
mfcc = canora.mfcc(y=y, sr=22050.0, n_mfcc=13)

# Beat tracking
tempo, beats = canora.beat_track(y=y, sr=22050)

# Chroma
chroma = canora.chroma_stft(y=y, sr=22050.0)

# Pitch estimation
f0, voiced, prob = canora.pyin(y, fmin=65.0, fmax=2093.0, sr=22050)
```

## Batch Analysis

Analyze multiple files in parallel using all CPU cores:

```python
import canora

files = ["track1.mp3", "track2.mp3", "track3.mp3"]
results = canora.analyze_batch(files, sr=22050)

for r in results:
    print(f"BPM: {r['bpm']:.0f}, Duration: {r['duration_sec']:.0f}s, "
          f"Centroid: {r['spectral_centroid_mean']:.0f}Hz")
```

Single file with all features:

```python
result = canora.analyze_file("track.mp3", sr=22050)
# Returns dict with: bpm, beats, onset_frames, rms_mean, rms_max,
# dynamic_range_db, spectral_centroid_mean, zero_crossing_rate, onset_density
```

## Display

```python
import canora
import canora.display as display
import matplotlib.pyplot as plt

y, sr = canora.load("track.mp3", sr=22050)
mel = canora.melspectrogram(y=y, sr=22050.0)
mel_db = canora.power_to_db(mel)

fig, ax = plt.subplots()
display.specshow(mel_db, x_axis='time', y_axis='mel', sr=22050, ax=ax)
plt.show()
```

## Performance

Benchmarked against librosa on real-world MP3 albums:

| Feature | Speedup vs librosa |
|---------|-------------------|
| Mel spectrogram | ~3x |
| MFCC | ~3x |
| Beat tracking | ~4x |
| Onset detection | ~3x |
| Spectral centroid | ~3x |
| Cold start (first call) | ~20-30x |
| **Batch analysis (parallel)** | **~5x** |

Performance varies by platform and signal length. Gains come from compiled Rust, fused FFT pipeline, sparse mel projection, and multi-core parallelism.

## API Compatibility

canora implements 92 of librosa's top-level functions with matching signatures:

**Core Audio:** `load`, `stft`, `istft`, `resample`, `to_mono`, `tone`, `chirp`, `clicks`

**Features:** `melspectrogram`, `mfcc`, `chroma_stft`, `tonnetz`, `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_contrast`, `rms`, `zero_crossing_rate`

**Rhythm:** `beat_track`, `onset_detect`, `onset_strength`, `tempo`

**Pitch:** `yin`, `pyin`, `piptrack`, `estimate_tuning`

**Transforms:** `cqt`, `vqt`, `icqt`, `hybrid_cqt`, `pseudo_cqt`, `griffinlim`

**Conversions:** `hz_to_mel`, `mel_to_hz`, `hz_to_midi`, `midi_to_hz`, `note_to_hz`, `note_to_midi`, `fft_frequencies`, `mel_frequencies`, `cqt_frequencies`, and 30+ more

**Effects:** `time_stretch`, `pitch_shift`, `trim`, `split`, `preemphasis`, `deemphasis`

**Notation:** `key_to_notes`, `key_to_degrees`, `mela_to_svara`, `thaat_to_degrees`, `hz_to_svara_h`, `hz_to_svara_c`

## Architecture

canora is a two-crate Rust workspace:

- **`canora`** — Pure Rust core library (~10,000 LOC, 214 functions)
- **`canora-python`** — PyO3 bindings (~1,000 LOC)

Key optimizations:

- **Fused sparse mel projection** — triangular mel filters are ~97% zeros; only non-zero weights are multiplied
- **Single-pass FFT pipeline** — mel spectrogram, spectral centroid, and RMS extracted from one FFT per frame
- **BLAS acceleration** — links to platform BLAS (Apple Accelerate, OpenBLAS) for matrix operations
- **Rayon parallelism** — multi-core batch file processing and parallel STFT
- **Zero-copy numpy interop** — PyO3 + rust-numpy passes arrays without copying

## License

[MIT](LICENSE)
