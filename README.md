# sonara

**High-performance audio analysis library for Python, written in Rust.**

High-performance audio feature extraction, batch analysis, and built-in perceptual features for playlist generation.

> *sonara* — from Latin *sonare*, "to sound, to resonate"

## Installation

```bash
pip install sonara
```

Requires Python 3.9+. Pre-built wheels available for Linux, macOS (Intel & Apple Silicon), and Windows.

Build from source:

```bash
git clone https://github.com/kkollsga/sonara.git
cd sonara
pip install maturin
maturin develop --release
```

## Quick Start

```python
import sonara
import numpy as np

# Load audio
y, sr = sonara.load("track.mp3", sr=22050)

# STFT
D = sonara.stft(y)
S_db = sonara.amplitude_to_db(np.abs(D))

# Mel spectrogram + MFCC
mel = sonara.melspectrogram(y=y, sr=22050.0)
mfcc = sonara.mfcc(y=y, sr=22050.0, n_mfcc=13)

# Beat tracking
tempo, beats = sonara.beat_track(y=y, sr=22050)

# Chroma
chroma = sonara.chroma_stft(y=y, sr=22050.0)

# Pitch estimation
f0, voiced, prob = sonara.pyin(y, fmin=65.0, fmax=2093.0, sr=22050)
```

## Analysis Pipeline

sonara includes a fused analysis pipeline that extracts all features in a single optimized pass. Three modes control the depth of analysis:

### Modes

| Mode | Features | Time (10s track) | Use case |
|------|----------|-------------------|----------|
| **`compact`** | 11 core features | ~1.2 ms | Fast scanning, metadata |
| **`playlist`** | 22+ features incl. perceptual | ~3.3 ms | Playlist generation, music discovery |
| **`playlist` + `accurate`** | Same features, higher precision | ~12 ms | When accuracy matters more than speed |

### Compact mode (default)

Core signal features, always computed:

```python
r = sonara.analyze_file("track.mp3", mode="compact")

r['bpm']                    # Tempo (BPM)
r['beats']                  # Beat frame positions
r['onset_frames']           # Onset positions
r['onset_density']          # Onsets per second
r['rms_mean']               # Average loudness (RMS)
r['rms_max']                # Peak loudness (RMS)
r['loudness_lufs']          # Integrated loudness (LUFS, ITU-R BS.1770-4)
r['dynamic_range_db']       # Loudness range (p95 - p5, dB)
r['spectral_centroid_mean'] # Brightness (Hz)
r['zero_crossing_rate']     # Percussiveness proxy
r['duration_sec']           # Track length
```

### Playlist mode

Everything for playlist generation: spectral features, MFCCs (timbre fingerprint), chroma (harmony), plus perceptual features:

```python
r = sonara.analyze_file("track.mp3", mode="playlist")

# Perceptual features (0.0 - 1.0)
r['energy']           # Perceived intensity (loudness + brightness + activity)
r['danceability']     # Beat regularity + tempo sweet spot + rhythm
r['valence']          # Mood (0 = sad/dark, 1 = happy/bright)
r['acousticness']     # Acoustic vs electronic character

# Musical key
r['key']              # e.g. "C major", "A minor"
r['key_confidence']   # How confident the key detection is (0.0 - 1.0)

# Spectral features
r['spectral_bandwidth_mean']   # Frequency spread
r['spectral_rolloff_mean']     # Frequency below which 85% of energy sits
r['spectral_flatness_mean']    # Tonal (0) vs noise-like (1)
r['spectral_contrast_mean']    # Peak-valley ratio per band (7 values)
r['mfcc_mean']                 # Timbre fingerprint (13 coefficients)
r['chroma_mean']               # Pitch class distribution (12 values)
```

### Accurate flag

Trades speed for precision on select features:

```python
r = sonara.analyze_file("track.mp3", mode="playlist", accurate=True)
```

| Feature | Default | Accurate |
|---------|---------|----------|
| Chroma | Mel-band approximation (fast, +/-1 semitone) | Proper chroma filterbank (exact) |
| Spectral contrast | Mel sub-bands | Log-spaced frequency bands on magnitude spectrum |
| Danceability | Beat heuristic | Detrended Fluctuation Analysis (Streich & Herrera 2005) |

### Custom feature selection

Cherry-pick specific features regardless of mode:

```python
r = sonara.analyze_file("track.mp3", features=["bpm", "energy", "key", "loudness_lufs"])
```

Valid feature names: `bpm`, `beats`, `onsets`, `rms`, `dynamic_range`, `centroid`, `zcr`, `onset_density`, `bandwidth`, `rolloff`, `flatness`, `contrast`, `mfcc`, `chroma`, `energy`, `danceability`, `key`, `valence`, `acousticness`

### Batch analysis

Analyze entire music libraries in parallel using all CPU cores:

```python
import sonara
from pathlib import Path

files = [str(p) for p in Path("~/Music").rglob("*.mp3")]
results = sonara.analyze_batch(files, mode="playlist")

for r in results:
    print(f"{r['bpm']:5.0f} BPM | {r['energy']:.2f} energy | "
          f"{r['key']:>10} | {r['loudness_lufs']:6.1f} LUFS | "
          f"{r['valence']:.2f} valence")
```

## Display

```python
import sonara
import sonara.display as display
import matplotlib.pyplot as plt

y, sr = sonara.load("track.mp3", sr=22050)
mel = sonara.melspectrogram(y=y, sr=22050.0)
mel_db = sonara.power_to_db(mel)

fig, ax = plt.subplots()
display.specshow(mel_db, x_axis='time', y_axis='mel', sr=22050, ax=ax)
plt.show()
```

## Performance

All arithmetic uses f32 precision (matching native decoder format), with a parallelized fused FFT pipeline and fast-path 2:1 decimation for the common 44100 Hz to 22050 Hz resampling case.

### Benchmarks

| Feature | Speedup vs Python |
|---------|-------------------|
| Mel spectrogram | ~3x |
| MFCC | ~3x |
| Beat tracking | ~4x |
| Onset detection | ~3x |
| Spectral centroid | ~3x |
| Cold start (first call) | ~20-30x |
| **Batch analysis (parallel)** | **~5x** |

### Analysis pipeline benchmarks (10s signal, Apple Silicon)

| Mode | Time | Features |
|------|------|----------|
| `compact` | 1.2 ms | 11 core features |
| `playlist` | 3.3 ms | 22+ features incl. perceptual |
| `playlist` + `accurate` | 12.4 ms | Same, with accurate chroma/DFA |

### Key optimizations

- **f32 precision** — halves memory bandwidth vs f64, matches Symphonia's native decode format (zero-cost conversion)
- **Fused single-pass pipeline** — one FFT per frame simultaneously produces mel, centroid, RMS, bandwidth, rolloff, flatness
- **Parallel FFT frames** — rayon parallelism across frames (for signals > 32 frames)
- **Sparse mel projection** — triangular mel filters are ~97% zeros; only non-zero weights multiplied
- **Fast 2:1 decimation** — half-band FIR filter for 44100-to-22050 Hz instead of full sinc resampling
- **Thread-local FFT cache** — plan and scratch buffer reuse with RefCell (no mutex contention)
- **Mel filterbank caching** — reused across calls in batch processing
- **K-weighted LUFS** — two-biquad IIR filter, single-pass (~0.05ms per second of audio)

## API Compatibility

sonara provides 92+ audio analysis functions:

**Core Audio:** `load`, `stft`, `istft`, `resample`, `to_mono`, `tone`, `chirp`, `clicks`

**Features:** `melspectrogram`, `mfcc`, `chroma_stft`, `tonnetz`, `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_contrast`, `rms`, `zero_crossing_rate`

**Rhythm:** `beat_track`, `onset_detect`, `onset_strength`, `tempo`

**Pitch:** `yin`, `pyin`, `piptrack`, `estimate_tuning`

**Transforms:** `cqt`, `vqt`, `icqt`, `hybrid_cqt`, `pseudo_cqt`, `griffinlim`

**Conversions:** `hz_to_mel`, `mel_to_hz`, `hz_to_midi`, `midi_to_hz`, `note_to_hz`, `note_to_midi`, `fft_frequencies`, `mel_frequencies`, `cqt_frequencies`, and 30+ more

**Effects:** `time_stretch`, `pitch_shift`, `trim`, `split`, `preemphasis`, `deemphasis`

**Notation:** `key_to_notes`, `key_to_degrees`, `mela_to_svara`, `thaat_to_degrees`, `hz_to_svara_h`, `hz_to_svara_c`

## Architecture

sonara is a two-crate Rust workspace:

- **`sonara`** — Pure Rust core library (~10,000 LOC)
- **`sonara-python`** — PyO3 bindings (~1,000 LOC)

```text
sonara/src/
  analyze.rs      — Fused analysis pipeline (compact/playlist/full modes)
  perceptual.rs   — LUFS, energy, danceability, key detection, valence, acousticness
  beat.rs         — Beat tracking (Ellis 2007 DP algorithm)
  onset.rs        — Onset detection (spectral flux + peak picking)
  core/
    audio.rs      — Audio I/O, resampling, fast 2:1 decimation
    spectrum.rs   — STFT, power spectrogram, dB conversions
    fft.rs        — FFT with thread-local plan caching
    pitch.rs      — YIN / pYIN pitch estimation
    convert.rs    — Hz/mel/MIDI/note conversions, frequency weighting
  feature/
    spectral.rs   — Mel, MFCC, chroma, centroid, bandwidth, rolloff, flatness, contrast
  dsp/
    windows.rs    — Window functions (Hann, Hamming, Blackman, Kaiser)
    iir.rs        — IIR filters (lfilter, sosfiltfilt)
  filters.rs      — Mel/chroma filterbanks
```

## License

[MIT](LICENSE)
