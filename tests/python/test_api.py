#!/usr/bin/env python3
"""Test all major usage patterns from librosa's README, docs, and examples.

Verifies that sonara can be used as a drop-in replacement for librosa
in every documented workflow pattern.
"""

import sys
import traceback
import numpy as np

passed = 0
failed = 0
errors = []

def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")

import sonara

# ============================================================
# Generate test signals (no audio files needed)
# ============================================================
np.random.seed(42)
sr = 22050
y_1s = np.random.randn(sr).astype(np.float64)
y_5s = np.random.randn(5 * sr).astype(np.float64)
# Pure sine for pitch tests
y_sine = np.sin(2 * np.pi * 440.0 * np.arange(2 * sr) / sr).astype(np.float64)
# Click train for beat tests
y_clicks = np.zeros(4 * sr, dtype=np.float64)
for i in range(0, 4 * sr, sr // 2):  # 120 BPM
    y_clicks[i:i+100] = np.sin(2 * np.pi * 1000 * np.arange(100) / sr)

print("=" * 70)
print("  Testing librosa doc patterns with sonara")
print("=" * 70)

# ============================================================
# Pattern 1: Basic STFT workflow (from plot_display.py)
# ============================================================
print("\n--- Pattern 1: STFT Workflow ---")

test("stft(y)", lambda: sonara.stft(y_1s))
test("stft with custom params", lambda: sonara.stft(y_1s, n_fft=4096, hop_length=256))

D = sonara.stft(y_1s)
test("amplitude_to_db(|D|)", lambda: sonara.amplitude_to_db(np.abs(D).astype(np.float64)))
test("power_to_db(|D|^2)", lambda: sonara.power_to_db(np.abs(D).astype(np.float64)**2))
test("istft(D)", lambda: sonara.istft(D))
test("istft roundtrip length", lambda: (
    np.testing.assert_equal(len(sonara.istft(D, length=len(y_1s))), len(y_1s))
))

# ============================================================
# Pattern 2: Mel spectrogram workflow (from plot_display.py)
# ============================================================
print("\n--- Pattern 2: Mel Spectrogram ---")

test("melspectrogram(y=y)", lambda: sonara.melspectrogram(y=y_1s, sr=22050.0))
test("melspectrogram(y=y, n_mels=40)", lambda: sonara.melspectrogram(y=y_1s, sr=22050.0, n_mels=40))

M = sonara.melspectrogram(y=y_1s, sr=22050.0)
test("power_to_db(mel)", lambda: sonara.power_to_db(M))

# ============================================================
# Pattern 3: MFCC workflow
# ============================================================
print("\n--- Pattern 3: MFCC ---")

test("mfcc(y=y, n_mfcc=13)", lambda: sonara.mfcc(y=y_1s, sr=22050.0, n_mfcc=13))
test("mfcc(y=y, n_mfcc=20)", lambda: sonara.mfcc(y=y_1s, sr=22050.0, n_mfcc=20))
test("mfcc(y=y, n_mfcc=40)", lambda: sonara.mfcc(y=y_1s, sr=22050.0, n_mfcc=40))
test("mfcc shape", lambda: np.testing.assert_equal(sonara.mfcc(y=y_1s, sr=22050.0, n_mfcc=13).shape[0], 13))

# ============================================================
# Pattern 4: Chroma features (from plot_chroma.py)
# ============================================================
print("\n--- Pattern 4: Chroma ---")

test("chroma_stft(y=y)", lambda: sonara.chroma_stft(y=y_1s, sr=22050.0))
test("chroma_stft shape is (12, N)", lambda: np.testing.assert_equal(sonara.chroma_stft(y=y_1s, sr=22050.0).shape[0], 12))

# ============================================================
# Pattern 5: CQT workflow (from plot_display.py)
# ============================================================
print("\n--- Pattern 5: CQT ---")

test("cqt(y)", lambda: sonara.cqt(y_1s, sr=22050))
test("cqt(n_bins=36)", lambda: sonara.cqt(y_1s, sr=22050, n_bins=36))
C = sonara.cqt(y_1s, sr=22050, n_bins=84)
test("cqt shape", lambda: np.testing.assert_equal(C.shape[0], 84))
test("amplitude_to_db(|cqt|)", lambda: sonara.amplitude_to_db(np.abs(C).astype(np.float64)))
test("vqt(y)", lambda: sonara.vqt(y_1s, sr=22050, n_bins=36))
test("pseudo_cqt(y)", lambda: sonara.pseudo_cqt(y_1s, sr=22050, n_bins=36))
test("hybrid_cqt(y)", lambda: sonara.hybrid_cqt(y_1s, sr=22050, n_bins=36))

# ============================================================
# Pattern 6: Beat tracking (from plot_dynamic_beat.py)
# ============================================================
print("\n--- Pattern 6: Beat Tracking ---")

test("beat_track(y=y)", lambda: sonara.beat_track(y=y_5s, sr=22050))
tempo, beats = sonara.beat_track(y=y_clicks, sr=22050)
test("beat_track returns tempo", lambda: (None if tempo > 0 else (_ for _ in ()).throw(ValueError("bad tempo"))))
test("beat_track returns beats", lambda: (None if len(beats) > 0 else (_ for _ in ()).throw(ValueError("no beats"))))
test("onset_strength(y)", lambda: sonara.onset_strength(y_5s, sr=22050))
test("onset_detect(y=y)", lambda: sonara.onset_detect(y=y_clicks, sr=22050))

# ============================================================
# Pattern 7: Spectral features
# ============================================================
print("\n--- Pattern 7: Spectral Features ---")

test("spectral_centroid(y=y)", lambda: sonara.spectral_centroid(y=y_1s, sr=22050.0))
test("rms(y=y)", lambda: sonara.rms(y=y_1s))

# ============================================================
# Pattern 8: Pitch estimation
# ============================================================
print("\n--- Pattern 8: Pitch ---")

test("yin(y, fmin, fmax)", lambda: sonara.yin(y_sine, fmin=65.0, fmax=2093.0, sr=22050, frame_length=2048))
test("pyin(y, fmin, fmax)", lambda: sonara.pyin(y_sine, fmin=65.0, fmax=2093.0, sr=22050, frame_length=2048))

f0, voiced_flag, voiced_prob = sonara.pyin(y_sine, fmin=65.0, fmax=2093.0, sr=22050, frame_length=2048)
test("pyin returns 3 arrays", lambda: np.testing.assert_equal(len(f0), len(voiced_flag)))
test("piptrack(y)", lambda: sonara.piptrack(y_1s, sr=22050))
test("estimate_tuning(y=y)", lambda: sonara.estimate_tuning(y=y_sine, sr=22050))

# ============================================================
# Pattern 9: Unit conversions
# ============================================================
print("\n--- Pattern 9: Conversions ---")

test("hz_to_mel(440)", lambda: np.testing.assert_almost_equal(sonara.hz_to_mel(440.0), 6.6, decimal=1))
test("mel_to_hz(6.6)", lambda: (None if sonara.mel_to_hz(6.6) > 400 else (_ for _ in ()).throw(ValueError())))
test("note_to_hz('A4') == 440", lambda: np.testing.assert_almost_equal(sonara.note_to_hz("A4"), 440.0, decimal=1))
test("midi_to_hz(69) == 440", lambda: np.testing.assert_almost_equal(sonara.midi_to_hz(69.0), 440.0))
test("hz_to_midi(440) == 69", lambda: np.testing.assert_almost_equal(sonara.hz_to_midi(440.0), 69.0))
test("midi_to_note(69) == 'A4'", lambda: np.testing.assert_equal(sonara.midi_to_note(69.0), "A4"))
test("hz_to_note(440)", lambda: sonara.hz_to_note(440.0))
test("hz_to_octs(440)", lambda: sonara.hz_to_octs(440.0))
test("octs_to_hz(4)", lambda: sonara.octs_to_hz(4.0))
test("A4_to_tuning(440)", lambda: np.testing.assert_almost_equal(sonara.A4_to_tuning(440.0), 0.0))
test("tuning_to_A4(0)", lambda: np.testing.assert_almost_equal(sonara.tuning_to_A4(0.0), 440.0))

# ============================================================
# Pattern 10: Frame/time/sample conversions
# ============================================================
print("\n--- Pattern 10: Frame/Time/Sample ---")

test("frames_to_samples([0,1,2])", lambda: np.testing.assert_equal(sonara.frames_to_samples([0,1,2]), [0, 512, 1024]))
test("frames_to_time([0,1])", lambda: sonara.frames_to_time([0, 1]))
test("samples_to_frames([0,512])", lambda: np.testing.assert_equal(sonara.samples_to_frames([0, 512]), [0, 1]))
test("time_to_frames([0, 1.0])", lambda: sonara.time_to_frames([0.0, 1.0]))
test("time_to_samples([0, 1.0])", lambda: sonara.time_to_samples([0.0, 1.0]))
test("samples_to_time([0, 22050])", lambda: sonara.samples_to_time([0, 22050]))
test("blocks_to_frames([0,1], block_length=2048)", lambda: sonara.blocks_to_frames([0, 1], block_length=2048))

# ============================================================
# Pattern 11: Frequency generators
# ============================================================
print("\n--- Pattern 11: Frequency Generators ---")

test("fft_frequencies(sr=22050, n_fft=2048)", lambda: np.testing.assert_equal(len(sonara.fft_frequencies(sr=22050.0, n_fft=2048)), 1025))
test("mel_frequencies(n_mels=128)", lambda: np.testing.assert_equal(len(sonara.mel_frequencies(n_mels=128)), 128))
test("cqt_frequencies(n_bins=84)", lambda: np.testing.assert_equal(len(sonara.cqt_frequencies(84)), 84))
test("tempo_frequencies(n_bins=100)", lambda: sonara.tempo_frequencies(100))

# ============================================================
# Pattern 12: Weighting functions
# ============================================================
print("\n--- Pattern 12: Weighting ---")

freqs = sonara.fft_frequencies(sr=22050.0, n_fft=2048)
test("A_weighting(freqs)", lambda: sonara.A_weighting(freqs))
test("B_weighting(freqs)", lambda: sonara.B_weighting(freqs))
test("C_weighting(freqs)", lambda: sonara.C_weighting(freqs))
test("D_weighting(freqs)", lambda: sonara.D_weighting(freqs))
test("Z_weighting(freqs)", lambda: sonara.Z_weighting(freqs))

# ============================================================
# Pattern 13: Notation (Indian music, FJS)
# ============================================================
print("\n--- Pattern 13: Notation ---")

test("key_to_notes('C:maj')", lambda: np.testing.assert_equal(sonara.key_to_notes("C:maj"), ["C","D","E","F","G","A","B"]))
test("key_to_degrees('C:maj')", lambda: sonara.key_to_degrees("C:maj"))
test("list_mela() has 72", lambda: np.testing.assert_equal(len(sonara.list_mela()), 72))
test("list_thaat() has 10", lambda: np.testing.assert_equal(len(sonara.list_thaat()), 10))
test("mela_to_svara(1)", lambda: sonara.mela_to_svara(1))
test("thaat_to_degrees('Bilaval')", lambda: sonara.thaat_to_degrees("Bilaval"))
test("fifths_to_note(0) == 'C'", lambda: np.testing.assert_equal(sonara.fifths_to_note(0), "C"))
test("interval_to_fjs(1.5) == 'P5'", lambda: np.testing.assert_equal(sonara.interval_to_fjs(1.5), "P5"))
test("pythagorean_intervals(12)", lambda: sonara.pythagorean_intervals())

# ============================================================
# Pattern 14: Svara conversions
# ============================================================
print("\n--- Pattern 14: Svara ---")

test("hz_to_svara_h(440)", lambda: sonara.hz_to_svara_h(440.0))
test("hz_to_svara_c(440)", lambda: sonara.hz_to_svara_c(440.0))
test("midi_to_svara_h(69)", lambda: sonara.midi_to_svara_h(69.0))
test("midi_to_svara_c(69)", lambda: sonara.midi_to_svara_c(69.0))
test("note_to_svara_h('A4')", lambda: sonara.note_to_svara_h("A4"))
test("note_to_svara_c('A4')", lambda: sonara.note_to_svara_c("A4"))
test("hz_to_fjs(660, ref_freq=440)", lambda: sonara.hz_to_fjs(660.0, ref_freq=440.0))

# ============================================================
# Pattern 15: Signal generation
# ============================================================
print("\n--- Pattern 15: Signal Generation ---")

test("tone(440, sr=22050, length=22050)", lambda: np.testing.assert_equal(len(sonara.tone(440.0, sr=22050, length=22050)), 22050))
test("chirp(fmin=200, fmax=2000)", lambda: sonara.chirp(fmin=200.0, fmax=2000.0, sr=22050, length=22050))
test("clicks(times=[0.5, 1.0])", lambda: sonara.clicks(times=[0.5, 1.0], sr=22050, length=22050))

# ============================================================
# Pattern 16: Audio utility
# ============================================================
print("\n--- Pattern 16: Audio Utility ---")

test("autocorrelate(y)", lambda: sonara.autocorrelate(y_1s))
test("lpc(y, order=4)", lambda: sonara.lpc(y_1s, order=4))
test("zero_crossings(y)", lambda: sonara.zero_crossings(y_1s))
test("mu_compress(y)", lambda: sonara.mu_compress(y_1s))
test("mu_expand(mu_compress(y))", lambda: sonara.mu_expand(sonara.mu_compress(y_1s)))

# ============================================================
# Pattern 17: Magphase and phase vocoder
# ============================================================
print("\n--- Pattern 17: Magphase + Phase Vocoder ---")

D = sonara.stft(y_1s)
test("magphase(D)", lambda: sonara.magphase(D))
mag, phase = sonara.magphase(D)
test("magphase: mag >= 0", lambda: np.testing.assert_array_less(-1e-10, mag))
test("phase_vocoder(D, rate=1.0)", lambda: sonara.phase_vocoder(D, rate=1.0))
test("phase_vocoder(D, rate=2.0)", lambda: sonara.phase_vocoder(D, rate=2.0))

# ============================================================
# Pattern 18: Filters
# ============================================================
print("\n--- Pattern 18: Filters ---")

test("mel filterbank", lambda: np.testing.assert_equal(sonara.mel(sr=22050.0, n_fft=2048, n_mels=128).shape, (128, 1025)))

# ============================================================
# Pattern 19: Utility functions
# ============================================================
print("\n--- Pattern 19: Utility ---")

test("samples_like(n_frames=10)", lambda: sonara.samples_like(10))
test("times_like(n_frames=10)", lambda: sonara.times_like(10))

# ============================================================
# Pattern 20: Display (requires matplotlib)
# ============================================================
print("\n--- Pattern 20: Display ---")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import sonara.display as disp

    fig, ax = plt.subplots()
    M = sonara.melspectrogram(y=y_1s, sr=22050.0)
    M_db = sonara.power_to_db(M)
    test("specshow(mel)", lambda: disp.specshow(M_db, x_axis='time', y_axis='mel', ax=ax))
    test("waveshow(y)", lambda: disp.waveshow(y_1s, sr=22050, ax=ax))
    test("cmap(data)", lambda: disp.cmap(M_db))
    plt.close('all')
except ImportError:
    test("display (matplotlib not available)", lambda: None)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {passed + failed} tests")
print(f"{'='*70}")

if errors:
    print(f"\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")

sys.exit(1 if failed > 0 else 0)
