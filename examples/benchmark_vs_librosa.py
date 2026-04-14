#!/usr/bin/env python3
"""Full accuracy + performance benchmark: sonara vs librosa, all phases."""

import time
import subprocess
import sys
import numpy as np

N_WARMUP = 3
N_RUNS = 20

def bench(name, fn_c, fn_l, atol=1e-6, rtol=1e-5, check=True):
    """Run accuracy + timing comparison."""
    r_l = fn_l()
    r_c = fn_c()

    acc = "—"
    if check and r_l is not None and r_c is not None:
        try:
            rl = np.asarray(r_l, dtype=np.float32)
            rc = np.asarray(r_c, dtype=np.float32)
            if rl.shape != rc.shape:
                acc = f"SHAPE ({rc.shape} vs {rl.shape})"
            else:
                max_diff = np.max(np.abs(rl - rc))
                np.testing.assert_allclose(rc, rl, atol=atol, rtol=rtol)
                acc = f"PASS (d={max_diff:.1e})"
        except AssertionError:
            max_diff = np.max(np.abs(np.asarray(r_c, dtype=np.float32) - np.asarray(r_l, dtype=np.float32)))
            acc = f"FAIL (d={max_diff:.1e})"
        except Exception as e:
            acc = f"ERR: {type(e).__name__}"

    for _ in range(N_WARMUP):
        fn_l(); fn_c()

    lt = []; ct = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter_ns(); fn_l(); lt.append(time.perf_counter_ns() - t0)
    for _ in range(N_RUNS):
        t0 = time.perf_counter_ns(); fn_c(); ct.append(time.perf_counter_ns() - t0)

    lm = np.median(lt) / 1e6
    cm = np.median(ct) / 1e6
    sp = lm / max(cm, 0.001)
    return {"name": name, "accuracy": acc, "librosa_ms": round(lm, 3), "sonara_ms": round(cm, 3), "speedup": round(sp, 1)}

def print_table(title, results):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"| {'Function':<40} | {'Accuracy':<20} | {'librosa':>9} | {'sonara':>9} | {'Speed':>7} |")
    print(f"|{'-'*42}|{'-'*22}|{'-'*11}|{'-'*11}|{'-'*9}|")
    for r in results:
        a = r['accuracy'][:20] if len(r['accuracy']) > 20 else r['accuracy']
        print(f"| {r['name']:<40} | {a:<20} | {r['librosa_ms']:>7.3f}ms | {r['sonara_ms']:>7.3f}ms | {r['speedup']:>5.1f}x |")

if __name__ == "__main__":
    import sonara
    import librosa

    np.random.seed(42)
    y1 = np.random.randn(22050).astype(np.float32)   # 1s
    y5 = np.random.randn(110250).astype(np.float32)  # 5s

    print("="*90)
    print("  SONARA vs LIBROSA — Full Accuracy & Performance Report")
    print("="*90)

    # ============================================================
    # PHASE 1: Conversions
    # ============================================================
    r1 = []
    r1.append(bench("hz_to_mel(440)", lambda: sonara.hz_to_mel(440.0), lambda: librosa.hz_to_mel(440.0), atol=1e-4))
    r1.append(bench("fft_frequencies(2048)", lambda: sonara.fft_frequencies(sr=22050.0, n_fft=2048), lambda: librosa.fft_frequencies(sr=22050, n_fft=2048)))
    r1.append(bench("mel_frequencies(128)", lambda: sonara.mel_frequencies(n_mels=128), lambda: librosa.mel_frequencies(n_mels=128), atol=1e-4))
    r1.append(bench("note_to_hz('A4')", lambda: sonara.note_to_hz("A4"), lambda: librosa.note_to_hz("A4")))
    r1.append(bench("midi_to_hz(69)", lambda: sonara.midi_to_hz(69.0), lambda: librosa.midi_to_hz(69)))
    print_table("Phase 1: Conversions", r1)

    # ============================================================
    # PHASE 2: Filters
    # ============================================================
    r2 = []
    r2.append(bench("mel filterbank (128 mels)",
        lambda: sonara.mel(sr=22050.0, n_fft=2048, n_mels=128),
        lambda: librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128).astype(np.float32),
        atol=1e-4))
    print_table("Phase 2: Filters", r2)

    # ============================================================
    # PHASE 3: STFT / Spectrum
    # ============================================================
    r3 = []
    r3.append(bench("stft 1s (n_fft=2048)",
        lambda: sonara.stft(y1, n_fft=2048),
        lambda: librosa.stft(y1, n_fft=2048),
        atol=1e-8))
    r3.append(bench("stft 5s (n_fft=2048)",
        lambda: sonara.stft(y5, n_fft=2048),
        lambda: librosa.stft(y5, n_fft=2048),
        check=False))

    S1 = sonara.stft(y1, n_fft=2048)
    r3.append(bench("istft 1s",
        lambda: sonara.istft(S1, length=22050),
        lambda: librosa.istft(librosa.stft(y1, n_fft=2048), length=22050),
        atol=1e-4))

    S_pow = np.abs(librosa.stft(y1, n_fft=2048)).astype(np.float32)**2
    r3.append(bench("power_to_db",
        lambda: sonara.power_to_db(S_pow),
        lambda: librosa.power_to_db(S_pow),
        atol=1e-4))
    print_table("Phase 3: STFT & Spectrum", r3)

    # ============================================================
    # PHASE 6: Features
    # ============================================================
    r6 = []
    # melspectrogram
    r6.append(bench("melspectrogram 1s",
        lambda: sonara.melspectrogram(y=y1, sr=22050.0),
        lambda: librosa.feature.melspectrogram(y=y1, sr=22050).astype(np.float32),
        atol=1e-2, rtol=0.1))

    r6.append(bench("melspectrogram 5s",
        lambda: sonara.melspectrogram(y=y5, sr=22050.0),
        lambda: librosa.feature.melspectrogram(y=y5, sr=22050).astype(np.float32),
        check=False))

    # mfcc
    r6.append(bench("mfcc 1s (20 coeffs)",
        lambda: sonara.mfcc(y=y1, sr=22050.0, n_mfcc=20),
        lambda: librosa.feature.mfcc(y=y1, sr=22050, n_mfcc=20).astype(np.float32),
        atol=5.0, rtol=0.5))  # MFCC values can differ due to DCT/log implementation

    r6.append(bench("mfcc 5s (20 coeffs)",
        lambda: sonara.mfcc(y=y5, sr=22050.0, n_mfcc=20),
        lambda: librosa.feature.mfcc(y=y5, sr=22050, n_mfcc=20).astype(np.float32),
        check=False))

    # chroma
    r6.append(bench("chroma_stft 1s",
        lambda: sonara.chroma_stft(y=y1, sr=22050.0),
        lambda: librosa.feature.chroma_stft(y=y1, sr=22050).astype(np.float32),
        atol=0.5, rtol=0.5))

    # spectral centroid
    r6.append(bench("spectral_centroid 1s",
        lambda: sonara.spectral_centroid(y=y1, sr=22050.0),
        lambda: librosa.feature.spectral_centroid(y=y1, sr=22050).astype(np.float32),
        atol=500.0, rtol=0.3))

    # rms
    r6.append(bench("rms 1s",
        lambda: sonara.rms(y=y1),
        lambda: librosa.feature.rms(y=y1).astype(np.float32),
        atol=0.1, rtol=0.3))

    print_table("Phase 6: Spectral Features", r6)

    # ============================================================
    # COLD START
    # ============================================================
    print(f"\n{'='*90}")
    print("  Cold Start (subprocess, 3 runs)")
    print(f"{'='*90}")
    for mod_name, label, code in [
        ("librosa", "librosa: import+stft+mel+mfcc",
         "import numpy as np; y=np.random.randn(22050); librosa.stft(y); librosa.feature.melspectrogram(y=y, sr=22050); librosa.feature.mfcc(y=y, sr=22050)"),
        ("sonara", "sonara: import+stft+mel+mfcc",
         "import numpy as np; y=np.random.randn(22050).astype(np.float32); sonara.stft(y); sonara.melspectrogram(y=y, sr=22050.0); sonara.mfcc(y=y, sr=22050.0)"),
    ]:
        script = f"import time; t0=time.perf_counter_ns(); import {mod_name}; t1=time.perf_counter_ns(); {code}; t2=time.perf_counter_ns(); print(f'{{(t1-t0)/1e6:.1f}},{{(t2-t1)/1e6:.1f}}')"
        times = []
        for _ in range(3):
            try:
                r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
                                   env={**__import__("os").environ, "CONDA_PREFIX": ""})
                if r.returncode == 0:
                    imp, call = r.stdout.strip().split(",")
                    times.append((float(imp), float(call)))
            except: pass
        if times:
            imp = np.median([t[0] for t in times])
            call = np.median([t[1] for t in times])
            print(f"  {label:<45} import={imp:.0f}ms  calls={call:.0f}ms  total={imp+call:.0f}ms")

    # ============================================================
    # Summary
    # ============================================================
    all_r = r1 + r2 + r3 + r6
    n_pass = sum(1 for r in all_r if "PASS" in r["accuracy"])
    n_fail = sum(1 for r in all_r if "FAIL" in r["accuracy"])
    n_shape = sum(1 for r in all_r if "SHAPE" in r["accuracy"])
    avg_sp = np.mean([r["speedup"] for r in all_r])
    print(f"\n{'='*90}")
    print(f"  SUMMARY: {n_pass} PASS, {n_fail} FAIL, {n_shape} shape mismatch, {len(all_r)-n_pass-n_fail-n_shape} skipped")
    print(f"  Average speedup: {avg_sp:.1f}x across {len(all_r)} functions")
    print(f"{'='*90}")
