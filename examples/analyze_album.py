#!/usr/bin/env python3
"""Analyze MP3 files using canora — extract BPM, key features, and audio stats.

Usage:
    python examples/analyze_album.py /path/to/album/folder
"""

import os
import sys
import time
import numpy as np

import canora

if len(sys.argv) < 2:
    print("Usage: python examples/analyze_album.py /path/to/album/folder")
    sys.exit(1)

MUSIC_DIR = sys.argv[1]

SR = 22050

def analyze_track(filepath):
    """Analyze a single track and return stats dict."""
    filename = os.path.basename(filepath)
    stats = {"filename": filename}

    t0 = time.perf_counter()

    # Load audio
    try:
        y, sr = canora.load(filepath, sr=SR)
    except Exception as e:
        stats["error"] = str(e)
        return stats

    stats["duration_sec"] = round(len(y) / sr, 1)
    stats["samples"] = len(y)

    # BPM / Beat tracking
    try:
        tempo, beats = canora.beat_track(y=y, sr=SR)
        stats["bpm"] = round(tempo, 1)
        stats["n_beats"] = len(beats)
    except Exception as e:
        stats["bpm"] = None
        stats["bpm_error"] = str(e)

    # RMS energy (loudness proxy)
    try:
        rms = canora.rms(y=y)
        stats["rms_mean"] = round(float(np.mean(rms)), 4)
        stats["rms_max"] = round(float(np.max(rms)), 4)
        # Dynamic range in dB
        rms_vals = rms.flatten()
        rms_nonzero = rms_vals[rms_vals > 1e-10]
        if len(rms_nonzero) > 10:
            stats["dynamic_range_db"] = round(
                20 * np.log10(np.percentile(rms_nonzero, 95) / np.percentile(rms_nonzero, 5)), 1
            )
    except Exception as e:
        stats["rms_error"] = str(e)

    # Spectral centroid (brightness)
    try:
        cent = canora.spectral_centroid(y=y, sr=float(SR))
        stats["spectral_centroid_mean_hz"] = round(float(np.mean(cent)), 1)
    except Exception as e:
        stats["centroid_error"] = str(e)

    # Zero crossing rate (percussiveness proxy)
    try:
        zc = canora.zero_crossings(y)
        zcr = float(np.sum(zc.astype(np.float64))) / len(y)
        stats["zero_crossing_rate"] = round(zcr, 4)
    except Exception as e:
        stats["zcr_error"] = str(e)

    # MFCC summary (timbral fingerprint)
    try:
        mfcc = canora.mfcc(y=y, sr=float(SR), n_mfcc=13)
        stats["mfcc_mean"] = [round(float(x), 2) for x in np.mean(mfcc, axis=1)]
    except Exception as e:
        stats["mfcc_error"] = str(e)

    # Onset density (rhythmic activity)
    try:
        onsets = canora.onset_detect(y=y, sr=SR)
        stats["onset_density_per_sec"] = round(len(onsets) / stats["duration_sec"], 2)
    except Exception as e:
        stats["onset_error"] = str(e)

    stats["analysis_time_sec"] = round(time.perf_counter() - t0, 2)
    return stats


def main():
    if not os.path.isdir(MUSIC_DIR):
        print(f"Directory not found: {MUSIC_DIR}")
        sys.exit(1)

    mp3_files = sorted([
        os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR)
        if f.lower().endswith(".mp3")
    ])

    print(f"Found {len(mp3_files)} MP3 files")
    print("=" * 100)

    all_stats = []
    total_t0 = time.perf_counter()

    for filepath in mp3_files:
        stats = analyze_track(filepath)
        all_stats.append(stats)

        # Print per-track summary
        name = stats["filename"]
        if "error" in stats:
            print(f"  ERROR  {name}: {stats['error']}")
            continue

        bpm = stats.get("bpm", "?")
        dur = stats.get("duration_sec", "?")
        rms = stats.get("rms_mean", "?")
        cent = stats.get("spectral_centroid_mean_hz", "?")
        dyn = stats.get("dynamic_range_db", "?")
        onset_d = stats.get("onset_density_per_sec", "?")
        t_sec = stats.get("analysis_time_sec", "?")

        print(f"  {name:<50s} BPM={bpm:<6}  dur={dur}s  rms={rms}  centroid={cent}Hz  dyn={dyn}dB  onsets/s={onset_d}  [{t_sec}s]")

    total_time = time.perf_counter() - total_t0

    # Summary table
    print()
    print("=" * 100)
    print(f"{'Track':<50s} {'BPM':>6s} {'Duration':>8s} {'RMS':>6s} {'Bright':>7s} {'Dyn(dB)':>8s} {'Onsets/s':>8s}")
    print("-" * 100)

    valid = [s for s in all_stats if "error" not in s]
    for s in valid:
        print(f"{s['filename']:<50s} {s.get('bpm','?'):>6} {s['duration_sec']:>7.1f}s {s.get('rms_mean','?'):>6} {s.get('spectral_centroid_mean_hz','?'):>6}Hz {s.get('dynamic_range_db','?'):>7} {s.get('onset_density_per_sec','?'):>8}")

    print("-" * 100)

    # Aggregate stats
    if valid:
        bpms = [s["bpm"] for s in valid if s.get("bpm")]
        durations = [s["duration_sec"] for s in valid]
        print(f"\nAlbum summary:")
        print(f"  Tracks: {len(valid)}")
        print(f"  Total duration: {sum(durations)/60:.1f} min")
        if bpms:
            print(f"  BPM range: {min(bpms):.0f} - {max(bpms):.0f}")
            print(f"  Mean BPM: {np.mean(bpms):.1f}")
            print(f"  Median BPM: {np.median(bpms):.1f}")
        centroids = [s["spectral_centroid_mean_hz"] for s in valid if s.get("spectral_centroid_mean_hz")]
        if centroids:
            print(f"  Brightness range: {min(centroids):.0f} - {max(centroids):.0f} Hz")
        print(f"\n  Total analysis time: {total_time:.1f}s ({total_time/len(valid):.1f}s per track)")


if __name__ == "__main__":
    main()
