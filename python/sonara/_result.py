"""TrackAnalysis: dict subclass returned by analyze_file / analyze_signal / analyze_batch."""

from __future__ import annotations


def _fmt_duration(sec: float) -> str:
    total = int(round(sec))
    m, s = divmod(total, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class TrackAnalysis(dict):
    """Result of `sonara.analyze_*`. Behaves as a dict; adds `.print()` for a human-readable summary."""

    def __repr__(self) -> str:
        parts = []
        if "bpm" in self:
            parts.append(f"{self['bpm']:.0f} BPM")
        if "key" in self:
            parts.append(str(self["key"]))
        if "energy" in self:
            parts.append(f"energy {self['energy']:.2f}")
        if "duration_sec" in self:
            parts.append(_fmt_duration(self["duration_sec"]))
        return f"<TrackAnalysis {' | '.join(parts)}>" if parts else "<TrackAnalysis>"

    def print(self) -> None:
        """Print a mode-aware summary, including only fields that were computed."""
        lines: list[str] = []
        if "duration_sec" in self:
            lines.append(f"TrackAnalysis  ({_fmt_duration(self['duration_sec'])})")
        else:
            lines.append("TrackAnalysis")

        rhythm: list[tuple[str, str]] = []
        if "bpm" in self:
            rhythm.append(("BPM", f"{self['bpm']:.1f}"))
        if "n_beats" in self:
            rhythm.append(("Beats", str(self["n_beats"])))
        if "onset_density" in self:
            rhythm.append(("Onset density", f"{self['onset_density']:.2f}/sec"))
        if "tempo_variability" in self:
            rhythm.append(("Tempo variability", f"{self['tempo_variability']:.3f}"))
        if "time_signature" in self:
            ts = self["time_signature"]
            conf = self.get("time_signature_confidence")
            rhythm.append(("Time signature", f"{ts}  (conf {conf:.2f})" if conf is not None else str(ts)))

        tonal: list[tuple[str, str]] = []
        if "key" in self:
            conf = self.get("key_confidence")
            tonal.append(("Key", f"{self['key']}  (conf {conf:.2f})" if conf is not None else str(self["key"])))
        if "predominant_chord" in self:
            tonal.append(("Predominant chord", str(self["predominant_chord"])))
        if "chord_change_rate" in self:
            tonal.append(("Chord changes", f"{self['chord_change_rate']:.2f}/sec"))
        if "dissonance" in self:
            tonal.append(("Dissonance", f"{self['dissonance']:.3f}"))

        perceptual: list[tuple[str, str]] = []
        for key, label in (
            ("energy", "Energy"),
            ("danceability", "Danceability"),
            ("valence", "Valence"),
            ("acousticness", "Acousticness"),
            ("instrumentalness", "Instrumentalness"),
        ):
            if key in self:
                perceptual.append((label, f"{self[key]:.2f}"))
        if "loudness_lufs" in self:
            perceptual.append(("Loudness", f"{self['loudness_lufs']:.1f} LUFS"))
        if "dynamic_range_db" in self:
            perceptual.append(("Dynamic range", f"{self['dynamic_range_db']:.1f} dB"))

        spectral: list[tuple[str, str]] = []
        if "spectral_centroid_mean" in self:
            spectral.append(("Centroid", f"{self['spectral_centroid_mean']:.0f} Hz"))
        if "spectral_bandwidth_mean" in self:
            spectral.append(("Bandwidth", f"{self['spectral_bandwidth_mean']:.0f} Hz"))
        if "spectral_rolloff_mean" in self:
            spectral.append(("Rolloff", f"{self['spectral_rolloff_mean']:.0f} Hz"))
        if "spectral_flatness_mean" in self:
            spectral.append(("Flatness", f"{self['spectral_flatness_mean']:.3f}"))
        if "zero_crossing_rate" in self:
            spectral.append(("ZCR", f"{self['zero_crossing_rate']:.3f}"))

        for name, rows in (
            ("Rhythm", rhythm),
            ("Tonal", tonal),
            ("Perceptual", perceptual),
            ("Spectral", spectral),
        ):
            if not rows:
                continue
            lines.append("")
            lines.append(f"  {name}")
            width = max(len(label) for label, _ in rows)
            for label, value in rows:
                lines.append(f"    {label:<{width}}  {value}")

        print("\n".join(lines))
