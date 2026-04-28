"""Sonara: High-performance audio analysis library for music information retrieval."""

from sonara._sonara import *  # noqa: F401, F403
from sonara._sonara import __version__  # noqa: F401 — sourced from Cargo.toml
from sonara._sonara import (
    analyze_file as _analyze_file,
    analyze_signal as _analyze_signal,
    analyze_batch as _analyze_batch,
)
from sonara._result import TrackAnalysis
from sonara import display  # noqa: F401


def analyze_file(path, *, sr=22050, mode="compact", features=None):
    """Analyze an audio file and return a `TrackAnalysis` (dict subclass with `.print()`)."""
    return TrackAnalysis(_analyze_file(path, sr=sr, mode=mode, features=features))


def analyze_signal(y, *, sr=22050, mode="compact", features=None):
    """Analyze a signal array and return a `TrackAnalysis` (dict subclass with `.print()`)."""
    return TrackAnalysis(_analyze_signal(y, sr=sr, mode=mode, features=features))


def analyze_batch(paths, *, sr=22050, mode="compact", features=None):
    """Analyze a list of audio files in parallel; returns a `list[TrackAnalysis]`."""
    return [
        TrackAnalysis(r)
        for r in _analyze_batch(paths, sr=sr, mode=mode, features=features)
    ]
