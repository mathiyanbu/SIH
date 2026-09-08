"""Lightweight noise-type classifier for the prototype.

This keeps classification simple and hardware-independent. It estimates the
statistical character of the incoming audio chunk and maps it into a small set
of categories: stationary, non-stationary, impulsive, or unknown.
"""

from __future__ import annotations

import numpy as np


class NoiseClassifier:
    """Heuristic noise classifier based on spectral-temporal statistics."""

    def __init__(self):
        self.label = "Unknown"

    def classify(self, audio: np.ndarray) -> str:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return "Unknown"

        rms = float(np.sqrt(np.mean(arr ** 2)))
        if rms < 1e-6:
            return "Unknown"

        # Spectral flatness suggests stationary fan-like noise versus more dynamic
        # speech or environmental sounds.
        spec = np.abs(np.fft.rfft(arr[:4096]))
        spec = spec + 1e-8
        flatness = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-8))
        kurtosis = float(np.mean((arr - np.mean(arr)) ** 4) / (np.std(arr) ** 4 + 1e-8))

        if kurtosis > 8.0:
            self.label = "Impulsive"
        elif flatness > 0.6:
            self.label = "Stationary"
        elif flatness < 0.3 or kurtosis > 3.5:
            self.label = "Non-stationary"
        else:
            self.label = "Unknown"

        return self.label
