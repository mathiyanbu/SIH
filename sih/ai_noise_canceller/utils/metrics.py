"""Basic signal-quality metrics for the live enhancement prototype."""

from __future__ import annotations

import numpy as np


class NoiseMetrics:
    """Collect very simple real-time metrics for input/output monitoring."""

    def __init__(self):
        self.rms_in = 0.0
        self.rms_out = 0.0
        self.noise_in = 0.0
        self.noise_out = 0.0

    def update(self, noisy, enhanced):
        noisy = np.asarray(noisy, dtype=np.float32)
        enhanced = np.asarray(enhanced, dtype=np.float32)

        self.rms_in = float(np.sqrt(np.mean(np.square(noisy)))) if noisy.size else 0.0
        self.rms_out = float(np.sqrt(np.mean(np.square(enhanced)))) if enhanced.size else 0.0

        # Approximate background energy in the low-energy region.
        self.noise_in = float(np.median(np.abs(noisy))) if noisy.size else 0.0
        self.noise_out = float(np.median(np.abs(enhanced))) if enhanced.size else 0.0

    def as_dict(self):
        return {
            "rms_in": self.rms_in,
            "rms_out": self.rms_out,
            "noise_in": self.noise_in,
            "noise_out": self.noise_out,
        }
