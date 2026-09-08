"""Real-time inference wrapper for the noise suppression model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import signal

from ai_noise_canceller.ml.model import LightweightMaskCNN, build_default_model
from ai_noise_canceller.ml.preprocessing import normalize_audio


class RealtimeEnhancer:
    """A tiny real-time enhancer that applies a mask to the noisy STFT."""

    def __init__(self, model_path: str | Path = "models/cnn_mask_model.pt", sr: int = 16000):
        self.sr = sr
        self.model_path = Path(model_path)
        self.model = None
        self.model_loaded = False
        self._build_or_load_model()

    def _build_or_load_model(self):
        if self.model_path.exists():
            self.model = build_default_model(self.model_path)
            self.model_loaded = True
        else:
            self.model = LightweightMaskCNN(n_bins=129)
            self.model_loaded = False

    def _predict_mask(self, noisy_audio):
        if self.model is None:
            return np.ones_like(noisy_audio, dtype=np.float32)

        noisy_audio = np.asarray(noisy_audio, dtype=np.float32)
        _, _, zxx = signal.stft(
            noisy_audio,
            fs=self.sr,
            nperseg=256,
            noverlap=128,
            boundary=None,
            padded=False,
        )
        mag = np.abs(zxx)[:129, :]
        if mag.size == 0:
            return np.ones_like(noisy_audio, dtype=np.float32)

        frames = torch.tensor(mag.T, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            pred = self.model(frames).cpu().numpy()
        pred = np.clip(pred, 0.05, 1.2).astype(np.float32)
        return pred

    def process(self, audio):
        """Enhance a single audio chunk in place and return a waveform."""
        if audio is None:
            return np.zeros(0, dtype=np.float32)

        arr = normalize_audio(np.asarray(audio, dtype=np.float32)).astype(np.float32)
        if arr.size == 0:
            return arr

        _, _, zxx = signal.stft(
            arr,
            fs=self.sr,
            nperseg=256,
            noverlap=128,
            boundary=None,
            padded=False,
        )
        mag = np.abs(zxx)[:129, :]
        phase = np.angle(zxx)[:129, :]

        mask = self._predict_mask(arr)
        if mask.ndim == 1:
            mask = np.tile(mask.reshape(1, -1), (mag.shape[1], 1)).T
        if mask.shape != mag.shape:
            mask = np.ones_like(mag, dtype=np.float32)

        enhanced_mag = mag * np.clip(mask, 0.05, 1.2)
        complex_spec = enhanced_mag * np.exp(1j * phase)
        _, recon = signal.istft(
            complex_spec,
            fs=self.sr,
            nperseg=256,
            noverlap=128,
            input_onesided=True,
        )
        out = np.asarray(recon, dtype=np.float32)
        if out.size > arr.size:
            out = out[: arr.size]
        return out
