"""A lightweight CNN-based mask model for speech enhancement.

This is intentionally compact: the goal is a tiny real-time model that actually
processes microphonic audio, not a large transformer. The model estimates a mask
for the STFT magnitude and attenuates noise while preserving speech energy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightMaskCNN(nn.Module):
    """Small CNN that predicts a magnitude mask from a spectrogram frame."""

    def __init__(self, n_bins=129, hidden=32):
        super().__init__()
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.net(x)
        return x.squeeze(1)


def _synthetic_speech_like(length=16000):
    t = np.linspace(0, 1, length, endpoint=False)
    tone = np.sin(2 * np.pi * 220 * t)
    formant = np.sin(2 * np.pi * 300 * t + 0.3)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)
    return (0.7 * tone + 0.3 * formant) * envelope


def _synthetic_noise(length=16000, noise_type="fan"):
    rng = np.random.default_rng(42)
    if noise_type == "fan":
        base = rng.normal(0.0, 0.4, length)
    elif noise_type == "traffic":
        base = np.sin(2 * np.pi * 90 * np.linspace(0, 1, length, endpoint=False)) * 0.6
        base += rng.normal(0.0, 0.2, length)
    elif noise_type == "music":
        base = np.sin(2 * np.pi * 220 * np.linspace(0, 1, length, endpoint=False)) * 0.3
        base += np.sin(2 * np.pi * 440 * np.linspace(0, 1, length, endpoint=False)) * 0.2
    else:
        base = rng.normal(0.0, 0.5, length)
    return base.astype(np.float32)


def _make_training_pair(samples=100):
    rng = np.random.default_rng(0)
    x_batch = []
    y_batch = []
    for _ in range(samples):
        clean = _synthetic_speech_like(length=16000)
        noise_type = ["fan", "traffic", "music", "general", "background"][rng.integers(0, 5)]
        noise = _synthetic_noise(length=16000, noise_type=noise_type)
        snr = float(rng.uniform(0, 12))
        noise_gain = 10 ** (-snr / 20.0)
        noisy = clean + noise_gain * noise
        noisy = noisy.astype(np.float32)
        clean = clean.astype(np.float32)
        # Keep the model small by using a simple 1D magnitude target.
        # The mask is learned as a ratio between clean and noisy magnitudes.
        spec_noisy = np.abs(np.fft.rfft(noisy, n=256))
        spec_clean = np.abs(np.fft.rfft(clean, n=256))
        mask = np.clip(spec_clean / (spec_noisy + 1e-6), 0.0, 1.5)
        x_batch.append(spec_noisy[:129])
        y_batch.append(mask[:129])
    return np.stack(x_batch), np.stack(y_batch)


def build_default_model(path: Path | str | None = None):
    """Create the model instance used in inference."""
    model = LightweightMaskCNN(n_bins=129)
    if path is not None:
        path = Path(path)
        if path.exists():
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                model.load_state_dict(state)
    return model.eval()


def train_quick_model(model_path: Path | str = "models/cnn_mask_model.pt", epochs=5, samples=100):
    """Train a tiny model on synthetic noisy-clean pairs and save it."""
    model = LightweightMaskCNN(n_bins=129)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x, y = _make_training_pair(samples=samples)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, save_path)
    return model


def save_metadata(path: Path, info: dict):
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
