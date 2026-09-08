"""Feature extraction utilities for the lightweight speech-denoising model."""

from __future__ import annotations

import numpy as np
from scipy import signal


def to_mono(audio):
    """Convert multi-channel audio to mono by averaging channels."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=1, dtype=np.float32)
    if arr.ndim > 2:
        arr = np.reshape(arr, (arr.shape[0], -1))
        return arr.mean(axis=1, dtype=np.float32)
    return arr.astype(np.float32)


def normalize_audio(audio, target_peak=0.9):
    """Scale the waveform to a safe peak amplitude."""
    arr = np.asarray(audio, dtype=np.float32)
    peak = np.max(np.abs(arr)) if arr.size else 0.0
    if peak < 1e-8:
        return arr
    arr = arr / peak * target_peak
    return arr.astype(np.float32)


def stft_magnitude(audio, sr=16000, n_fft=256, hop_length=128):
    """Compute magnitude spectrogram from a 1D waveform."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)
    _, _, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None, padded=False)
    mag = np.abs(Zxx).astype(np.float32)
    mag = mag[:, : max(1, mag.shape[1])]
    return mag


def istft_from_mask(mag, phase, sr=16000, n_fft=256, hop_length=128):
    """Reconstruct a waveform from magnitude and phase representations."""
    complex_spec = mag * np.exp(1j * phase)
    _, recon = signal.istft(complex_spec, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, input_onesided=True)
    return recon.astype(np.float32)
