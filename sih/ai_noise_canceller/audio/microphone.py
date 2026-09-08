"""Microphone helpers for selecting devices and reading live audio."""

from __future__ import annotations

import numpy as np
import sounddevice as sd


def list_audio_devices():
    """Return a list of available input/output devices in a simple format."""
    devices = sd.query_devices()
    results = []
    for idx, device in enumerate(devices):
        if isinstance(device, dict):
            name = device.get("name", f"Device {idx}")
            hostapi = device.get("hostapi")
            max_input_channels = device.get("max_input_channels", 0)
            max_output_channels = device.get("max_output_channels", 0)
            results.append({
                "index": idx,
                "name": name,
                "input": int(max_input_channels),
                "output": int(max_output_channels),
                "hostapi": hostapi,
            })
    return results


def select_input_device(index: int):
    """Return a valid microphone device index or raise an error if missing."""
    devices = list_audio_devices()
    if not devices:
        raise RuntimeError("No audio devices were detected on this machine.")
    if index < 0 or index >= len(devices):
        raise ValueError(f"Device index {index} is out of range.")
    return int(devices[index]["index"])


class AudioInput:
    """Thread-safe-ish helper for reading small chunks of microphone audio."""

    def __init__(self, device=None, samplerate=16000, blocksize=1024):
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.stream = None
        self.device_name = "Default microphone"

        if device is not None:
            try:
                self.device_name = sd.query_devices(device)["name"]
            except Exception:
                self.device_name = self.device_name

    def start(self):
        if self.stream is not None:
            return
        try:
            self.stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype="float32",
            )
            self.stream.start()
        except Exception as exc:  # pragma: no cover - system dependent
            raise RuntimeError(f"Unable to start microphone stream: {exc}") from exc

    def read(self, blocksize=None, timeout=None):
        if self.stream is None:
            self.start()
        if self.stream is None:
            return None
        frames = blocksize or self.blocksize
        try:
            chunk, _overflow = self.stream.read(frames)
            audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return None
            return audio
        except Exception:  # pragma: no cover - runtime hardware dependent
            return None

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
