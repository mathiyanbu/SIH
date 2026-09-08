"""Audio playback helpers for enhanced output."""

from __future__ import annotations

import numpy as np
import sounddevice as sd


class AudioOutput:
    def __init__(self, device=None, samplerate=16000, blocksize=1024):
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.stream = None
        self.device_name = "Default output"

        if device is not None:
            try:
                self.device_name = sd.query_devices(device)["name"]
            except Exception:
                self.device_name = self.device_name

    def start(self):
        if self.stream is not None:
            return
        try:
            self.stream = sd.OutputStream(
                device=self.device,
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype="float32",
            )
            self.stream.start()
        except Exception as exc:  # pragma: no cover - device dependent
            raise RuntimeError(f"Unable to start playback stream: {exc}") from exc

    def write(self, audio):
        if self.stream is None:
            self.start()
        if self.stream is None:
            return
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        try:
            self.stream.write(audio[: self.blocksize])
        except Exception:
            pass

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
