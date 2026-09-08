"""Adapter for the real RNNoise implementation used by the application."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from pyrnnoise.pyrnnoise import create, destroy, process_frame


class RNNoiseLoadError(RuntimeError):
    """Raised when the real RNNoise backend cannot be initialized."""


class RNNoiseEngine:
    """Small application-facing adapter around the pyrnnoise native wrapper."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.sample_rate = 48000
        self.frame_size = 480
        self.backend = None
        self.state = None
        self._loaded = False
        self._load_error: str | None = None
        self.initialize()

    def initialize(self) -> bool:
        try:
            importlib.import_module("pyrnnoise")
            self.state = create()
            if not self.state:
                raise RNNoiseLoadError("pyrnnoise returned an empty RNNoise state")
            self.backend = "pyrnnoise-native"
            self._loaded = True
            return True
        except Exception as exc:  # pragma: no cover - depends on installed wheels
            self._load_error = str(exc)
            raise RNNoiseLoadError(
                "Unable to load the real pyrnnoise RNNoise backend. "
                "Install pyrnnoise==0.4.3, audiolab==0.4.9, and av==16.1.0. "
                f"Reason: {exc}"
            ) from exc

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._loaded or self.backend is None:
            raise RNNoiseLoadError(self._load_error or "RNNoise is not loaded")

        arr = np.asarray(frame, dtype=np.float32).reshape(-1)
        if arr.size < self.frame_size:
            arr = np.pad(arr, (0, self.frame_size - arr.size), mode="constant")
        elif arr.size > self.frame_size:
            arr = arr[: self.frame_size]

        try:
            output, _ = process_frame(self.state, arr)
            output = np.asarray(output).reshape(-1)
            if output.dtype == np.int16:
                output = output.astype(np.float32) / 32768.0
            else:
                output = output.astype(np.float32)
            if output.size != self.frame_size:
                raise RNNoiseLoadError(
                    f"RNNoise returned {output.size} samples; expected {self.frame_size}"
                )
            return output
        except RNNoiseLoadError:
            raise
        except Exception as exc:  # pragma: no cover - wrapper-specific runtime
            raise RNNoiseLoadError(f"RNNoise processing failed: {exc}") from exc

    def reset(self):
        if self.state is not None:
            destroy(self.state)
            self.state = create()

    def close(self):
        if self.state is not None:
            destroy(self.state)
            self.state = None
        self.backend = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def reason(self) -> str | None:
        return self._load_error


__all__ = ["RNNoiseEngine", "RNNoiseLoadError"]
