"""Microphone helpers for selecting devices and reading live audio."""

from __future__ import annotations

import sys

import numpy as np
import sounddevice as sd


def get_platform_name() -> str:
    """Return a normalized platform name for Windows/Linux handling."""
    name = sys.platform.lower()
    if name.startswith("win"):
        return "windows"
    if name.startswith("linux"):
        return "linux"
    return "other"


def list_audio_devices():
    """Return a list of available input/output devices in a simple format."""
    try:
        devices = sd.query_devices()
    except Exception:
        return []

    results = []
    for idx, device in enumerate(devices):
        if not isinstance(device, dict):
            continue
        name = device.get("name", f"Device {idx}")
        hostapi = device.get("hostapi")
        max_input_channels = int(device.get("max_input_channels", 0) or 0)
        max_output_channels = int(device.get("max_output_channels", 0) or 0)
        results.append({
            "index": int(idx),
            "name": str(name),
            "input": max_input_channels,
            "output": max_output_channels,
            "hostapi": hostapi,
        })
    return results


def select_input_device(index: int):
    """Return a valid microphone device index or raise an error if missing."""
    devices = list_audio_devices()
    if not devices:
        raise RuntimeError("No audio devices were detected on this machine.")

    valid_indexes = {int(device["index"]) for device in devices if int(device.get("input", 0) or 0) > 0}
    if index not in valid_indexes:
        if not valid_indexes:
            raise RuntimeError("No valid microphone input device was detected.")
        raise ValueError(f"Device index {index} is out of range for the available inputs.")
    return int(index)


def normalize_device_index(index, kind: str = "input"):
    """Return a non-negative device index only when it is valid for the requested kind."""
    try:
        value = int(index)
    except (TypeError, ValueError):
        return None

    if value < 0:
        return None

    device_list = list_audio_devices()
    if not device_list:
        return None

    required_channels = "input" if kind == "input" else "output"
    valid_indexes = {int(device["index"]) for device in device_list if int(device.get(required_channels, 0) or 0) > 0}
    if valid_indexes and value not in valid_indexes:
        return None
    return value


def get_default_output_index():
    """Return the default output device index for the current system when it is valid."""
    try:
        default_device = sd.default.device
    except Exception:
        return None

    if isinstance(default_device, (list, tuple)) and len(default_device) >= 2:
        return normalize_device_index(default_device[1], kind="output")
    if isinstance(default_device, dict):
        return normalize_device_index(default_device.get("output"), kind="output")
    return normalize_device_index(default_device, kind="output")


def is_bluetooth_device(device):
    name = str(device.get("name", "")).lower()
    return any(keyword in name for keyword in ("bluetooth", "bluez", "a2dp", "headset", "earbuds", "airpods"))


def select_output_device(devices=None):
    """Choose a capable output, preferring Bluetooth, then existing USB, then default."""
    devices = devices if devices is not None else list_audio_devices()
    outputs = [device for device in devices if int(device.get("output", 0) or 0) > 0]
    if not outputs:
        return None

    bluetooth_output = next((device for device in outputs if is_bluetooth_device(device)), None)
    if bluetooth_output is not None:
        return int(bluetooth_output["index"])

    usb_output = next(
        (device for device in outputs if any(keyword in str(device.get("name", "")).lower() for keyword in ("usb", "usb pnp", "pcm2902"))),
        None,
    )
    if usb_output is not None:
        return int(usb_output["index"])

    default_index = get_default_output_index()
    if default_index is not None:
        return default_index
    return int(outputs[0]["index"])


def _selection_matches_device(text, device):
    if ":" not in text:
        return True
    selected_name = text.split(":", 1)[1].strip()
    return selected_name == str(device.get("name", ""))


def resolve_device_index(selection: str | None, device_list: list[dict] | None = None):
    """Parse a UI device string and ensure its index still names the same device."""
    if selection is None:
        return None

    device_list = device_list or list_audio_devices()
    text = str(selection).strip()
    if not text or text.lower() in {"default output", "default microphone", "no microphone detected", "default"}:
        return None

    try:
        index = int(text.split(":", 1)[0].strip())
    except ValueError:
        return None

    selected_device = next((device for device in device_list if int(device.get("index", -1)) == index), None)
    if selected_device is None or not _selection_matches_device(text, selected_device):
        return None
    kind = "output" if int(selected_device.get("output", 0) or 0) > 0 else "input"
    normalized = normalize_device_index(index, kind=kind)
    if normalized is None:
        return None
    return normalized


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
