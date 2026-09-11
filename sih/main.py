"""NOISELESS record-first RNNoise speech enhancement prototype."""

from __future__ import annotations

import argparse
from collections import deque
import platform
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from gpiozero import Button
except ImportError:
    Button = None

from ai_noise_canceller.audio.microphone import (
    get_default_output_index,
    get_platform_name,
    list_audio_devices,
    normalize_device_index,
    resolve_device_index,
    is_bluetooth_device,
    is_pipewire_device,
    select_output_device,
)
from noise_suppression.rnnoise_engine import RNNoiseEngine, RNNoiseLoadError
from ai_noise_canceller.audio.network_stream import NETWORK_PORT, NetworkAudioServer

ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
SAMPLE_RATE = 48000
FRAME_SIZE = 480
PLAYBACK_GAIN = 2.0
GPIO_RECORD_PIN = 17
GPIO_PLAY_ORIGINAL_PIN = 27
GPIO_PLAY_PROCESSED_PIN = 22
GPIO_LIVE_PIN = 23


def dbfs(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(values * values))) if values.size else 0.0
    return -120.0 if rms <= 1e-7 else 20.0 * np.log10(min(1.0, rms))


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if np.asarray(audio).size else 0.0
    return -120.0 if peak <= 1e-7 else 20.0 * np.log10(min(1.0, peak))


def audio_devices():
    try:
        return list_audio_devices()
    except Exception:
        return []


def test_audio() -> int:
    devices = audio_devices()
    print(f"Detected devices: {len(devices)}")
    for item in devices:
        print(f"[{item['index']}] {item['name']} input={item['input']} output={item['output']}")
    try:
        engine = RNNoiseEngine()
        frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        output = engine.process_frame(frame)
        print(f"RNNoise loaded: {engine.is_loaded()}")
        print(f"RNNoise frame: {len(output)} samples at {engine.sample_rate} Hz")
        return 0
    except Exception as exc:
        print(f"Audio test failed: {exc}")
        return 1


class NoiselessApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NOISELESS | Real-Time Speech Enhancement")
        self.root.geometry("1240x780")
        self.root.minsize(960, 640)
        self.root.configure(bg="#0b100d")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.devices = audio_devices()
        self.inputs = [d for d in self.devices if d.get("input", 0) > 0]
        self.outputs = [d for d in self.devices if d.get("output", 0) > 0]
        self.recording = False
        self.capture_stream = None
        self.capture_queue = queue.Queue(maxsize=32)
        self.capture_worker = None
        self.record_chunks: list[np.ndarray] = []
        self.record_started = 0.0
        self.noisy_audio = np.zeros(0, dtype=np.float32)
        self.enhanced_audio = np.zeros(0, dtype=np.float32)
        self.playback_stream = None
        self.playback_stop = threading.Event()
        self.playback_done = threading.Event()
        self.playback_buffer = np.zeros(0, dtype=np.float32)
        self.playback_position = 0
        self.playback_channels = 1
        self.playback_lock = threading.Lock()
        self.playback_thread = None
        self.live_streams = []
        self.live_worker = None
        self.live_stop = threading.Event()
        self.live_input_queue = queue.Queue(maxsize=8)
        self.live_output_queue = queue.Queue(maxsize=16)
        self.live_processing = False
        self.live_overflows = 0
        self.live_underruns = 0
        self.live_last_diagnostic = 0.0
        self.live_waveform_lock = threading.Lock()
        self.live_input_wave_frames = deque(maxlen=100)
        self.live_output_wave_frames = deque(maxlen=100)
        self.live_waveform_after_id = None
        self.closing = False
        self.network_server = None
        self.network_enabled = True
        self.ai_enabled = True
        self.input_overflows = 0
        self.processing_errors = 0
        self.frames_processed = 0
        self.zero_output_frames = 0
        self.nonzero_output_frames = 0
        self.process_time_ms = 0.0
        self.engine = None
        self.engine_error = ""
        self.gpio_buttons = []
        self.gpio_auto_process = False
        try:
            self.engine = RNNoiseEngine()
        except RNNoiseLoadError as exc:
            self.engine_error = str(exc)

        self.status_var = tk.StringVar(value="SYSTEM READY")
        self.timer_var = tk.StringVar(value="00:00")
        self.input_level_var = tk.StringVar(value="NOT MEASURED")
        self.output_level_var = tk.StringVar(value="NOT MEASURED")
        self.input_rms_var = tk.StringVar(value="NOT MEASURED")
        self.output_rms_var = tk.StringVar(value="NOT MEASURED")
        self.input_peak_var = tk.StringVar(value="NOT MEASURED")
        self.output_peak_var = tk.StringVar(value="NOT MEASURED")
        self.duration_original_var = tk.StringVar(value="Duration: --")
        self.duration_enhanced_var = tk.StringVar(value="Duration: --")
        self.process_time_var = tk.StringVar(value="NOT MEASURED")
        self.frames_var = tk.StringVar(value="0")
        self.zero_frames_var = tk.StringVar(value="0")
        self.nonzero_frames_var = tk.StringVar(value="0")
        self.input_overflow_var = tk.StringVar(value="0")
        self.processing_error_var = tk.StringVar(value="0")
        self.analysis_var = tk.StringVar(value="UNCLASSIFIED")
        self.speech_var = tk.StringVar(value="NOT MEASURED")
        self.impulse_var = tk.StringVar(value="NOT MEASURED")
        self.engine_status_var = tk.StringVar(value="READY" if self.engine else "ERROR")
        self.mic_var = tk.StringVar(value=self._device_label(self.inputs[0]) if self.inputs else "No microphone detected")
        default_output = self._preferred_output_index()
        default_output_device = next((device for device in self.outputs if device["index"] == default_output), None)
        self.output_var = tk.StringVar(value=self._device_label(default_output_device) if default_output_device else "Default output")
        self.ab_var = tk.StringVar(value="ORIGINAL")
        self.network_status_var = tk.StringVar(value="Starting")
        self.network_url_var = tk.StringVar(value="Phone URL: unavailable")

        self._build_ui()
        self._setup_gpio_buttons()
        self._setup_network_server()
        self._wave_stacked = False
        self.root.bind("<Configure>", self._on_resize)
        self.root.after(40, self._ui_tick)

    @staticmethod
    def _device_label(device):
        return f"{device['index']}: {device['name']}"

    def _setup_gpio_buttons(self):
        if Button is None:
            print("[GPIO] gpiozero is not installed; physical buttons are disabled.")
            return
        try:
            record_button = Button(GPIO_RECORD_PIN, pull_up=True, bounce_time=0.2)
            original_button = Button(GPIO_PLAY_ORIGINAL_PIN, pull_up=True, bounce_time=0.2)
            processed_button = Button(GPIO_PLAY_PROCESSED_PIN, pull_up=True, bounce_time=0.2)
            live_button = Button(GPIO_LIVE_PIN, pull_up=True, bounce_time=0.2)
            self.gpio_buttons = [record_button, original_button, processed_button, live_button]
            record_button.when_pressed = lambda: self.root.after(0, self._toggle_recording)
            original_button.when_pressed = lambda: self.root.after(0, self._play_original_audio)
            processed_button.when_pressed = lambda: self.root.after(0, self._play_processed_audio)
            live_button.when_pressed = lambda: self.root.after(0, self.toggle_live_processing)
            print("[GPIO] Buttons ready: GPIO17 record/stop, GPIO27 original, GPIO22 processed, GPIO23 live")
        except Exception as exc:
            print(f"[GPIO] Buttons unavailable: {exc}")
            self._close_gpio_buttons()

    def _close_gpio_buttons(self):
        for button in self.gpio_buttons:
            try:
                button.close()
            except Exception as exc:
                print(f"[GPIO] Button close ERROR: {exc}")
        self.gpio_buttons = []

    def _toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.gpio_auto_process = True
            self.start_recording()

    def _play_original_audio(self):
        self.play_audio(self.noisy_audio)

    def _play_processed_audio(self):
        self.play_audio(self.enhanced_audio)

    def _preferred_output_index(self):
        return select_output_device(self.outputs)

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#0b100d", padx=22, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="NOISELESS", fg="#eef0e7", bg="#0b100d", font=("Arial", 25, "bold")).pack(anchor="w")
        tk.Label(header, text="REAL-TIME SPEECH ENHANCEMENT FOR DEFENCE COMMUNICATION", fg="#aab8a1", bg="#0b100d", font=("Arial", 9, "bold")).pack(anchor="w", pady=(2, 0))
        header_right = tk.Frame(header, bg="#0b100d")
        header_right.place(relx=1.0, rely=0.22, anchor="e")
        self.led = tk.Label(header_right, text="●", fg="#a8c86b", bg="#0b100d", font=("Arial", 14, "bold"))
        self.led.pack(side="left")
        tk.Label(header_right, textvariable=self.status_var, fg="#a8c86b", bg="#0b100d", font=("Arial", 10, "bold")).pack(side="left", padx=(5, 14))
        tk.Label(header_right, text="48 kHz  |  RNNoise", fg="#eef0e7", bg="#0b100d", font=("Arial", 10, "bold")).pack(side="left")

        body = tk.Frame(self.root, bg="#101610", padx=12, pady=12)
        body.pack(fill="both", expand=True)
        top = tk.Frame(body, bg="#101610")
        top.pack(fill="x")
        self.input_wave = self._wave_panel(top, "INPUT - NOISY AUDIO", "#f0a23a")
        self.output_wave = self._wave_panel(top, "OUTPUT - ENHANCED AUDIO", "#8ed45b")

        middle = tk.Frame(body, bg="#101610")
        middle.pack(fill="x", pady=(8, 0))
        self._metric(middle, "INPUT LEVEL", self.input_level_var, "#ff6670")
        self._metric(middle, "OUTPUT LEVEL", self.output_level_var, "#36db91")
        self._metric(middle, "RNNOISE PROCESSING", self.process_time_var, "#b7d57b")
        self._metric(middle, "RECORDING TIME", self.timer_var, "#e4b55d")

        controls = tk.Frame(body, bg="#101610")
        controls.pack(fill="x", pady=(8, 0))
        self._control_panel(controls, "RECORD AUDIO", self._record_controls).pack(side="left", fill="both", expand=True, padx=(0, 10))
        self._control_panel(controls, "PROCESS", self._process_controls).pack(side="left", fill="both", expand=True, padx=(0, 10))
        self._control_panel(controls, "PLAYBACK / A-B", self._playback_controls).pack(side="left", fill="both", expand=True)
        self._control_panel(body, "NETWORK STREAMING", self._network_controls).pack(fill="x", pady=(8, 0))

        bottom = tk.Frame(body, bg="#101610")
        bottom.pack(fill="x", pady=(8, 0))
        self._info_panel(bottom, "AUDIO ANALYSIS", self._analysis_rows()).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._info_panel(bottom, "PROCESSING ENGINE", self._engine_rows()).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._info_panel(bottom, "AUDIO HEALTH", self._health_rows()).pack(side="left", fill="x", expand=True)
        footer = tk.Frame(body, bg="#101610")
        footer.pack(fill="x", pady=(9, 0))
        tk.Label(footer, text="RECORD  ->  PROCESS  ->  COMPARE", fg="#b7d57b", bg="#101610", font=("Arial", 9, "bold")).pack(side="left")
        tk.Label(footer, text="Capture noisy speech, enhance with RNNoise, and compare the result.", fg="#7f917f", bg="#101610", font=("Arial", 9)).pack(side="left", padx=(18, 0))
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground="#242d22", background="#242d22", foreground="#ecf0e5", bordercolor="#637052", arrowcolor="#b3c77a")
        style.map("TCombobox", fieldbackground=[("readonly", "#242d22")], foreground=[("readonly", "#ecf0e5")])
        self._apply_military_theme(self.root)

    def _apply_military_theme(self, widget):
        kind = widget.winfo_class()
        if kind == "Canvas":
            widget.configure(bg="#0a100d")
        elif kind == "Button":
            text = str(widget.cget("text"))
            button_color = "#b93c36" if "RECORD" in text else "#6e9b42" if "PROCESS" in text else "#303b30"
            widget.configure(bg=button_color, fg="#f5f7ef", activebackground="#789d4b", activeforeground="#ffffff")
        elif kind in ("Label", "Frame"):
            widget.configure(bg="#171d18")
            if kind == "Label":
                widget.configure(fg="#ecf0e5")
        elif kind == "Radiobutton":
            widget.configure(bg="#171d18", fg="#ecf0e5", activebackground="#171d18", activeforeground="#ffffff", selectcolor="#374531")
        for child in widget.winfo_children():
            self._apply_military_theme(child)

    def _wave_panel(self, parent, title, color):
        panel = tk.Frame(parent, bg="#171d18", bd=1, relief="solid")
        tk.Label(panel, text=title, fg="#ecf0e5", bg="#171d18", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 0))
        canvas = tk.Canvas(panel, height=150, bg="#0a100d", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=9, pady=7)
        canvas.wave_color = color
        canvas.wave_data = np.zeros(FRAME_SIZE, dtype=np.float32)
        placeholder = "NO ENHANCED AUDIO" if "OUTPUT" in title else "WAITING FOR AUDIO"
        canvas.create_text(18, 75, text=placeholder, fill="#4e604e", anchor="w", font=("Arial", 9, "bold"))
        panel.pack(side="left", fill="both", expand=True, padx=(0, 10) if "INPUT" in title else (0, 0))
        return canvas

    def _on_resize(self, event):
        if event.widget is not self.root:
            return
        narrow = event.width < 980
        if narrow == self._wave_stacked:
            return
        input_panel = self.input_wave.master
        output_panel = self.output_wave.master
        input_panel.pack_forget()
        output_panel.pack_forget()
        if narrow:
            input_panel.pack(side="top", fill="x", expand=False, pady=(0, 8))
            output_panel.pack(side="top", fill="x", expand=False)
        else:
            input_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
            output_panel.pack(side="left", fill="both", expand=True)
        self._wave_stacked = narrow

    def _metric(self, parent, title, variable, color):
        card = tk.Frame(parent, bg="#171d18", bd=1, relief="solid")
        card.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(card, text=title, fg="#aab8a1", bg="#171d18", font=("Arial", 8, "bold")).pack(anchor="w", padx=10, pady=(7, 0))
        tk.Label(card, textvariable=variable, fg=color, bg="#171d18", font=("Arial", 13, "bold")).pack(anchor="w", padx=10, pady=(2, 7))

    def _control_panel(self, parent, title, builder):
        panel = tk.Frame(parent, bg="#171d18", bd=1, relief="solid")
        tk.Label(panel, text=title, fg="#ecf0e5", bg="#171d18", font=("Arial", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        builder(panel)
        return panel

    def _record_controls(self, panel):
        row = tk.Frame(panel, bg="#0d2942")
        row.pack(fill="x", padx=10)
        self.mic_selector = ttk.Combobox(row, textvariable=self.mic_var, values=[self._device_label(d) for d in self.inputs] or ["No microphone detected"], state="readonly", width=25)
        self.mic_selector.pack(side="left")
        self.record_start_button = tk.Button(row, text="START RECORDING", command=self.start_recording, bg="#e84b54", fg="white", bd=0, padx=10, pady=7, font=("Arial", 10, "bold"))
        self.record_start_button.pack(side="left", padx=(8, 4))
        tk.Button(row, text="STOP", command=self.stop_recording, bg="#203f5e", fg="white", bd=0, padx=12, pady=7, font=("Arial", 10, "bold")).pack(side="left")
        tk.Label(panel, text="Recording is stored as real microphone audio", fg="#89a9c4", bg="#0d2942", font=("Arial", 9)).pack(anchor="w", padx=12, pady=(8, 10))

    def _process_controls(self, panel):
        self.process_button = tk.Button(panel, text="PROCESS WITH RNNOISE", command=self.process_audio, bg="#2e9ee8", fg="white", bd=0, padx=14, pady=9, font=("Arial", 11, "bold"))
        self.process_button.pack(anchor="w", padx=12, pady=(2, 6))
        self.live_button = tk.Button(panel, text="START LIVE PROCESSING", command=self.toggle_live_processing, bg="#6e9b42", fg="white", bd=0, padx=14, pady=9, font=("Arial", 10, "bold"))
        self.live_button.pack(anchor="w", padx=12, pady=(0, 6))
        tk.Label(panel, textvariable=self.engine_status_var, fg="#36db91", bg="#0d2942", font=("Arial", 10, "bold")).pack(anchor="w", padx=12)
        tk.Label(panel, text="Recorded and live processing are independent", fg="#89a9c4", bg="#0d2942", font=("Arial", 9)).pack(anchor="w", padx=12, pady=(4, 10))

    def _playback_controls(self, panel):
        row = tk.Frame(panel, bg="#0d2942")
        row.pack(fill="x", padx=10)
        self.output_selector = ttk.Combobox(row, textvariable=self.output_var, values=[self._device_label(d) for d in self.outputs] or ["Default output"], state="readonly", width=25)
        self.output_selector.pack(side="left", padx=(0, 8))
        tk.Button(row, text="REFRESH", command=self.refresh_audio_devices, bg="#303b30", fg="white", bd=0, padx=8, pady=7, font=("Arial", 9, "bold")).pack(side="left")
        tk.Button(row, text="PLAY ORIGINAL", command=lambda: self.play_audio(self.noisy_audio), bg="#b74754", fg="white", bd=0, padx=9, pady=7, font=("Arial", 10, "bold")).pack(side="left", padx=(0, 4))
        tk.Button(row, text="PLAY ENHANCED", command=lambda: self.play_audio(self.enhanced_audio), bg="#229d70", fg="white", bd=0, padx=9, pady=7, font=("Arial", 10, "bold")).pack(side="left", padx=4)
        tk.Button(row, text="STOP AUDIO", command=self.stop_playback, bg="#203f5e", fg="white", bd=0, padx=9, pady=7, font=("Arial", 10, "bold")).pack(side="left", padx=4)
        ab = tk.Frame(panel, bg="#0d2942")
        ab.pack(anchor="w", padx=12, pady=(9, 10))
        tk.Label(ab, text="A / B:", fg="#91aec9", bg="#0d2942", font=("Arial", 10, "bold")).pack(side="left")
        tk.Radiobutton(ab, text="ORIGINAL", variable=self.ab_var, value="ORIGINAL", command=self.play_ab, bg="#0d2942", fg="#eef7ff", selectcolor="#1b4162", activebackground="#0d2942").pack(side="left")
        tk.Radiobutton(ab, text="ENHANCED", variable=self.ab_var, value="ENHANCED", command=self.play_ab, bg="#0d2942", fg="#eef7ff", selectcolor="#1b4162", activebackground="#0d2942").pack(side="left")

    def _network_controls(self, panel):
        row = tk.Frame(panel, bg="#0d2942")
        row.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(row, textvariable=self.network_status_var, fg="#36db91", bg="#0d2942", font=("Arial", 9, "bold")).pack(side="left")
        self.network_button = tk.Button(row, text="STOP PHONE STREAM", command=self.toggle_phone_stream, bg="#6e9b42", fg="white", bd=0, padx=10, pady=6, font=("Arial", 9, "bold"))
        self.network_button.pack(side="right")
        tk.Label(panel, textvariable=self.network_url_var, fg="#89a9c4", bg="#0d2942", font=("Arial", 9)).pack(anchor="w", padx=12, pady=(0, 8))

    def _info_panel(self, parent, title, rows):
        panel = tk.Frame(parent, bg="#171d18", bd=1, relief="solid")
        tk.Label(panel, text=title, fg="#ecf0e5", bg="#171d18", font=("Arial", 11, "bold")).pack(anchor="w", padx=13, pady=(11, 7))
        for label, variable in rows:
            row = tk.Frame(panel, bg="#171d18")
            row.pack(fill="x", padx=13, pady=3)
            tk.Label(row, text=label, fg="#aab8a1", bg="#171d18", font=("Arial", 9)).pack(side="left")
            tk.Label(row, textvariable=variable, fg="#ecf0e5", bg="#171d18", font=("Arial", 9, "bold")).pack(side="right")
        return panel

    def _analysis_rows(self):
        return [("Condition", self.analysis_var), ("Speech", self.speech_var), ("Impulse", self.impulse_var), ("Input RMS", self.input_rms_var), ("Output RMS", self.output_rms_var), ("Input Peak", self.input_peak_var), ("Output Peak", self.output_peak_var)]

    def _engine_rows(self):
        return [("Model", tk.StringVar(value="RNNoise")), ("Status", self.engine_status_var), ("Frames Processed", self.frames_var), ("Non-zero Output", self.nonzero_frames_var), ("Zero Output", self.zero_frames_var), ("Processing Time", self.process_time_var)]

    def _health_rows(self):
        return [("Input Overflow", self.input_overflow_var), ("Output Underflow", tk.StringVar(value="Not measured in record mode")), ("Processing Errors", self.processing_error_var), ("Dropped Frames", tk.StringVar(value="0"))]

    def _ui_tick(self):
        if self.recording:
            elapsed = max(0, int(time.time() - self.record_started))
            self.timer_var.set(f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        if self.status_var.get().endswith("ERROR"):
            self.led.config(fg="#ff5f68")
        elif self.recording:
            self.led.config(fg="#e4b55d")
        else:
            self.led.config(fg="#37df8b")
        self._report_live_diagnostics()
        self._update_network_status()
        self.root.after(40, self._ui_tick)

    def _setup_network_server(self):
        self.network_server = NetworkAudioServer(status_provider=self._network_status_snapshot, port=NETWORK_PORT)
        if self.network_server.start():
            self.network_url_var.set(f"Phone URL: {self.network_server.display_url}")
            self.network_status_var.set("Phone Streaming: Ready")
        else:
            print(f"[Network] Audio server unavailable: {self.network_server.start_error}")
            self.network_status_var.set("Phone Streaming: Unavailable")
            self.network_button.config(text="START PHONE STREAM", state="disabled")

    def _network_status_snapshot(self):
        return {"live": bool(self.live_processing), "enabled": bool(self.network_enabled)}

    def _update_network_status(self):
        if self.network_server is None or self.network_server.start_error:
            return
        clients = self.network_server.client_count
        if not self.network_enabled:
            status = "Phone Streaming: Stopped"
        elif clients:
            status = f"Phone Streaming: Connected ({clients})"
        elif self.live_processing:
            status = "Phone Streaming: Ready"
        else:
            status = "Phone Streaming: Waiting for live processing"
        self.network_status_var.set(status)

    def toggle_phone_stream(self):
        if self.network_server is None or self.network_server.start_error:
            return
        self.network_enabled = not self.network_enabled
        self.network_server.set_enabled(self.network_enabled)
        if self.network_enabled:
            self.network_button.config(text="STOP PHONE STREAM", bg="#6e9b42")
        else:
            self.network_button.config(text="START PHONE STREAM", bg="#303b30")
        self._update_network_status()

    def _draw_wave(self, canvas, values):
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        canvas.delete("all")
        if not arr.size:
            return
        display_limit = 1800
        if arr.size > display_limit:
            stride = int(np.ceil(arr.size / display_limit))
            arr = arr[::stride]
        width = max(200, canvas.winfo_width())
        height = max(100, canvas.winfo_height())
        peak = max(0.00001, float(np.max(np.abs(arr))))
        canvas.create_line(10, height / 2, width - 10, height / 2, fill="#42503c")
        points = []
        for index, value in enumerate(arr):
            x = 10 + index * (width - 20) / max(1, len(arr) - 1)
            y = height / 2 - (float(value) / peak) * height * 0.39
            points.extend((x, y))
        canvas.create_line(points, fill=canvas.wave_color, width=2)
        canvas.create_text(16, 14, text="+1", fill="#718169", anchor="w", font=("Arial", 8))
        canvas.create_text(16, height - 10, text="-1", fill="#718169", anchor="w", font=("Arial", 8))

    def _start_live_waveform_updates(self):
        if self.live_waveform_after_id is None:
            self.live_waveform_after_id = self.root.after(60, self._update_live_waveforms)

    def _update_live_waveforms(self):
        self.live_waveform_after_id = None
        if self.closing:
            return
        with self.live_waveform_lock:
            input_frames = list(self.live_input_wave_frames)
            output_frames = list(self.live_output_wave_frames)
        if input_frames:
            self._draw_wave(self.input_wave, np.concatenate(input_frames))
        if output_frames:
            self._draw_wave(self.output_wave, np.concatenate(output_frames))
        if self.live_processing:
            self.live_waveform_after_id = self.root.after(60, self._update_live_waveforms)

    def _capture_callback(self, indata, frames, time_info, status):
        if not self.recording:
            return
        try:
            self.capture_queue.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            self.input_overflows += 1

    def _live_diagnostic(self, message, count):
        now = time.monotonic()
        if count == 1 or now - self.live_last_diagnostic >= 2.0:
            print(f"[Live] {message}: {count}")
            self.live_last_diagnostic = now

    def _report_live_diagnostics(self):
        total = self.live_overflows + self.live_underruns
        if total:
            self._live_diagnostic(
                f"buffer events (underruns/overflows={self.live_underruns}/{self.live_overflows})",
                total,
            )

    def _enumerate_devices(self):
        self.devices = audio_devices()
        self.inputs = [device for device in self.devices if device.get("input", 0) > 0]
        self.outputs = [device for device in self.devices if device.get("output", 0) > 0]

    def _clear_live_queues(self):
        for live_queue in (self.live_input_queue, self.live_output_queue):
            try:
                while True:
                    live_queue.get_nowait()
            except queue.Empty:
                pass

    def refresh_audio_devices(self):
        if self.live_processing or self.recording:
            self._error("Stop recording or live processing before refreshing audio devices.")
            return
        old_mic = self.mic_var.get()
        old_output = self.output_var.get()
        self._enumerate_devices()
        if hasattr(self, "mic_selector"):
            self.mic_selector.configure(values=[self._device_label(device) for device in self.inputs] or ["No microphone detected"])
        selected_mic = resolve_device_index(old_mic, self.inputs)
        mic_device = next((device for device in self.inputs if device["index"] == selected_mic), None)
        self.mic_var.set(self._device_label(mic_device) if mic_device else (self._device_label(self.inputs[0]) if self.inputs else "No microphone detected"))
        if hasattr(self, "output_selector"):
            values = [self._device_label(device) for device in self.outputs] or ["Default output"]
            self.output_selector.configure(values=values)
        selected_output = resolve_device_index(old_output, self.outputs)
        output_device = next((device for device in self.outputs if device["index"] == selected_output), None)
        if output_device is None:
            preferred = self._preferred_output_index()
            output_device = next((device for device in self.outputs if device["index"] == preferred), None)
        self.output_var.set(self._device_label(output_device) if output_device else "Default output")
        print("[Audio] Devices refreshed")
        for device in self.devices:
            print(f"[Audio] [{device['index']}] {device['name']} input={device['input']} output={device['output']}")

    def _live_input_callback(self, indata, frames, time_info, status):
        if status:
            self.input_overflows += int(getattr(status, "input_overflow", False))
        if not self.live_processing:
            return
        try:
            audio = np.asarray(indata[:, 0], dtype=np.float32).copy()
            self.live_input_queue.put_nowait(audio)
            with self.live_waveform_lock:
                self.live_input_wave_frames.append(audio)
        except queue.Full:
            self.live_overflows += 1

    def _live_output_callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        if status:
            self.live_underruns += int(getattr(status, "output_underflow", False))
        try:
            frame = self.live_output_queue.get_nowait()
        except queue.Empty:
            self.live_underruns += 1
            return
        count = min(frames, len(frame))
        outdata[:count, 0] = frame[:count]
        if outdata.shape[1] > 1:
            outdata[:count, 1:] = frame[:count, None]

    def toggle_live_processing(self):
        if self.live_processing:
            if self.stop_live_processing():
                self.live_button.config(text="START LIVE PROCESSING", bg="#6e9b42")
        elif self.start_live_processing():
            self.live_button.config(text="STOP LIVE PROCESSING", bg="#b93c36")

    def start_live_processing(self):
        if self.live_processing:
            return False
        if self.recording:
            self._error("Stop recording before starting live processing.")
            return False
        if self.engine is None:
            self._error(self.engine_error or "RNNoise is not loaded.")
            return False

        self._enumerate_devices()
        if not self.inputs:
            self._error("No microphone is available. Connect a USB mic or headset and retry.")
            return False
        if not self.outputs:
            print("[Live] Bluetooth sink is not visible to PortAudio; no output-capable fallback is available.")
            self._error("No output device visible to PortAudio. Check PipeWire/PulseAudio/ALSA and Bluetooth routing.")
            return False

        input_index = resolve_device_index(self.mic_var.get(), self.inputs)
        output_index = self._selected_output_index()
        if input_index is None:
            self._error("The selected microphone is unavailable. Choose a detected input device.")
            return False
        if output_index is None:
            self._error("No valid output device was detected.")
            return False

        input_device = next((device for device in self.inputs if device["index"] == input_index), None)
        output_device = next((device for device in self.outputs if device["index"] == output_index), None)
        if is_pipewire_device(output_device or {}):
            print("[Live] Using PipeWire default output route.")
        elif not any(is_bluetooth_device(device) for device in self.outputs):
            print("[Live] Direct Bluetooth sink is not visible to PortAudio; using a valid fallback output.")
        self.stop_playback()
        print(f"[Live] Input device: [{input_index}] {input_device['name'] if input_device else 'Default input'}")
        print(f"[Live] Output device: [{output_index}] {output_device['name'] if output_device else 'Default output'}")
        try:
            self.live_input_queue = queue.Queue(maxsize=8)
            self.live_output_queue = queue.Queue(maxsize=16)
            self._clear_live_queues()
            with self.live_waveform_lock:
                self.live_input_wave_frames.clear()
                self.live_output_wave_frames.clear()
            self.live_stop.clear()
            self.live_overflows = 0
            self.live_underruns = 0
            self.live_processing = True
            output_channels = 2 if int(output_device.get("output", 1) if output_device else 1) >= 2 else 1
            input_stream = sd.InputStream(device=input_index, samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SIZE, dtype="float32", callback=self._live_input_callback)
            output_stream = sd.OutputStream(device=output_index, samplerate=SAMPLE_RATE, channels=output_channels, blocksize=FRAME_SIZE, dtype="float32", callback=self._live_output_callback)
            self.live_streams = [input_stream, output_stream]
            input_stream.start()
            output_stream.start()
            self.live_worker = threading.Thread(target=self._live_worker, daemon=True)
            self.live_worker.start()
            self._start_live_waveform_updates()
            self.status_var.set("LIVE PROCESSING")
            self.engine_status_var.set("LIVE")
            return True
        except Exception as exc:
            self.live_processing = False
            self.live_stop.set()
            self._close_live_streams()
            self._clear_live_queues()
            self._error(f"Live processing start failed: {exc}. Check the microphone, output device, and Linux audio configuration.")
            return False

    def _live_worker(self):
        while not self.live_stop.is_set():
            try:
                frame = self.live_input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                output = self.engine.process_frame(frame)
                with self.live_waveform_lock:
                    self.live_output_wave_frames.append(np.asarray(output, dtype=np.float32).copy())
                if self.network_server is not None:
                    self.network_server.publish_frame(output)
                try:
                    self.live_output_queue.put_nowait(output)
                except queue.Full:
                    try:
                        self.live_output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.live_output_queue.put_nowait(output)
                    except queue.Full:
                        self.live_overflows += 1
                        self._live_diagnostic("output queue overflow; dropped oldest frame", self.live_overflows)
            except Exception as exc:
                self.processing_errors += 1
                print(f"[Live] Processing ERROR: {exc}")
                self.live_processing = False
                self.live_stop.set()
                self.root.after(0, self._live_processing_failed, str(exc))
                return

    def _close_live_streams(self):
        for stream in self.live_streams:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception as exc:
                print(f"[Live] Stream close ERROR: {exc}")
        self.live_streams = []

    def _live_processing_failed(self, error):
        self.stop_live_processing()
        self.live_button.config(text="START LIVE PROCESSING", bg="#6e9b42")
        self._error(f"Live RNNoise processing failed: {error}")

    def stop_live_processing(self):
        if hasattr(self, "live_button"):
            self.live_button.config(text="START LIVE PROCESSING", bg="#6e9b42")
        if not self.live_processing and not self.live_streams:
            self._clear_live_queues()
            return True
        self.live_processing = False
        self.live_stop.set()
        self._close_live_streams()
        if self.live_worker is not None:
            self.live_worker.join()
            self.live_worker = None
        self._clear_live_queues()
        if self.live_waveform_after_id is not None:
            try:
                self.root.after_cancel(self.live_waveform_after_id)
            except tk.TclError:
                pass
            self.live_waveform_after_id = None
        self.status_var.set("SYSTEM READY")
        print(f"[Live] Stopped safely; input overflows={self.live_overflows}, output underruns={self.live_underruns}")
        return True

    def start_recording(self):
        if self.recording:
            return
        if self.live_processing:
            self._error("Stop live processing before starting a recording.")
            return
        if not self.inputs:
            self._error("No microphone is available. Connect a USB mic or headset and retry.")
            return

        device_index = resolve_device_index(self.mic_var.get(), self.inputs)
        if device_index is None:
            self._error("The selected microphone is unavailable. Choose a detected input device.")
            return

        try:
            self.capture_queue = queue.Queue(maxsize=32)
            self.record_chunks = []
            self.recording = True
            self.record_started = time.time()
            self.status_var.set("RECORDING")
            self.record_start_button.config(text="RECORDING...", bg="#e4a33b")
            self.capture_stream = sd.InputStream(device=device_index, samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SIZE, dtype="float32", callback=self._capture_callback)
            self.capture_stream.start()
            self.capture_worker = threading.Thread(target=self._record_worker, daemon=True)
            self.capture_worker.start()
        except Exception as exc:
            self.recording = False
            self._error(f"Recording start failed: {exc}. Check the microphone device and Linux audio configuration.")

    def _record_worker(self):
        while self.recording or not self.capture_queue.empty():
            try:
                self.record_chunks.append(self.capture_queue.get(timeout=0.1))
            except queue.Empty:
                continue

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.capture_stream is not None:
            self.capture_stream.stop()
            self.capture_stream.close()
            self.capture_stream = None
        if self.capture_worker is not None:
            self.capture_worker.join(timeout=1.0)
        if self.record_chunks:
            chunks = list(self.record_chunks)
            threading.Thread(target=self._finish_recording_worker, args=(chunks,), daemon=True).start()
        else:
            self._error("No microphone samples were captured.")
        self.record_start_button.config(text="START RECORDING", bg="#e84b54")
        self.timer_var.set("00:00")

    def _finish_recording_worker(self, chunks):
        try:
            noisy = np.concatenate(chunks).astype(np.float32)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            RECORDINGS_DIR.mkdir(exist_ok=True)
            path = RECORDINGS_DIR / f"noisy_{timestamp}.wav"
            sf.write(path, noisy, SAMPLE_RATE)
            self.root.after(0, self._recording_ready, noisy)
        except Exception as exc:
            self.root.after(0, self._error, f"Recording failed: {exc}")

    def _recording_ready(self, noisy):
        self.noisy_audio = noisy
        self._show_recorded_audio()
        self.status_var.set("RECORDED")
        if self.gpio_auto_process:
            self.gpio_auto_process = False
            self.process_audio()

    def _show_recorded_audio(self):
        self._draw_wave(self.input_wave, self.noisy_audio)
        self.input_level_var.set(f"{dbfs(self.noisy_audio):.1f} dB")
        self.input_rms_var.set(f"{dbfs(self.noisy_audio):.1f} dB")
        self.input_peak_var.set(f"{peak_dbfs(self.noisy_audio):.1f} dB")
        self.duration_original_var.set(f"Duration: {len(self.noisy_audio) / SAMPLE_RATE:.2f} s")

    def process_audio(self):
        if self.live_processing:
            self._error("Stop live processing before offline processing.")
            return
        if not self.noisy_audio.size:
            self._error("Record audio before processing.")
            return
        if self.engine is None:
            self._error(self.engine_error or "RNNoise is not loaded.")
            return
        self.process_button.config(state="disabled", text="PROCESSING...")
        self.engine_status_var.set("PROCESSING")
        threading.Thread(target=self._process_worker, daemon=True).start()

    def _process_worker(self):
        try:
            started = time.perf_counter()
            parts = []
            self.frames_processed = 0
            self.zero_output_frames = 0
            self.nonzero_output_frames = 0
            for offset in range(0, len(self.noisy_audio), FRAME_SIZE):
                frame = self.noisy_audio[offset : offset + FRAME_SIZE]
                if len(frame) < FRAME_SIZE:
                    frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
                output = self.engine.process_frame(frame)
                parts.append(output)
                self.frames_processed += 1
                if np.any(np.abs(output) > 1e-7):
                    self.nonzero_output_frames += 1
                else:
                    self.zero_output_frames += 1
            enhanced = np.concatenate(parts)[: len(self.noisy_audio)]
            elapsed = (time.perf_counter() - started) * 1000.0
            self.root.after(0, self._processing_complete, enhanced, elapsed)
        except Exception as exc:
            self.processing_errors += 1
            self.root.after(0, self._processing_failed, str(exc))

    def _processing_complete(self, enhanced, elapsed):
        self.enhanced_audio = np.asarray(enhanced, dtype=np.float32)
        self.process_time_ms = elapsed
        self._draw_wave(self.output_wave, self.enhanced_audio)
        self.output_level_var.set(f"{dbfs(self.enhanced_audio):.1f} dB")
        self.output_rms_var.set(f"{dbfs(self.enhanced_audio):.1f} dB")
        self.output_peak_var.set(f"{peak_dbfs(self.enhanced_audio):.1f} dB")
        self.duration_enhanced_var.set(f"Duration: {len(self.enhanced_audio) / SAMPLE_RATE:.2f} s")
        self.process_time_var.set(f"{elapsed:.2f} ms")
        self.frames_var.set(str(self.frames_processed))
        self.zero_frames_var.set(str(self.zero_output_frames))
        self.nonzero_frames_var.set(str(self.nonzero_output_frames))
        self.engine_status_var.set("COMPLETE")
        self.status_var.set("PROCESSING COMPLETE")
        self.process_button.config(state="normal", text="PROCESS WITH RNNOISE")

    def _processing_failed(self, error):
        self.processing_errors += 1
        self.engine_status_var.set("ERROR")
        self.process_button.config(state="normal", text="PROCESS WITH RNNOISE")
        self._error(f"RNNoise processing failed: {error}")

    def play_audio(self, audio):
        print("[Playback] Starting audio playback")
        values = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        if not self._validate_playback_audio(values):
            return

        if not self.outputs:
            self._error("No output device is available. Connect a speaker or Bluetooth headset and retry.")
            return

        if self.live_processing:
            self.stop_live_processing()

        values = np.clip(values * PLAYBACK_GAIN, -1.0, 1.0).astype(np.float32)
        print(f"[Playback] Listening gain: {PLAYBACK_GAIN:.1f}x (playback only)")
        self.stop_playback()
        self.playback_stop.clear()
        self.playback_done.clear()
        with self.playback_lock:
            self.playback_buffer = values
            self.playback_position = 0
        output_index = self._selected_output_index()
        if output_index is None:
            self._error("The selected output device is unavailable. Choose a valid output device.")
            return
        self.playback_thread = threading.Thread(target=self._play_worker, args=(output_index,), daemon=True)
        self.playback_thread.start()

    def _selected_output_index(self):
        selection = self.output_var.get()
        if not selection or selection.lower() in {"default output", "default", "no output"}:
            return self._preferred_output_index()
        index = resolve_device_index(selection, self.outputs)
        if index is not None:
            return index
        return self._preferred_output_index()

    def _validate_playback_audio(self, values):
        if not values.size:
            self._error("PLAYBACK ERROR: No recorded audio is available.")
            return False
        if not np.all(np.isfinite(values)):
            self._error("PLAYBACK ERROR: Invalid recorded audio contains non-finite samples.")
            return False
        if float(np.max(np.abs(values))) > 1.0:
            print("[Playback] Clipping samples to float32 range [-1, 1]")
            values[:] = np.clip(values, -1.0, 1.0)
        print(f"[Playback] Samples: {values.size}")
        print(f"[Playback] Duration: {values.size / SAMPLE_RATE:.2f} sec")
        print(f"[Playback] Sample rate: {SAMPLE_RATE} Hz")
        print("[Playback] Channels: 1")
        print(f"[Playback] Sample format: {values.dtype}")
        print(f"[Playback] Min: {float(np.min(values)):.6f}")
        print(f"[Playback] Max: {float(np.max(values)):.6f}")
        print(f"[Playback] RMS: {float(np.sqrt(np.mean(values * values))):.6f}")
        return True

    def _play_worker(self, output_index):
        stream = None
        try:
            normalized_index = normalize_device_index(output_index, kind="output")
            if normalized_index is None:
                normalized_index = get_default_output_index()
            if normalized_index is None:
                device = sd.query_devices(kind="output")
                self.playback_channels = 1
                print("[Playback] No valid output index found; using system default device")
            else:
                device = sd.query_devices(normalized_index, "output")
                max_channels = int(device.get("max_output_channels", 1))
                self.playback_channels = 2 if max_channels >= 2 else 1
                print(f"[Playback] Output device: {device.get('name', 'Default output')}")
            print(f"[Playback] Buffer size: {FRAME_SIZE}")
            print(f"[Playback] Output channels: {self.playback_channels}")
            stream = sd.OutputStream(
                device=normalized_index,
                samplerate=SAMPLE_RATE,
                channels=self.playback_channels,
                dtype="float32",
                blocksize=FRAME_SIZE,
                callback=self._playback_callback,
                finished_callback=self._playback_finished,
            )
            self.playback_stream = stream
            stream.start()
            print("[Playback] Stream opened")
            print("[Playback] Playback started")
            while not self.playback_done.wait(0.05):
                if self.playback_stop.is_set():
                    break
        except Exception as exc:
            print(f"[Playback] ERROR: {exc}")
            self.root.after(0, self._error, f"PLAYBACK ERROR: {exc}. Check the output device and Linux audio stack (PipeWire/ALSA).")
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as cleanup_error:
                    print(f"[Playback] Cleanup ERROR: {cleanup_error}")
                finally:
                    if self.playback_stream is stream:
                        self.playback_stream = None
            print("[Playback] Playback stopped safely")

    def _playback_callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        with self.playback_lock:
            remaining = len(self.playback_buffer) - self.playback_position
            count = min(frames, max(0, remaining))
            if count:
                samples = self.playback_buffer[self.playback_position:self.playback_position + count]
                outdata[:count, :] = samples[:, None]
                self.playback_position += count
            if count < frames:
                self.playback_done.set()

    def _playback_finished(self):
        self.playback_done.set()

    def stop_playback(self):
        self.playback_stop.set()
        self.playback_done.set()
        stream = self.playback_stream
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                print(f"[Playback] Stop ERROR: {exc}")
            finally:
                self.playback_stream = None

    def play_ab(self):
        self.play_audio(self.noisy_audio if self.ab_var.get() == "ORIGINAL" else self.enhanced_audio)

    def _error(self, text):
        self.status_var.set("ERROR")
        self.led.config(fg="#ff5f68")
        try:
            messagebox.showerror("NOISELESS", text)
        except Exception:
            print(text)

    def close(self):
        self.closing = True
        self._close_gpio_buttons()
        if self.recording:
            self.recording = False
            if self.capture_stream is not None:
                self.capture_stream.stop()
                self.capture_stream.close()
                self.capture_stream = None
            if self.capture_worker is not None:
                self.capture_worker.join()
                self.capture_worker = None
        self.stop_playback()
        self.stop_live_processing()
        if self.network_server is not None:
            self.network_server.stop()
        if self.capture_stream is not None:
            self.capture_stream.stop()
            self.capture_stream.close()
        if self.engine is not None:
            self.engine.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-audio", action="store_true")
    args = parser.parse_args()
    if args.test_audio:
        raise SystemExit(test_audio())
    root = tk.Tk()
    NoiselessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
