"""NOISELESS record-first RNNoise speech enhancement prototype."""

from __future__ import annotations

import argparse
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

from ai_noise_canceller.audio.microphone import list_audio_devices
from noise_suppression.rnnoise_engine import RNNoiseEngine, RNNoiseLoadError

ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
SAMPLE_RATE = 48000
FRAME_SIZE = 480
PLAYBACK_GAIN = 2.0


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
        self.ai_enabled = True
        self.input_overflows = 0
        self.processing_errors = 0
        self.frames_processed = 0
        self.zero_output_frames = 0
        self.nonzero_output_frames = 0
        self.process_time_ms = 0.0
        self.engine = None
        self.engine_error = ""
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
        self.output_var = tk.StringVar(value=self._device_label(self.outputs[0]) if self.outputs else "Default output")
        self.ab_var = tk.StringVar(value="ORIGINAL")

        self._build_ui()
        self._wave_stacked = False
        self.root.bind("<Configure>", self._on_resize)
        self.root.after(40, self._ui_tick)

    @staticmethod
    def _device_label(device):
        return f"{device['index']}: {device['name']}"

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
        ttk.Combobox(row, textvariable=self.mic_var, values=[self._device_label(d) for d in self.inputs] or ["No microphone detected"], state="readonly", width=25).pack(side="left")
        self.record_start_button = tk.Button(row, text="START RECORDING", command=self.start_recording, bg="#e84b54", fg="white", bd=0, padx=10, pady=7, font=("Arial", 10, "bold"))
        self.record_start_button.pack(side="left", padx=(8, 4))
        tk.Button(row, text="STOP", command=self.stop_recording, bg="#203f5e", fg="white", bd=0, padx=12, pady=7, font=("Arial", 10, "bold")).pack(side="left")
        tk.Label(panel, text="Recording is stored as real microphone audio", fg="#89a9c4", bg="#0d2942", font=("Arial", 9)).pack(anchor="w", padx=12, pady=(8, 10))

    def _process_controls(self, panel):
        self.process_button = tk.Button(panel, text="PROCESS WITH RNNOISE", command=self.process_audio, bg="#2e9ee8", fg="white", bd=0, padx=14, pady=9, font=("Arial", 11, "bold"))
        self.process_button.pack(anchor="w", padx=12, pady=(2, 6))
        tk.Label(panel, textvariable=self.engine_status_var, fg="#36db91", bg="#0d2942", font=("Arial", 10, "bold")).pack(anchor="w", padx=12)
        tk.Label(panel, text="Only recorded audio is processed in this mode", fg="#89a9c4", bg="#0d2942", font=("Arial", 9)).pack(anchor="w", padx=12, pady=(4, 10))

    def _playback_controls(self, panel):
        row = tk.Frame(panel, bg="#0d2942")
        row.pack(fill="x", padx=10)
        tk.Button(row, text="PLAY ORIGINAL", command=lambda: self.play_audio(self.noisy_audio), bg="#b74754", fg="white", bd=0, padx=9, pady=7, font=("Arial", 10, "bold")).pack(side="left", padx=(0, 4))
        tk.Button(row, text="PLAY ENHANCED", command=lambda: self.play_audio(self.enhanced_audio), bg="#229d70", fg="white", bd=0, padx=9, pady=7, font=("Arial", 10, "bold")).pack(side="left", padx=4)
        tk.Button(row, text="STOP AUDIO", command=self.stop_playback, bg="#203f5e", fg="white", bd=0, padx=9, pady=7, font=("Arial", 10, "bold")).pack(side="left", padx=4)
        ab = tk.Frame(panel, bg="#0d2942")
        ab.pack(anchor="w", padx=12, pady=(9, 10))
        tk.Label(ab, text="A / B:", fg="#91aec9", bg="#0d2942", font=("Arial", 10, "bold")).pack(side="left")
        tk.Radiobutton(ab, text="ORIGINAL", variable=self.ab_var, value="ORIGINAL", command=self.play_ab, bg="#0d2942", fg="#eef7ff", selectcolor="#1b4162", activebackground="#0d2942").pack(side="left")
        tk.Radiobutton(ab, text="ENHANCED", variable=self.ab_var, value="ENHANCED", command=self.play_ab, bg="#0d2942", fg="#eef7ff", selectcolor="#1b4162", activebackground="#0d2942").pack(side="left")

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
        self.root.after(40, self._ui_tick)

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

    def _capture_callback(self, indata, frames, time_info, status):
        if not self.recording:
            return
        try:
            self.capture_queue.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            self.input_overflows += 1

    def start_recording(self):
        if self.recording:
            return
        if not self.inputs:
            self._error("No microphone is available.")
            return
        try:
            device_index = int(self.mic_var.get().split(":", 1)[0])
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
            self._error(f"Recording start failed: {exc}")

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

    def _show_recorded_audio(self):
        self._draw_wave(self.input_wave, self.noisy_audio)
        self.input_level_var.set(f"{dbfs(self.noisy_audio):.1f} dB")
        self.input_rms_var.set(f"{dbfs(self.noisy_audio):.1f} dB")
        self.input_peak_var.set(f"{peak_dbfs(self.noisy_audio):.1f} dB")
        self.duration_original_var.set(f"Duration: {len(self.noisy_audio) / SAMPLE_RATE:.2f} s")

    def process_audio(self):
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
        values = np.clip(values * PLAYBACK_GAIN, -1.0, 1.0).astype(np.float32)
        print(f"[Playback] Listening gain: {PLAYBACK_GAIN:.1f}x (playback only)")
        self.stop_playback()
        self.playback_stop.clear()
        self.playback_done.clear()
        with self.playback_lock:
            self.playback_buffer = values
            self.playback_position = 0
        output_index = self._selected_output_index()
        self.playback_thread = threading.Thread(target=self._play_worker, args=(output_index,), daemon=True)
        self.playback_thread.start()

    def _selected_output_index(self):
        if self.output_var.get() and ":" in self.output_var.get():
            try:
                return int(self.output_var.get().split(":", 1)[0])
            except ValueError:
                return None
        return None

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
        try:
            device = sd.query_devices(output_index, "output") if output_index is not None else sd.query_devices(kind="output")
            max_channels = int(device.get("max_output_channels", 1))
            self.playback_channels = 2 if max_channels >= 2 else 1
            print(f"[Playback] Output device: {device.get('name', 'Default output')}")
            print(f"[Playback] Buffer size: {FRAME_SIZE}")
            print(f"[Playback] Output channels: {self.playback_channels}")
            self.playback_stream = sd.OutputStream(
                device=output_index,
                samplerate=SAMPLE_RATE,
                channels=self.playback_channels,
                dtype="float32",
                blocksize=FRAME_SIZE,
                callback=self._playback_callback,
                finished_callback=self._playback_finished,
            )
            self.playback_stream.start()
            print("[Playback] Stream opened")
            print("[Playback] Playback started")
            while not self.playback_done.wait(0.05):
                if self.playback_stop.is_set():
                    break
            if self.playback_stream is not None:
                self.playback_stream.stop()
                self.playback_stream.close()
                self.playback_stream = None
            print("[Playback] Playback stopped safely")
        except Exception as exc:
            self.playback_stream = None
            print(f"[Playback] ERROR: {exc}")
            self.root.after(0, self._error, f"PLAYBACK ERROR: {exc}")

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
        self.recording = False
        self.stop_playback()
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
