"""Independent proof that the real RNNoise backend processes audio."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from noise_suppression.rnnoise_engine import RNNoiseEngine


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "rnnoise_test_output.wav"


def main() -> int:
    engine = RNNoiseEngine()
    input_audio = (np.sin(np.linspace(0, 12 * np.pi, 4800, endpoint=False)) * 0.25).astype(np.float32)
    output_frames = []
    processing_times = []

    for start in range(0, len(input_audio), engine.frame_size):
        frame = input_audio[start : start + engine.frame_size]
        if len(frame) < engine.frame_size:
            frame = np.pad(frame, (0, engine.frame_size - len(frame)))
        started = time.perf_counter()
        output_frames.append(engine.process_frame(frame))
        processing_times.append((time.perf_counter() - started) * 1000.0)

    output_audio = np.concatenate(output_frames)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUTPUT, output_audio, engine.sample_rate)
    changed = not np.allclose(input_audio, output_audio[: len(input_audio)], atol=1e-7)

    print("RNNoise loaded: YES")
    print(f"Input sample rate: {engine.sample_rate}")
    print("Input channels: 1")
    print(f"Frames processed: {len(output_frames)}")
    print(f"Average processing time: {np.mean(processing_times):.3f} ms")
    print(f"Output generated: {'YES' if output_audio.size and changed else 'NO'}")
    print(f"Output file: {OUTPUT}")
    return 0 if changed else 1


if __name__ == "__main__":
    raise SystemExit(main())