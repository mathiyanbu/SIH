"""Lightweight training script for the noise-suppression mask model.

This script generates a synthetic speech + noise dataset, trains a tiny CNN to
estimate a magnitude mask, and saves the weights to models/cnn_mask_model.pt.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ai_noise_canceller.ml.model import train_quick_model

MODEL_PATH = Path(__file__).resolve().parent / "models" / "cnn_mask_model.pt"


if __name__ == "__main__":
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[INFO] Training lightweight mask model...")
    train_quick_model(model_path=MODEL_PATH, epochs=8, samples=200)
    print(f"[INFO] Model saved to {MODEL_PATH}")
