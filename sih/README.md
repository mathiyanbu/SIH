# AI Real-Time Noise Cancellation Prototype

This project is a small proof-of-concept for real-time speech enhancement on a laptop. It captures audio from the default microphone, processes the signal with a lightweight CNN mask model, and plays the enhanced speech through headphones or speakers.

## What the project does

- Captures live microphone audio in small chunks.
- Converts the signal to mono at 16 kHz.
- Computes a short-time Fourier transform (STFT).
- Uses a tiny deep-learning mask model to suppress background noise.
- Reconstructs enhanced audio and plays it back in real time.
- Provides a terminal menu to list devices, select microphones, and stop the stream.

## System architecture

- Input: default or selected laptop microphone
- Signal path: microphone -> STFT -> lightweight CNN mask -> inverse STFT -> playback
- Output: headphones recommended to avoid acoustic feedback

## Installation

```bash
pip install -r requirements.txt
python main.py
```

## How to select a microphone

When you run the app, choose option 4 from the menu and enter the device index reported by the audio device listing.

## How to run real-time suppression

1. Start the program.
2. Select the correct microphone if needed.
3. Choose option 1 to start noise suppression.
4. Use headphones for testing.
5. Choose option 2 to stop.

## How to train the model

The lightweight model is trained with synthetic speech-plus-noise examples. Run:

```bash
python train.py
```

The training script saves the model at `models/cnn_mask_model.pt`.

## How to test the model

- Record a short noisy sample and compare it with the enhanced output using the comparison option in the menu.
- Listen with headphones to check the quality and noise reduction.
- For a more serious evaluation, add clean reference recordings and compute RMS and SNR estimates.

## Limitations

- This is a lightweight prototype, not a production-grade speech enhancer.
- Synthetic training data cannot replace a large real-world dataset.
- CPU-only inference may be limited on older laptops.
- Real-time processing depends heavily on the laptop’s audio configuration and CPU speed.

## Raspberry Pi 5 deployment path

This prototype is designed to be portable to Raspberry Pi 5 later by:

- exporting the model to ONNX or TensorFlow Lite,
- reducing the FFT size and model width,
- using a lower-latency audio callback,
- optimizing the buffering strategy for low-power hardware.

## Notes

- The requirement to use a default microphone is handled automatically.
- The prototype defaults to 16 kHz mono audio.
- The training pipeline supports fan/AC, traffic, human background, music, and general environmental noise patterns.
