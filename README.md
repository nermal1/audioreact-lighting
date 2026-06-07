# AudioReact Lighting 

An industrial-grade, real-time audio-reactive lighting controller built in Python. This system goes beyond simple volume-reactive LEDs by utilizing a high-definition DSP pipeline, a UDP Sidecar architecture, and API chaining to perfectly sync hardware lighting to the actual rhythm and metadata of the music.

## Core Architecture

* **Real-Time DSP Engine:** Uses `numpy` and `sounddevice` to capture audio in 2048-sample chunks (46ms), extracting bass, mid, and treble energies with professional dynamic expansion (AGC) to handle heavily compressed music (e.g., Thrash Metal).
* **Stable Comb Filter Tracking:** Features a highly tuned NumPy comb-filter bank with a moving average buffer and "Thrash Bias" to detect rapid BPMs and prevent half-time octave errors.
* **Metadata API Chaining:** Utilizes an asynchronous state machine to fingerprint live audio, query the AudD API for track identification, and automatically chain to GetSongBPM to retrieve high-definition tempo and key data without freezing the main UI thread.
* **Hardware Integration:** Streams compressed 7-byte UDP packets directly to an Arduino over serial at 115200 baud for ultra-low latency RGB LED control.

## Visualizer GUI
Built with `PyQt6` and `pyqtgraph`, the system features a 60FPS dashboard monitoring:
* Time Domain (Raw Waveform)
* Frequency Domain (AGC Extracted Energies)
* Bass Envelope & Dynamic Floor tracking
* Current System State & API sync status

## Acknowledgments & APIs

This project relies on the following services for real-time music identification and metadata extraction:
* Audio fingerprinting and track identification provided by [AudD](https://audd.io/).
* Tempo and Key metadata provided by [GetSongBPM](https://getsongbpm.com/).
