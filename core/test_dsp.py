import numpy as np
from dsp_engine import DSPEngine

def test_dsp():
    # 1. Initialize our engine
    SAMPLE_RATE = 44100
    CHUNK_SIZE = 1024
    dsp = DSPEngine(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)

    # 2. Create an array of 1024 time steps
    # This represents 23 milliseconds of time passing
    t = np.linspace(0, CHUNK_SIZE / SAMPLE_RATE, CHUNK_SIZE, endpoint=False)

    # 3. Generate a pure 60 Hz Bass wave (like a deep EDM kick drum)
    # The formula for a sine wave is: sin(2 * pi * frequency * time)
    pure_bass_wave = np.sin(2 * np.pi * 60 * t)

    # 4. Generate a pure 5000 Hz Treble wave (like a sharp hi-hat)
    pure_treble_wave = np.sin(2 * np.pi * 5000 * t)

    # 5. Run them through the DSP Engine
    print("--- TESTING PURE 60 Hz BASS ---")
    bass_features = dsp.process_audio(pure_bass_wave)
    print(f"Bass Energy:   {bass_features.get('bass'):.4f}  <-- Should be high!")
    print(f"Mids Energy:   {bass_features.get('mids'):.4f}  <-- Should be near 0")
    print(f"Treble Energy: {bass_features.get('treble'):.4f}  <-- Should be near 0\n")

    print("--- TESTING PURE 5000 Hz TREBLE ---")
    treble_features = dsp.process_audio(pure_treble_wave)
    print(f"Bass Energy:   {treble_features.get('bass'):.4f}  <-- Should be near 0")
    print(f"Mids Energy:   {treble_features.get('mids'):.4f}  <-- Should be near 0")
    print(f"Treble Energy: {treble_features.get('treble'):.4f}  <-- Should be high!")

if __name__ == "__main__":
    test_dsp()