import numpy as np

class DSPEngine:
    def __init__(self, sample_rate=44100, chunk_size=2048):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self.window = np.hanning(chunk_size)

        self.bands = {
            "bass": (20, 120),
            "mids": (120, 4000),
            "treble": (4000, 16000)
        }

        self.freqs = np.fft.rfftfreq(
            chunk_size,
            d=1.0 / sample_rate
        )

    def process_audio(self, audio_chunk):

        if len(audio_chunk) != self.chunk_size:
            return None

        windowed = audio_chunk * self.window

        fft_mag = np.abs(
            np.fft.rfft(windowed)
        )

        energies = {}

        for band_name, (low, high) in self.bands.items():

            mask = (
                (self.freqs >= low) &
                (self.freqs <= high)
            )

            band = fft_mag[mask]

            if len(band) == 0:
                energies[band_name] = 0.0
            else:
                energies[band_name] = np.sqrt(
                    np.mean(band)
                )

        return energies