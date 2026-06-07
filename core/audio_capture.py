import sounddevice as sd
import numpy as np

# Configuration
# standard CD quality sample rate 
# we capure 44,100 points per second
SAMPLE_RATE = 44100  # Sample rate in Hz

# chunk size determines latency. 1024 sample at 44,100 Hz means about 23 milliseconds of audio
CHUNK_SIZE = 1024

def audio_callback(indata, frames, time, status):
    """
    This function is called automatically by the audio driver every time a new chunk of 1024 audio samples in ready
    """

    if status:
        print(f"Status: {status}")
    
    # indata is a 2D array. (CHUNK_SIZE, channels)
    # flatten to a 1D for easier math.
    audio_data = indata[:, 0]

    # calculate RMS. A standard way of capturing the volume.
    rms_volume = np.sqrt(np.mean(audio_data**2))

    # Scale up the volume so it easier to see.
    scaled_volume = int(rms_volume * 300)
    print("[]" * scaled_volume)

if __name__ == "__main__":
    print("Starting audio capture. Press Ctrl+C to stop.")

    try:
        # open the audio stream.
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1, # mono audio is all needed for lighting
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            device=1
        )

        with stream:
            while True:
                sd.sleep(1000)  # Keep the stream open
    except KeyboardInterrupt:
        print("Audio capture stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
        


