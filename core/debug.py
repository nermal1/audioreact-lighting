import sys
import csv
import queue
import signal
import numpy as np
import sounddevice as sd
import time
from dsp_engine import DSPEngine
 
SAMPLE_RATE  = 44100
CHUNK_SIZE   = 512
DEVICE_ID    = 2
OUTPUT_FILE  = "debug_log.csv"
 
# ---- Copy of your current thresholds so we can see what they miss ----
KICK_RATIO   = 1.3   # b > mean * this  → kick
KICK_FLOOR   = 60    # b > this absolute floor → kick
DROP_RATIO   = 1.8   # b > mean * this  → bass drop
DROP_FLOOR   = 120   # b > this absolute floor → bass drop
B_MULT       = 45    # EDM profile multiplier
 
dsp          = DSPEngine(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
audio_queue  = queue.Queue()
bass_history = np.zeros(40)
smoothed     = 0.0
ALPHA        = 0.50
 
last_beat_time = 0.0
beat_deltas    = []
current_bpm    = 0.0
 
rows = []
start_time = time.time()
 
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())
 
def process():
    global smoothed, bass_history, last_beat_time, beat_deltas, current_bpm
 
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        blocksize=CHUNK_SIZE, device=DEVICE_ID,
        callback=audio_callback,
    )
    stream.start()
    print(f"Recording to {OUTPUT_FILE} — play your song now. Ctrl+C to stop.\n")
 
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'elapsed_s', 'raw_bass', 'smoothed_b', 'scaled_b',
            'mean_history', 'kick_ratio_needed', 'drop_ratio_needed',
            'is_kick', 'is_drop', 'beat_bpm'
        ])
 
        try:
            while True:
                if audio_queue.empty():
                    time.sleep(0.001)
                    continue
 
                chunk = audio_queue.get_nowait()
                feat  = dsp.process_audio(chunk)
                if not feat:
                    continue
 
                raw_bass = feat.get('bass', 0.0)
                smoothed = ALPHA * raw_bass + (1 - ALPHA) * smoothed
                scaled_b = int(min(max(smoothed * B_MULT, 0), 254))
 
                if scaled_b > 10:
                    bass_history = np.roll(bass_history, -1)
                    bass_history[-1] = scaled_b
 
                mean_bass = float(np.mean(bass_history)) if np.any(bass_history) else 1.0
 
                is_drop = (scaled_b > mean_bass * DROP_RATIO) and (scaled_b > DROP_FLOOR)
                is_kick = (scaled_b > mean_bass * KICK_RATIO) and (scaled_b > KICK_FLOOR) and not is_drop
 
                # BPM tracking
                if is_drop or is_kick:
                    now = time.time()
                    dt  = now - last_beat_time
                    last_beat_time = now
                    if 0.3 <= dt <= 1.0:
                        beat_deltas.append(dt)
                        if len(beat_deltas) > 8:
                            beat_deltas.pop(0)
                        current_bpm = 60.0 / (sum(beat_deltas) / len(beat_deltas))
                    elif dt > 2.0:
                        beat_deltas = []
                        current_bpm = 0.0
 
                elapsed = time.time() - start_time
                rows.append([
                    f"{elapsed:.3f}",
                    f"{raw_bass:.5f}",
                    f"{smoothed:.5f}",
                    scaled_b,
                    f"{mean_bass:.2f}",
                    f"{mean_bass * KICK_RATIO:.2f}",
                    f"{mean_bass * DROP_RATIO:.2f}",
                    1 if is_kick else 0,
                    1 if is_drop else 0,
                    f"{current_bpm:.1f}",
                ])
                writer.writerow(rows[-1])
 
                # Live console so you can see it in real time
                flag = " KICK" if is_kick else (" DROP" if is_drop else "     ")
                print(f"t={elapsed:6.2f}s  b={scaled_b:3}  mean={mean_bass:5.1f}"
                      f"  need>{mean_bass*KICK_RATIO:5.1f}{flag}  BPM={current_bpm:5.1f}")
 
        except KeyboardInterrupt:
            print(f"\nSaved {len(rows)} frames to {OUTPUT_FILE}")
            stream.stop()
            stream.close()
 
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    process()