import sys
import queue
import signal
import struct
import numpy as np
import sounddevice as sd
import serial
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import time
from dsp_engine import DSPEngine

# --- Configuration ---
SAMPLE_RATE  = 44100
CHUNK_SIZE   = 2048
DEVICE_ID    = 2
ARDUINO_PORT = '/dev/cu.usbmodem101' # Adjust if necessary

class RealTimeVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.dsp = DSPEngine(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
        self.audio_queue = queue.Queue()

        self.smoothed = {'bass': 0.0, 'mids': 0.0, 'treble': 0.0}
        self.dynamic_max = {'bass': 0.1, 'mids': 0.1, 'treble': 0.1}

        # --- BEAT DETECTION STATE ---
        self.bass_history = np.zeros(43, dtype=float)  
        self.history_idx = 0
        self.last_beat_time = 0.0
        self.beat_cooldown = 0.25  
        
        # --- STATE MACHINE VARIABLES ---
        self.current_state = "CHILL"
        self.macro_energy = 0.0         # Tracks the long-term momentum (0.0 to 1.0)
        self.last_state_change = 0.0    # Prevents rapid flickering between states

        # --- RGB FLASH STATE ---
        self.current_rgb = [0.0, 0.0, 0.0]
        self.rgb_decay = 0.85  

        # --- Serial ---
        try:
            self.arduino = serial.Serial(ARDUINO_PORT, 115200, timeout=0)
            print(f"Successfully connected to Arduino on {ARDUINO_PORT}")
            time.sleep(2)
        except Exception as e:
            print(f"Warning: Arduino not connected. {e}")
            self.arduino = None

        self.init_ui()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(11)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()

        self.start_audio_stream()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def init_ui(self):
        self.setWindowTitle("State Machine Visualizer")
        self.resize(800, 650)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

        self.waveform_plot = self.win.addPlot(title="Time Domain (Raw Waveform Input)")
        self.waveform_plot.setYRange(-1.0, 1.0)
        self.waveform_plot.setXRange(0, CHUNK_SIZE)
        self.waveform_curve = self.waveform_plot.plot(pen='g')

        self.win.nextRow()

        self.fft_plot = self.win.addPlot(title="Frequency Domain & State Machine")
        self.fft_plot.setYRange(0, 1.0)

        self.bar_x = [1, 2, 3]
        self.bar_graph = pg.BarGraphItem(x=self.bar_x, height=[0, 0, 0], width=0.6, brush='r')
        self.fft_plot.addItem(self.bar_graph)
        
        # --- NEW: State Machine GUI Overlay ---
        self.state_text = pg.TextItem(text="STATE: CHILL | Energy: 0%", color=(255, 255, 255), anchor=(0, 1))
        self.state_text.setPos(0.5, 0.9)
        self.state_text.setFont(pg.QtGui.QFont("Arial", 16, pg.QtGui.QFont.Weight.Bold))
        self.fft_plot.addItem(self.state_text)

        ax = self.fft_plot.getAxis('bottom')
        ax.setTicks([[(1, 'Bass'), (2, 'Mids'), (3, 'Treble')]])

    # ------------------------------------------------------------------ #
    #  Audio
    # ------------------------------------------------------------------ #
    def start_audio_stream(self):
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                device=DEVICE_ID,
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            sys.exit(1)

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata[:, 0].copy())

    # ------------------------------------------------------------------ #
    #  Main update loop
    # ------------------------------------------------------------------ #
    def update_plots(self):
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get_nowait())
        if not chunks:
            return

        latest_chunk = chunks[-1]
        self.waveform_curve.setData(latest_chunk)
        feat = self.dsp.process_audio(latest_chunk)
        if not feat:
            return

        alpha = 0.35
        for k in ("bass", "mids", "treble"):
            self.smoothed[k] = alpha * feat.get(k, 0.0) + (1 - alpha) * self.smoothed[k]

        b, m, t = 0, 0, 0
        bar_heights = []
        
        for i, k in enumerate(("bass", "mids", "treble")):
            self.dynamic_max[k] = max(self.dynamic_max[k] * 0.996, self.smoothed[k])
            safe_max = max(self.dynamic_max[k], 0.01)
            normalized = self.smoothed[k] / safe_max
            expanded = normalized ** 2 
            bar_heights.append(expanded)
            
            clipped_val = int(np.clip(expanded * 254, 0, 254))
            if i == 0: b = clipped_val
            elif i == 1: m = clipped_val
            else: t = clipped_val

        self.bar_graph.setOpts(x=self.bar_x, height=bar_heights, width=0.6, brush="r")

        # --- 1. MACRO-ENERGY TRACKER (The "Vibe" Sensor) ---
        # We use the normalized 'expanded' bass (bar_heights[0]) to track volume-independent momentum.
        # Alpha of 0.98 means it takes about 2-3 seconds to fully react to a new energy level.
        self.macro_energy = (0.98 * self.macro_energy) + (0.02 * bar_heights[0])

        now = time.time()

        # --- 2. STATE MACHINE LOGIC ---
        # We use a 2-second hysteresis (cooldown) so the state doesn't flip back and forth rapidly
        if (now - self.last_state_change) > 2.0:
            if self.macro_energy > 0.55 and self.current_state != "HYPE":
                self.current_state = "HYPE"
                self.last_state_change = now
                self.state_text.setColor((255, 50, 50)) # Turn text Red
                print(f"\n========================================")
                print(f"  ENTERING HYPE STATE ")
                print(f"========================================\n")
            
            elif self.macro_energy < 0.20 and self.current_state != "CHILL":
                self.current_state = "CHILL"
                self.last_state_change = now
                self.state_text.setColor((50, 150, 255)) # Turn text Blue
                print(f"\n========================================")
                print(f" ENTERING CHILL STATE ")
                print(f"========================================\n")

        # Update GUI Text
        self.state_text.setText(f"STATE: {self.current_state} | Energy: {int(self.macro_energy * 100)}%")

        # --- 3. BEAT DETECTION ---
        raw_bass = feat.get("bass", 0.0)
        self.bass_history[self.history_idx] = raw_bass
        self.history_idx = (self.history_idx + 1) % len(self.bass_history)
        
        valid_history = self.bass_history[self.bass_history > 0]
        baseline = np.percentile(valid_history, 50) if len(valid_history) > 5 else raw_bass
        
        is_beat = ((raw_bass > baseline * 1.5) and (raw_bass > 0.05) and ((now - self.last_beat_time) > self.beat_cooldown))
        
        if is_beat:
            self.last_beat_time = now

        # --- 4. STATE-AWARE RGB LOGIC ---
        ambient_r, ambient_g, ambient_b = 0.0, 0.0, 0.0
        
        # Calculate ambient colors normally
        if b > m and b > t:
            ambient_r, ambient_b = b * 0.8, b * 0.1 
        elif m > b and m > t:
            ambient_b, ambient_g = m * 0.9, m * 0.2 
        else:
            ambient_g, ambient_r = t * 0.8, t * 0.2 

        # Modify behavior based on STATE
        if self.current_state == "CHILL":
            self.rgb_decay = 0.95 # Slower, dreamier fades
            drift_speed = 0.05    # Very slow color transitions
            
            # In Chill mode, we suppress violent white flashes.
            # A beat just causes a soft, rich pulse of the dominant color.
            if is_beat:
                self.current_rgb = [ambient_r * 1.2, ambient_g * 1.2, ambient_b * 1.2]
            else:
                self.current_rgb[0] += (ambient_r - self.current_rgb[0]) * drift_speed
                self.current_rgb[1] += (ambient_g - self.current_rgb[1]) * drift_speed
                self.current_rgb[2] += (ambient_b - self.current_rgb[2]) * drift_speed

        elif self.current_state == "HYPE":
            self.rgb_decay = 0.80 # Fast, sharp, aggressive fades
            drift_speed = 0.35    # Fast snapping to new colors
            
            if is_beat:
                if raw_bass > baseline * 2.0:
                    self.current_rgb = [255.0, 255.0, 255.0] # Massive white strobe
                else:
                    if m > t: self.current_rgb = [255.0, 0.0, 100.0] # Aggressive Pink
                    else: self.current_rgb = [0.0, 255.0, 200.0]     # Aggressive Teal
            else:
                self.current_rgb[0] += (ambient_r - self.current_rgb[0]) * drift_speed
                self.current_rgb[1] += (ambient_g - self.current_rgb[1]) * drift_speed
                self.current_rgb[2] += (ambient_b - self.current_rgb[2]) * drift_speed
                
        rgb_r = int(np.clip(self.current_rgb[0], 0, 255))
        rgb_g = int(np.clip(self.current_rgb[1], 0, 255))
        rgb_b = int(np.clip(self.current_rgb[2], 0, 255))

        if self.arduino:
            try:
                packet = struct.pack('<BBBBBBB', 255, b, m, t, rgb_r, rgb_g, rgb_b)
                self.arduino.write(packet)
            except Exception as e:
                pass

    def closeEvent(self, event):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        if self.arduino:
            self.arduino.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    visualizer = RealTimeVisualizer()
    visualizer.show()
    sys.exit(app.exec())