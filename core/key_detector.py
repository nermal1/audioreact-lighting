import sys
import queue
import signal
import numpy as np
import sounddevice as sd
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

# --- Configuration ---
SAMPLE_RATE  = 44100
CHUNK_SIZE   = 4096     # Large chunk size required for high-resolution pitch detection
DEVICE_ID    = 2

# ------------------------------------------------------------------ #
#  Pure NumPy Krumhansl-Schmuckler Key Detector
# ------------------------------------------------------------------ #
class KSKeyDetector:
    def __init__(self, sample_rate, fft_size):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        
        # Standard Krumhansl-Schmuckler pitch-class profiles (How often a note appears in a key)
        self.maj_prof = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.min_prof = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Pre-roll the arrays to create 24 target matrices (12 Major, 12 Minor)
        self.maj_matrix = np.array([np.roll(self.maj_prof, i) for i in range(12)])
        self.min_matrix = np.array([np.roll(self.min_prof, i) for i in range(12)])
        
        # Z-Score the matrices for Pearson Correlation math
        self.maj_matrix = (self.maj_matrix - np.mean(self.maj_prof)) / np.std(self.maj_prof)
        self.min_matrix = (self.min_matrix - np.mean(self.min_prof)) / np.std(self.min_prof)

        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Map every FFT frequency bin to a chromatic musical note
        freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)
        self.bin_to_pc = np.full(len(freqs), -1, dtype=int)
        for i, f in enumerate(freqs):
            if 40 < f < 5000: # Restrict to human musical range (Ignore sub-bass rumble and high-hat hiss)
                note_num = int(round(12 * np.log2(f / 440.0) + 69))
                self.bin_to_pc[i] = note_num % 12
                
        self.chromagram_history = np.zeros(12)
        
    def process_chunk(self, audio_chunk):
        # Window the audio to prevent spectral leakage
        window = np.hanning(len(audio_chunk))
        spectrum = np.abs(np.fft.rfft(audio_chunk * window))
        
        # Fold the entire frequency spectrum into 12 "Pitch Class" buckets
        current_chroma = np.zeros(12)
        for i in range(12):
            current_chroma[i] = np.sum(spectrum[self.bin_to_pc == i])
            
        # Keep a rolling ~3 second average of the notes playing
        self.chromagram_history = (self.chromagram_history * 0.90) + (current_chroma * 0.10)
        
        detected_key = "Analyzing..."
        confidence = 0.0
        
        if np.sum(self.chromagram_history) > 0:
            # Z-Score the live audio data
            c_norm = (self.chromagram_history - np.mean(self.chromagram_history)) / (np.std(self.chromagram_history) + 1e-6)
            
            # Cross-Correlate against all 24 possible keys instantly
            maj_corrs = np.dot(self.maj_matrix, c_norm) / 12.0
            min_corrs = np.dot(self.min_matrix, c_norm) / 12.0
            
            best_maj_idx = np.argmax(maj_corrs)
            best_min_idx = np.argmax(min_corrs)
            
            # Which key had the highest mathematical alignment?
            if maj_corrs[best_maj_idx] > min_corrs[best_min_idx]:
                detected_key = f"{self.notes[best_maj_idx]} Major"
                confidence = maj_corrs[best_maj_idx] * 100
            else:
                detected_key = f"{self.notes[best_min_idx]} Minor"
                confidence = min_corrs[best_min_idx] * 100
                
        return detected_key, confidence, self.chromagram_history

# ------------------------------------------------------------------ #
#  Key Visualizer GUI
# ------------------------------------------------------------------ #
class KeyVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.key_detector = KSKeyDetector(SAMPLE_RATE, CHUNK_SIZE)
        self.audio_queue = queue.Queue()

        self.init_ui()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(20) 
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()

        self.start_audio_stream()

    def init_ui(self):
        self.setWindowTitle("Harmonic Analysis & Key Detection")
        self.resize(1000, 600)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

        # Plot 1: The Chromagram
        self.chroma_plot = self.win.addPlot(title="Live Chromagram (Pitch Class Profile)")
        self.chroma_plot.setYRange(0, 1.0)
        
        self.bar_x = np.arange(12)
        self.bar_graph = pg.BarGraphItem(x=self.bar_x, height=np.zeros(12), width=0.8, brush='c')
        self.chroma_plot.addItem(self.bar_graph)
        
        # Add the note names to the X-Axis
        ax = self.chroma_plot.getAxis('bottom')
        ticks = [[(i, note) for i, note in enumerate(self.key_detector.notes)]]
        ax.setTicks(ticks)

        # Plot 2: Output Console
        self.key_label = QtWidgets.QLabel("DETECTED KEY: -- | Confidence: 0%")
        self.key_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.key_label.setFixedHeight(100)
        self.key_label.setStyleSheet("""
            font-size: 48px; 
            font-weight: bold; 
            color: #00FFFF; 
            background-color: #111111; 
            border: 2px solid #333333;
            border-radius: 10px;
        """)
        layout.addWidget(self.key_label)

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

    def update_plots(self):
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get_nowait())
        if not chunks:
            return

        latest_chunk = chunks[-1]
        
        # Feed the raw audio straight into the Key Detector
        key_str, confidence, chromagram = self.key_detector.process_chunk(latest_chunk)

        # Normalize the chromagram array from 0.0 to 1.0 purely for the GUI visualization
        if np.max(chromagram) > 0:
            gui_chroma = chromagram / np.max(chromagram)
        else:
            gui_chroma = chromagram

        self.bar_graph.setOpts(height=gui_chroma)
        
        # Change color based on confidence rating
        if confidence > 70:
            color = "#00FF00" # Green for strong lock
        elif confidence > 40:
            color = "#FFFF00" # Yellow for transition
        else:
            color = "#FF0000" # Red for unsure/dissonance
            
        self.key_label.setStyleSheet(f"""
            font-size: 48px; 
            font-weight: bold; 
            color: {color}; 
            background-color: #111111; 
            border: 2px solid #333333;
            border-radius: 10px;
        """)
        
        self.key_label.setText(f"{key_str} | Conf: {confidence:.1f}%")

    def closeEvent(self, event):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    visualizer = KeyVisualizer()
    visualizer.show()
    sys.exit(app.exec())