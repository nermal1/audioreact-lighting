import sys
import queue
import struct
import numpy as np
import sounddevice as sd
import serial
import socket
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import time
from dsp_engine import DSPEngine

# --- Configuration ---
SAMPLE_RATE  = 44100
CHUNK_SIZE   = 2048
DEVICE_ID    = 2
ARDUINO_PORT = '/dev/cu.usbmodem101'

# ------------------------------------------------------------------ #
#  BPM Visualizer GUI (UDP Sidecar Architecture)
# ------------------------------------------------------------------ #
class BPMVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.dsp = DSPEngine(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)

        # --- UDP Microservice Setup ---
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.ts_server_address = ('127.0.0.1', 5005)

        # --- BPM Smoothing ---
        self.bpm_buffer: list[float] = []
        self.BPM_BUFFER_SIZE = 15       # Reduced from 30 for faster response
        self.current_bpm = 0.0

        # --- Feature History ---
        self.HISTORY_SIZE = 300
        self.bass_history     = np.zeros(self.HISTORY_SIZE)
        self.baseline_history = np.zeros(self.HISTORY_SIZE)
        self.beat_markers     = np.zeros(self.HISTORY_SIZE)

        self.last_beat_time = 0.0

        # --- LED State ---
        self.current_rgb = [0.0, 0.0, 0.0]

        # --- Serial Connection ---
        self.arduino: serial.Serial | None = None
        try:
            self.arduino = serial.Serial(ARDUINO_PORT, 115200, timeout=0)
            print(f"Connected to Arduino on {ARDUINO_PORT}")
            time.sleep(2)
        except Exception as e:
            print(f"Warning: Arduino not connected — {e}")

        self._init_ui()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(45)  # ~22 FPS
        self.timer.timeout.connect(self._update)
        self.timer.start()

        self._start_audio_stream()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def _init_ui(self):
        self.setWindowTitle("Industrial Audio & Rhythm Signal Core")
        self.resize(1000, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

        # Waveform
        self.waveform_plot = self.win.addPlot(title="Raw Audio Waveform")
        self.waveform_plot.setYRange(-0.5, 0.5)
        self.waveform_plot.setXRange(0, CHUNK_SIZE)
        self.waveform_curve = self.waveform_plot.plot(pen='g')

        self.win.nextRow()

        # Bass / beat monitor
        self.monitor_plot = self.win.addPlot(title="Bass Envelope & BPM Sync")
        self.monitor_plot.setYRange(0, 1.0)
        self.monitor_plot.setXRange(0, self.HISTORY_SIZE)

        self.bass_curve     = self.monitor_plot.plot(pen=pg.mkPen('w', width=2), name="Bass Energy")
        self.baseline_curve = self.monitor_plot.plot(
            pen=pg.mkPen('b', width=1, style=QtCore.Qt.PenStyle.DashLine), name="Dynamic Floor"
        )
        self.beat_scatter = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 220)
        )
        self.monitor_plot.addItem(self.beat_scatter)

        # BPM label
        self.bpm_label = QtWidgets.QLabel("SYSTEM BPM: 0")
        self.bpm_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.bpm_label.setFixedHeight(80)
        self.bpm_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #00FF00;"
            "background-color: #111111; border: 2px solid #222222; border-radius: 10px;"
        )
        layout.addWidget(self.bpm_label)

    # ------------------------------------------------------------------ #
    #  Audio
    # ------------------------------------------------------------------ #
    def _audio_callback(self, indata, frames, t, status):
        """Drop oldest chunk if queue is full to avoid growing lag."""
        chunk = indata[:, 0].copy()
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
        self.audio_queue.put_nowait(chunk)

    def _start_audio_stream(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            device=DEVICE_ID,
            callback=self._audio_callback,
        )
        self.stream.start()

    # ------------------------------------------------------------------ #
    #  Main update loop
    # ------------------------------------------------------------------ #
    def _update(self):
        if self.audio_queue.empty():
            return

        # Drain to the latest chunk — skip stale ones
        chunk = None
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get_nowait()
        if chunk is None:
            return

        self.waveform_curve.setData(chunk)

        feat = self.dsp.process_audio(chunk)
        if not feat:
            return

        raw_bass   = float(feat.get("bass",   0.0))
        raw_mids   = float(feat.get("mids",   0.0))
        raw_treble = float(feat.get("treble", 0.0))

        # --- Send chunk to Node.js BPM service ---
        try:
            self.sock.sendto(chunk.astype(np.float32).tobytes(), self.ts_server_address)
        except OSError:
            pass  # Expected when Node is not running

        # --- Receive BPM reply ---
        try:
            data, _ = self.sock.recvfrom(1024)
            if len(data) == 4:
                new_bpm = struct.unpack('<f', data)[0]
                if 40.0 < new_bpm < 220.0:  # Sanity clamp
                    self.bpm_buffer.append(new_bpm)
                    if len(self.bpm_buffer) > self.BPM_BUFFER_SIZE:
                        self.bpm_buffer.pop(0)
                    self.current_bpm = sum(self.bpm_buffer) / len(self.bpm_buffer)
        except BlockingIOError:
            pass  # No reply this tick — fine

        bpm = int(self.current_bpm)

        # --- Roll history buffers ---
        self.bass_history     = np.roll(self.bass_history, -1)
        self.bass_history[-1] = raw_bass

        self.beat_markers = np.roll(self.beat_markers, -1)
        self.beat_markers[-1] = 0.0

        # Dynamic baseline using the 30th percentile of recent bass
        valid = self.bass_history[self.bass_history > 0]
        baseline = float(np.percentile(valid, 30)) if len(valid) > 5 else raw_bass

        self.baseline_history     = np.roll(self.baseline_history, -1)
        self.baseline_history[-1] = baseline

        # --- Beat detection ---
        now = time.monotonic()  # More stable than time.time() for intervals
        is_beat = (
            raw_bass > baseline * 1.2
            and raw_bass > 0.04
            and (now - self.last_beat_time) > 0.22
        )
        if is_beat:
            self.last_beat_time = now
            self.beat_markers[-1] = raw_bass
            print(f"\r[BEAT] Bass: {raw_bass:.3f} | BPM: {bpm}      ", end="", flush=True)

        # --- Update plots ---
        self.bass_curve.setData(self.bass_history)
        self.baseline_curve.setData(self.baseline_history)

        beat_x = np.where(self.beat_markers > 0)[0]
        beat_y = self.beat_markers[beat_x]
        self.beat_scatter.setData(x=beat_x.tolist(), y=beat_y.tolist())

        self.bpm_label.setText(f"SYSTEM BPM: {bpm}")

        # --- Hardware ---
        self._send_arduino(raw_bass, raw_mids, raw_treble, is_beat, now)

    # ------------------------------------------------------------------ #
    #  Arduino
    # ------------------------------------------------------------------ #
    def _send_arduino(self, raw_bass, raw_mids, raw_treble, is_beat, now):
        if self.arduino is None:
            return
        try:
            b = 254 if is_beat else 0
            m = int(np.clip(raw_mids * 550, 0, 254))
            t = 254 if (now - self.last_beat_time) < 0.05 else 0

            if is_beat:
                self.current_rgb = [255.0, 255.0, 255.0]
            else:
                lerp = 0.3
                self.current_rgb[0] += (min(raw_bass   * 200, 120) - self.current_rgb[0]) * lerp
                self.current_rgb[1] += (min(raw_treble * 900, 254) - self.current_rgb[1]) * lerp
                self.current_rgb[2] += (min(raw_mids   * 600, 254) - self.current_rgb[2]) * lerp

            r, g, b_led = (int(np.clip(c, 0, 255)) for c in self.current_rgb)
            packet = struct.pack('<BBBBBBB', 255, b, m, t, r, g, b_led)
            self.arduino.write(packet)
        except serial.SerialException as e:
            print(f"\nArduino write error: {e}")
            self.arduino = None  # Stop retrying after a hard failure

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #
    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.sock.close()
        if self.arduino:
            self.arduino.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    visualizer = BPMVisualizer()
    visualizer.show()
    sys.exit(app.exec())