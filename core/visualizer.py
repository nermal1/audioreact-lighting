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
import socket
from dsp_engine import DSPEngine
from song_identifier import SongIdentifier, IDState

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
SAMPLE_RATE    = 44100
CHUNK_SIZE     = 2048
ARDUINO_PORT   = '/dev/cu.usbmodem101'
DEVICE_NAME    = "BlackHole 2ch"   # change to "BlackHole 16ch" if needed

WAVEFORM_GAIN  = 1.0
NOISE_GATE_RMS = 0.005

# Paste your AudD key here (https://audd.io — free tier: ~300 req/day).
# Leave blank to run in fallback-only mode with no identification.
AUDD_API_KEY   = '8bbb9bc3fb0b1528e85b942d46e7547a'


def _find_blackhole_device(name: str) -> int:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if name.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
            print(f"[Audio] Using device {i}: {dev['name']}")
            return i
    print(f"\n[Audio] ERROR — could not find input device matching '{name}'.")
    print("[Audio] Available input devices:")
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  [{i}] {dev['name']}  (inputs: {dev['max_input_channels']})")
    print(f"\nSet DEVICE_NAME in visualizer.py to one of the names above.\n")
    sys.exit(1)


class RealTimeVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.device_id = _find_blackhole_device(DEVICE_NAME)
        self.dsp       = DSPEngine(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)

        # --- Song Identification ---
        self.song_id = SongIdentifier(api_key=AUDD_API_KEY, sample_rate=SAMPLE_RATE)

        # When LOCKED, beat_cooldown is derived from DB BPM so it stays phase-accurate.
        # When FALLBACK, we use the reactive bass-detection cooldown of 0.25s.
        self.beat_cooldown = 0.25

        self.smoothed    = {'bass': 0.0, 'mids': 0.0, 'treble': 0.0}
        self.dynamic_max = {'bass': 0.1, 'mids': 0.1, 'treble': 0.1}

        self.bass_history  = np.full(43, 0.01, dtype=float)
        self.history_idx   = 0
        self.last_beat_time = 0.0

        self.current_rgb = [0.0, 0.0, 0.0]

        # --- BPM beat LED (green, pin 13) ---
        self.last_bpm_beat_time = 0.0
        self.BPM_FLASH_DURATION = 0.08

        self.bpm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bpm_sock.setblocking(False)
        self.bpm_sock.bind(('127.0.0.1', 5006))

        self.bpm_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bpm_send_sock.setblocking(False)
        self.BPM_SERVER_ADDR = ('127.0.0.1', 5005)

        # --- Serial ---
        self.arduino: serial.Serial | None = None
        try:
            self.arduino = serial.Serial(ARDUINO_PORT, 115200, timeout=0)
            print(f"Connected to Arduino on {ARDUINO_PORT}")
            time.sleep(2)
        except Exception as e:
            print(f"Warning: Arduino not connected — {e}")

        self._init_ui()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(33)   # ~30 FPS
        self.timer.timeout.connect(self._update)
        self.timer.start()

        # Separate slower timer to refresh the song info label (no need for 30 FPS)
        self.meta_timer = QtCore.QTimer()
        self.meta_timer.setInterval(500)
        self.meta_timer.timeout.connect(self._update_meta_label)
        self.meta_timer.start()

        self._start_audio_stream()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def _init_ui(self):
        self.setWindowTitle("Real-Time Audio Reactive System")
        self.resize(900, 750)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # --- Song metadata banner ---
        self.meta_label = QtWidgets.QLabel("◉ Listening for song…")
        self.meta_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.meta_label.setFixedHeight(36)
        self.meta_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #CCCCCC;"
            "background-color: #1a1a2e; border-bottom: 1px solid #333;"
            "padding: 4px;"
        )
        layout.addWidget(self.meta_label)

        # --- BPM source indicator ---
        self.bpm_source_label = QtWidgets.QLabel("BPM SOURCE: reactive")
        self.bpm_source_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.bpm_source_label.setFixedHeight(28)
        self.bpm_source_label.setStyleSheet(
            "font-size: 12px; color: #888888;"
            "background-color: #111111;"
        )
        layout.addWidget(self.bpm_source_label)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

        self.waveform_plot = self.win.addPlot(title="Time Domain (Raw Waveform Input)")
        self.waveform_plot.setYRange(-1.0, 1.0)
        self.waveform_plot.setXRange(0, CHUNK_SIZE)
        self.waveform_plot.setLabel('left',   'Amplitude')
        self.waveform_plot.setLabel('bottom', 'Sample')
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen('g', width=1))

        self.win.nextRow()

        self.fft_plot = self.win.addPlot(title="Frequency Domain (Extracted Feature Energies)")
        self.fft_plot.setYRange(0, 1.0)
        self.fft_plot.setLabel('left', 'Normalised Energy')

        self.bar_x     = [1, 2, 3]
        self.bar_graph = pg.BarGraphItem(x=self.bar_x, height=[0, 0, 0], width=0.6, brush='r')
        self.fft_plot.addItem(self.bar_graph)

        ax = self.fft_plot.getAxis('bottom')
        ax.setTicks([[(1, 'Bass'), (2, 'Mids'), (3, 'Treble')]])

        # --- BPM display ---
        self.bpm_label = QtWidgets.QLabel("BPM: —")
        self.bpm_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.bpm_label.setFixedHeight(60)
        self.bpm_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #00FF00;"
            "background-color: #111111; border: 1px solid #222; border-radius: 8px;"
        )
        layout.addWidget(self.bpm_label)

    # ------------------------------------------------------------------ #
    #  Metadata label update (500 ms timer)
    # ------------------------------------------------------------------ #
    def _update_meta_label(self):
        state  = self.song_id.state
        status = self.song_id.status_line()
        self.meta_label.setText(status)

        # Colour the banner by state
        if state == IDState.LOCKED:
            self.meta_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #00FF88;"
                "background-color: #0d2b1a; border-bottom: 1px solid #1a4a2e; padding: 4px;"
            )
        elif state == IDState.IDENTIFYING:
            self.meta_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #FFD700;"
                "background-color: #2b2500; border-bottom: 1px solid #4a3f00; padding: 4px;"
            )
        elif state in (IDState.FALLBACK, IDState.COOLDOWN):
            self.meta_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #FF8844;"
                "background-color: #2b1500; border-bottom: 1px solid #4a2500; padding: 4px;"
            )
        else:  # LISTENING
            self.meta_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #CCCCCC;"
                "background-color: #1a1a2e; border-bottom: 1px solid #333; padding: 4px;"
            )

        # BPM source indicator
        bpm_override = self.song_id.bpm_override
        if bpm_override is not None:
            self.bpm_source_label.setText("BPM SOURCE: database ✓")
            self.bpm_source_label.setStyleSheet(
                "font-size: 12px; color: #00FF88; background-color: #111111;"
            )
        else:
            self.bpm_source_label.setText("BPM SOURCE: reactive engine")
            self.bpm_source_label.setStyleSheet(
                "font-size: 12px; color: #888888; background-color: #111111;"
            )

    # ------------------------------------------------------------------ #
    #  Audio
    # ------------------------------------------------------------------ #
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Audio] {status}", file=sys.stderr)
        chunk = indata[:, 0].copy()
        if np.sqrt(np.mean(chunk ** 2)) < NOISE_GATE_RMS:
            return
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
        self.audio_queue.put_nowait(chunk)

    def _start_audio_stream(self):
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                device=self.device_id,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as e:
            print(f"Fatal: could not open audio stream — {e}")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Main update loop (~30 FPS)
    # ------------------------------------------------------------------ #
    def _update(self):
        chunk = None
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get_nowait()
        if chunk is None:
            return

        # Feed the identifier (non-blocking — it manages its own thread)
        self.song_id.feed(chunk)

        # Waveform
        display = np.clip(chunk * WAVEFORM_GAIN, -1.0, 1.0)
        self.waveform_curve.setData(display)

        # Forward to bpm_server.ts only when in fallback mode
        if self.song_id.use_fallback:
            try:
                self.bpm_send_sock.sendto(
                    chunk.astype(np.float32).tobytes(), self.BPM_SERVER_ADDR
                )
            except OSError:
                pass

        # DSP
        feat = self.dsp.process_audio(chunk)
        if not feat:
            return

        # 1. Smooth
        alpha = 0.35
        for k in ('bass', 'mids', 'treble'):
            self.smoothed[k] = alpha * feat.get(k, 0.0) + (1 - alpha) * self.smoothed[k]

        # 2. Dynamic normalisation
        b_val = m_val = t_val = 0
        bar_heights = []
        for i, k in enumerate(('bass', 'mids', 'treble')):
            self.dynamic_max[k] = max(self.dynamic_max[k] * 0.996, self.smoothed[k])
            safe_max   = max(self.dynamic_max[k], 0.01)
            expanded   = (self.smoothed[k] / safe_max) ** 2
            bar_heights.append(expanded)
            val = int(np.clip(expanded * 254, 0, 254))
            if   i == 0: b_val = val
            elif i == 1: m_val = val
            else:        t_val = val
        self.bar_graph.setOpts(x=self.bar_x, height=bar_heights, width=0.6, brush='r')

        # 3. Beat detection
        #    LOCKED  → cooldown derived from DB BPM for phase-accurate strobing
        #    FALLBACK → standard reactive cooldown
        raw_bass = float(feat.get('bass', 0.0))
        self.bass_history[self.history_idx] = raw_bass
        self.history_idx = (self.history_idx + 1) % len(self.bass_history)

        valid    = self.bass_history[self.bass_history > 0]
        baseline = float(np.percentile(valid, 50)) if len(valid) > 5 else raw_bass

        bpm_override = self.song_id.bpm_override
        if bpm_override is not None:
            # Phase-lock: cooldown = one beat period at the known BPM
            self.beat_cooldown = 60.0 / bpm_override
            self.bpm_label.setText(f"BPM: {bpm_override:.1f}")
        else:
            self.beat_cooldown = 0.25

        now     = time.monotonic()
        is_beat = (
            raw_bass > baseline * 1.25
            and raw_bass > 0.02
            and (now - self.last_beat_time) > self.beat_cooldown
        )
        if is_beat:
            self.last_beat_time = now

        # 4. RGB ambient + flash
        b_f = b_val / 254; m_f = m_val / 254; t_f = t_val / 254
        ambient_r = ambient_g = ambient_b = 0.0

        if b_f >= m_f and b_f >= t_f:
            ambient_r = b_f * 0.8;  ambient_b = b_f * 0.1
        elif m_f >= b_f and m_f >= t_f:
            ambient_b = m_f * 0.9;  ambient_g = m_f * 0.2
        else:
            ambient_g = t_f * 0.8;  ambient_r = t_f * 0.2

        if is_beat:
            if raw_bass > baseline * 2.0:
                self.current_rgb = [255.0, 255.0, 255.0]
            elif m_val > t_val:
                self.current_rgb = [150.0, 0.0, 255.0]
            else:
                self.current_rgb = [0.0, 200.0, 255.0]
        else:
            drift = 0.3
            self.current_rgb[0] += (ambient_r * 255 - self.current_rgb[0]) * drift
            self.current_rgb[1] += (ambient_g * 255 - self.current_rgb[1]) * drift
            self.current_rgb[2] += (ambient_b * 255 - self.current_rgb[2]) * drift

        rgb_r = int(np.clip(self.current_rgb[0], 0, 255))
        rgb_g = int(np.clip(self.current_rgb[1], 0, 255))
        rgb_b = int(np.clip(self.current_rgb[2], 0, 255))

        # 5. Poll BPM beat pulse from bpm_server.ts (only active in fallback)
        try:
            while True:
                self.bpm_sock.recvfrom(16)
                self.last_bpm_beat_time = now
        except BlockingIOError:
            pass

        bpm_beat_on = (now - self.last_bpm_beat_time) < self.BPM_FLASH_DURATION
        # In LOCKED mode, drive the green LED directly from is_beat instead
        if bpm_override is not None:
            bpm_beat_on = is_beat
        beat_byte = 254 if bpm_beat_on else 0

        # 6. Arduino
        if self.arduino:
            try:
                packet = struct.pack(
                    '<BBBBBBBB',
                    255, b_val, m_val, t_val, rgb_r, rgb_g, rgb_b, beat_byte
                )
                self.arduino.write(packet)
            except serial.SerialException as e:
                print(f"Arduino write error: {e}")
                self.arduino = None

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #
    def closeEvent(self, event):
        self.timer.stop()
        self.meta_timer.stop()
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.bpm_sock.close()
        self.bpm_send_sock.close()
        if self.arduino:
            self.arduino.close()
        event.accept()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    visualizer = RealTimeVisualizer()
    visualizer.show()
    sys.exit(app.exec())