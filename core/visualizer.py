import sys
import os
import queue
import signal
import struct
import threading
import numpy as np
import sounddevice as sd
import serial
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import time
import socket
import json
import sqlite3
from dotenv import load_dotenv

from dsp_engine import DSPEngine
from song_identifier import SongIdentifier, IDState

load_dotenv()

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
SAMPLE_RATE       = 44100
CHUNK_SIZE        = 2048
ARDUINO_PORT      = '/dev/cu.usbmodem101'
DEVICE_NAME       = "BlackHole 2ch"   

WAVEFORM_GAIN     = 1.0
NOISE_GATE_RMS    = 0.005

AUDD_API_KEY      = os.getenv('AUDD_API_KEY')
PRESETS_FILE      = "presets.json"
DB_FILE           = "local_library.sqlite"

# Base default presets
DEFAULT_PRESETS = {
    "EDM (Default)": {"bpm": 128.0, "strong": [255, 50, 255], "mid": [100, 0, 255], "high": [0, 150, 255]},
    "Rock (Default)": {"bpm": 140.0, "strong": [255, 255, 255], "mid": [255, 0, 0], "high": [200, 50, 50]},
    "Pop (Default)": {"bpm": 115.0, "strong": [255, 105, 180], "mid": [255, 20, 147], "high": [255, 182, 193]},
    "Country (Default)": {"bpm": 90.0, "strong": [255, 100, 0], "mid": [255, 0, 0], "high": [0, 255, 0]}
}

def _find_blackhole_device(name: str) -> int:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if name.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
            print(f"[Audio] Using device {i}: {dev['name']}")
            return i
    print(f"\n[Audio] ERROR — could not find input device matching '{name}'.")
    sys.exit(1)


class RealTimeVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.device_id = _find_blackhole_device(DEVICE_NAME)
        self.dsp       = DSPEngine(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=10)

        self.song_id = SongIdentifier(api_key=AUDD_API_KEY, sample_rate=SAMPLE_RATE)

        # Basic Tracking Variables
        self.beat_cooldown = 0.25
        self.smoothed    = {'bass': 0.0, 'mids': 0.0, 'treble': 0.0}
        self.dynamic_max = {'bass': 0.1, 'mids': 0.1, 'treble': 0.1}
        self.bass_history  = np.full(43, 0.01, dtype=float)
        self.history_idx   = 0
        self.last_beat_time = 0.0
        self.current_rgb = [0.0, 0.0, 0.0]
        
        # Live Brightness Control
        self.global_brightness = 0.5 

        # Reactive Math
        self.last_bpm_beat_time = 0.0
        self.estimated_reactive_bpm = 0.0
        self.BPM_FLASH_DURATION = 0.10  

        # Manual Mode States
        self.mode = "AUTO"  
        self.manual_bpm = 120.0
        self.manual_colors = {"strong": [255.0, 255.0, 255.0], "mid": [150.0, 150.0, 150.0], "high": [50.0, 50.0, 50.0]}
        self.current_manual_meta = {} # Holds the info for the 'i' button
        self.presets = self._load_presets()

        # Database Setup
        self._init_db()

        # Networking
        self.bpm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bpm_sock.setblocking(False)
        self.bpm_sock.bind(('127.0.0.1', 5006))

        self.bpm_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bpm_send_sock.setblocking(False)
        self.BPM_SERVER_ADDR = ('127.0.0.1', 5005)

        # Serial
        self.arduino: serial.Serial | None = None
        try:
            self.arduino = serial.Serial(ARDUINO_PORT, 115200, timeout=0)
            print(f"Connected to Arduino on {ARDUINO_PORT}")
            time.sleep(2)
        except Exception as e:
            print(f"Warning: Arduino not connected — {e}")

        # Initialization
        self._init_ui()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(33)   
        self.timer.timeout.connect(self._update)
        self.timer.start()

        self.meta_timer = QtCore.QTimer()
        self.meta_timer.setInterval(500)
        self.meta_timer.timeout.connect(self._update_meta_label)
        self.meta_timer.start()

        self._start_audio_stream()

    # ------------------------------------------------------------------ #
    #  Database Management
    # ------------------------------------------------------------------ #
    def _init_db(self):
        self.db_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                artist TEXT,
                title TEXT,
                bpm REAL,
                key_of TEXT,
                time_sig TEXT,
                danceability INTEGER,
                acousticness INTEGER,
                genres TEXT,
                PRIMARY KEY (artist, title)
            )
        """)
        self.db_conn.commit()

    def _update_completer_list(self):
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT artist, title FROM songs ORDER BY artist, title")
        rows = cursor.fetchall()
        self.song_list = [f"{r[0]} - {r[1]}" for r in rows]
        if hasattr(self, 'completer_model'):
            self.completer_model.setStringList(self.song_list)

    # ------------------------------------------------------------------ #
    #  Preset & API Management
    # ------------------------------------------------------------------ #
    def _load_presets(self):
        loaded = DEFAULT_PRESETS.copy()
        if os.path.exists(PRESETS_FILE):
            try:
                with open(PRESETS_FILE, 'r') as f:
                    user_presets = json.load(f)
                    loaded.update(user_presets)
            except Exception as e:
                print(f"Error loading presets: {e}")
        return loaded

    def _save_preset(self):
        name = self.preset_name_input.text().strip()
        if not name: return
        new_preset = {
            "bpm": self.manual_bpm_spin.value(),
            "strong": self.manual_colors["strong"],
            "mid": self.manual_colors["mid"],
            "high": self.manual_colors["high"]
        }
        self.presets[name] = new_preset
        custom_only = {k: v for k, v in self.presets.items() if "(Default)" not in k}
        try:
            with open(PRESETS_FILE, 'w') as f:
                json.dump(custom_only, f, indent=4)
        except Exception as e:
            print(f"Error saving preset: {e}")
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(self.presets.keys())
        self.preset_combo.setCurrentText(name)
        self.preset_combo.blockSignals(False)
        self.preset_name_input.clear()

    def _load_song_action(self):
        query = self.api_search_input.text().strip()
        if not query: return
            
        parts = query.split("-", 1)
        if len(parts) == 2:
            artist, title = parts[0].strip(), parts[1].strip()
        else:
            artist, title = "Unknown", query

        # 1. Search Local Database First
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM songs WHERE artist=? COLLATE NOCASE AND title=? COLLATE NOCASE", (artist, title))
        row = cursor.fetchone()
        
        if row:
            # Load instantly from Hard Drive
            meta = {
                "bpm": row[2],
                "key_of": row[3],
                "time_sig": row[4],
                "danceability": row[5],
                "acousticness": row[6],
                "genres": json.loads(row[7]) if row[7] else []
            }
            self._apply_fetched_metadata(meta, query, save_to_db=False, artist=row[0], title=row[1])
            return

        # 2. Call API if not found locally
        self.meta_label.setText(f"⟳ Fetching metadata for '{query}'...")
        self.bpm_source_label.setText("BPM SOURCE: Contacting GetSongBPM...")
        
        def worker():
            meta = self.song_id._fetch_getsongbpm_fallback(title, artist)
            QtCore.QMetaObject.invokeMethod(
                self, 
                "_apply_fetched_metadata", 
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(object, meta),
                QtCore.Q_ARG(str, query),
                QtCore.Q_ARG(bool, True),
                QtCore.Q_ARG(str, artist),
                QtCore.Q_ARG(str, title)
            )
        threading.Thread(target=worker, daemon=True).start()

    @QtCore.pyqtSlot(object, str, bool, str, str)
    def _apply_fetched_metadata(self, meta, query, save_to_db, artist, title):
        if not meta:
            self.bpm_source_label.setText("❌ Not found in API or Local DB")
            self.meta_label.setText("⚙️ MANUAL OVERRIDE ACTIVE")
            return
            
        bpm = meta.get("bpm", 0.0)
        if bpm > 0:
            self.manual_bpm_spin.setValue(bpm)
            
        genres = meta.get("genres", [])
        genre_str = ", ".join(genres[:2]) if genres else "Unknown Genre"
        
        # Save to Current Context for the 'i' button
        self.current_manual_meta = meta
        self.current_manual_meta['artist'] = artist
        self.current_manual_meta['title'] = title
        
        # Save to Local DB
        if save_to_db:
            try:
                genres_json = json.dumps(genres)
                self.db_conn.execute(
                    "INSERT OR REPLACE INTO songs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (artist, title, bpm, meta.get("key_of",""), meta.get("time_sig",""), 
                     meta.get("danceability",50), meta.get("acousticness",0), genres_json)
                )
                self.db_conn.commit()
                self._update_completer_list()
                self.bpm_source_label.setText(f"API FETCHED & SAVED ✓ | Genre: {genre_str}")
            except Exception as e:
                print(f"Error saving to DB: {e}")
        else:
            self.bpm_source_label.setText(f"LOADED FROM LOCAL DB ✓ | Genre: {genre_str}")
            
        self.meta_label.setText(f"✓ Manual Track: {query}")
        self._apply_genre_colors_to_manual(genres, meta.get("acousticness", 0))

    def _show_metadata_info(self):
        if not self.current_manual_meta:
            QtWidgets.QMessageBox.information(self, "Song Info", "No song loaded in manual mode yet.\nSearch for a song to view its metadata.")
            return
            
        m = self.current_manual_meta
        info_text = (
            f"Artist: {m.get('artist', 'Unknown')}\n"
            f"Title: {m.get('title', 'Unknown')}\n\n"
            f"BPM: {m.get('bpm', 'Unknown')}\n"
            f"Key: {m.get('key_of', 'Unknown')}\n"
            f"Time Signature: {m.get('time_sig', 'Unknown')}\n\n"
            f"Danceability: {m.get('danceability', 'Unknown')}/100\n"
            f"Acousticness: {m.get('acousticness', 'Unknown')}/100\n"
            f"Genres: {', '.join(m.get('genres', []))}\n"
        )
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Track Metadata")
        msg_box.setText(info_text)
        msg_box.setStyleSheet("QLabel{min-width: 300px; font-size: 14px;}")
        msg_box.exec()

    def _apply_genre_colors_to_manual(self, genres, acousticness):
        active_genres = [g.lower() for g in genres]
        if any(g in ['edm', 'electronic', 'dance', 'house', 'techno'] for g in active_genres):
            pal_strong = [255.0, 50.0, 255.0]; pal_mid = [100.0, 0.0, 255.0]; pal_high = [0.0, 150.0, 255.0]
        elif any(g in ['country', 'folk', 'americana'] for g in active_genres):
            pal_strong = [255.0, 100.0, 0.0]; pal_mid = [255.0, 0.0, 0.0]; pal_high = [0.0, 255.0, 0.0]
        elif any(g in ['hip hop', 'rap', 'trap'] for g in active_genres):
            pal_strong = [50.0, 255.0, 50.0]; pal_mid = [150.0, 0.0, 255.0]; pal_high = [255.0, 100.0, 0.0]
        elif any(g in ['rock', 'metal', 'heavy metal', 'punk'] for g in active_genres):
            pal_strong = [255.0, 255.0, 255.0]; pal_mid = [255.0, 0.0, 0.0]; pal_high = [200.0, 50.0, 50.0]
        else:
            if acousticness > 60:
                pal_strong = [255.0, 200.0, 100.0]; pal_mid = [255.0, 150.0, 0.0]; pal_high = [255.0, 100.0, 0.0]
            else:
                pal_strong = [255.0, 255.0, 255.0]; pal_mid = [150.0, 0.0, 255.0]; pal_high = [0.0, 200.0, 255.0]
                
        self.manual_colors["strong"] = pal_strong
        self.manual_colors["mid"] = pal_mid
        self.manual_colors["high"] = pal_high
        self._update_btn_color(self.btn_color_strong, pal_strong)
        self._update_btn_color(self.btn_color_mid, pal_mid)
        self._update_btn_color(self.btn_color_high, pal_high)

    # ------------------------------------------------------------------ #
    #  UI Construction
    # ------------------------------------------------------------------ #
    def _init_ui(self):
        self.setWindowTitle("Real-Time Audio Reactive System")
        self.resize(900, 950)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.meta_label = QtWidgets.QLabel("◉ Listening for song…")
        self.meta_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.meta_label.setFixedHeight(40)
        self.meta_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #CCCCCC; background-color: #1a1a2e; border-bottom: 1px solid #333; padding: 4px; border-radius: 4px;")
        layout.addWidget(self.meta_label)

        self.bpm_source_label = QtWidgets.QLabel("BPM SOURCE: reactive")
        self.bpm_source_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.bpm_source_label.setFixedHeight(22)
        self.bpm_source_label.setStyleSheet("font-size: 11px; color: #888888; background-color: #111111;")
        layout.addWidget(self.bpm_source_label)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

        self.waveform_plot = self.win.addPlot(title="Time Domain (Raw Waveform Input)")
        self.waveform_plot.setYRange(-1.0, 1.0)
        self.waveform_plot.setXRange(0, CHUNK_SIZE)
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen('g', width=1))
        self.win.nextRow()

        self.fft_plot = self.win.addPlot(title="Frequency Domain (Extracted Feature Energies)")
        self.fft_plot.setYRange(0, 1.0)
        self.bar_x     = [1, 2, 3]
        self.bar_graph = pg.BarGraphItem(x=self.bar_x, height=[0, 0, 0], width=0.6, brush='r')
        self.fft_plot.addItem(self.bar_graph)

        ax = self.fft_plot.getAxis('bottom')
        ax.setTicks([[(1, 'Bass'), (2, 'Mids'), (3, 'Treble')]])

        self.bpm_label = QtWidgets.QLabel("BPM: —")
        self.bpm_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.bpm_label.setFixedHeight(60)
        self.bpm_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #00FF00; background-color: #111111; border: 1px solid #222; border-radius: 8px;")
        layout.addWidget(self.bpm_label)

        # Master Brightness Slider
        brightness_layout = QtWidgets.QHBoxLayout()
        self.brightness_label = QtWidgets.QLabel("Master Brightness: 50%")
        self.brightness_label.setFixedWidth(150)
        self.brightness_label.setStyleSheet("font-weight: bold; color: #CCCCCC;")
        
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(50)  
        self.brightness_slider.valueChanged.connect(self._update_brightness)
        
        brightness_layout.addWidget(self.brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        layout.addLayout(brightness_layout)

        # Lighting Engine Control
        control_frame = QtWidgets.QGroupBox("Lighting Engine Control")
        control_frame.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #444; margin-top: 10px;} QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        control_layout = QtWidgets.QVBoxLayout(control_frame)
        
        mode_layout = QtWidgets.QHBoxLayout()
        self.radio_auto = QtWidgets.QRadioButton("Automatic (API & Reactive)")
        self.radio_manual = QtWidgets.QRadioButton("Manual Override")
        self.radio_auto.setChecked(True)
        self.radio_auto.toggled.connect(self._toggle_mode)
        mode_layout.addWidget(self.radio_auto)
        mode_layout.addWidget(self.radio_manual)
        control_layout.addLayout(mode_layout)

        self.manual_panel = QtWidgets.QWidget()
        manual_grid = QtWidgets.QGridLayout(self.manual_panel)
        
        manual_grid.addWidget(QtWidgets.QLabel("Preset:"), 0, 0)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(self.presets.keys())
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        manual_grid.addWidget(self.preset_combo, 0, 1)

        manual_grid.addWidget(QtWidgets.QLabel("Manual BPM:"), 1, 0)
        self.manual_bpm_spin = QtWidgets.QDoubleSpinBox()
        self.manual_bpm_spin.setRange(40.0, 220.0)
        self.manual_bpm_spin.setValue(128.0)
        self.manual_bpm_spin.valueChanged.connect(self._update_manual_bpm)
        manual_grid.addWidget(self.manual_bpm_spin, 1, 1)

        manual_grid.addWidget(QtWidgets.QLabel("Colors:"), 2, 0)
        color_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_color_strong = QtWidgets.QPushButton("Bass (Strong)")
        self.btn_color_strong.clicked.connect(lambda: self._pick_color("strong", self.btn_color_strong))
        self.btn_color_mid = QtWidgets.QPushButton("Mids")
        self.btn_color_mid.clicked.connect(lambda: self._pick_color("mid", self.btn_color_mid))
        self.btn_color_high = QtWidgets.QPushButton("Treble (High)")
        self.btn_color_high.clicked.connect(lambda: self._pick_color("high", self.btn_color_high))
        color_btn_layout.addWidget(self.btn_color_strong)
        color_btn_layout.addWidget(self.btn_color_mid)
        color_btn_layout.addWidget(self.btn_color_high)
        manual_grid.addLayout(color_btn_layout, 2, 1)

        manual_grid.addWidget(QtWidgets.QLabel("Load Track:"), 3, 0)
        search_layout = QtWidgets.QHBoxLayout()
        self.api_search_input = QtWidgets.QLineEdit()
        self.api_search_input.setPlaceholderText("Artist - Song Title")
        
        # Setup Autocomplete Completer
        self.completer_model = QtCore.QStringListModel()
        self.completer = QtWidgets.QCompleter()
        self.completer.setModel(self.completer_model)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.api_search_input.setCompleter(self.completer)
        self._update_completer_list()

        btn_api_search = QtWidgets.QPushButton("Load")
        btn_api_search.clicked.connect(self._load_song_action)
        
        btn_info = QtWidgets.QPushButton("ℹ")
        btn_info.setFixedWidth(30)
        btn_info.setToolTip("View Track Metadata")
        btn_info.clicked.connect(self._show_metadata_info)

        search_layout.addWidget(self.api_search_input)
        search_layout.addWidget(btn_api_search)
        search_layout.addWidget(btn_info)
        manual_grid.addLayout(search_layout, 3, 1)

        manual_grid.addWidget(QtWidgets.QLabel("Save:"), 4, 0)
        save_layout = QtWidgets.QHBoxLayout()
        self.preset_name_input = QtWidgets.QLineEdit()
        self.preset_name_input.setPlaceholderText("Name new preset...")
        btn_save = QtWidgets.QPushButton("Save Custom Preset")
        btn_save.clicked.connect(self._save_preset)
        save_layout.addWidget(self.preset_name_input)
        save_layout.addWidget(btn_save)
        manual_grid.addLayout(save_layout, 4, 1)

        self.manual_panel.setVisible(False)
        control_layout.addWidget(self.manual_panel)
        layout.addWidget(control_frame)
        self._apply_preset(self.preset_combo.currentText())

    def _update_brightness(self, val):
        self.global_brightness = val / 100.0
        self.brightness_label.setText(f"Master Brightness: {val}%")

    def _toggle_mode(self):
        if self.radio_auto.isChecked():
            self.mode = "AUTO"
            self.manual_panel.setVisible(False)
        else:
            self.mode = "MANUAL"
            self.manual_panel.setVisible(True)

    def _apply_preset(self, preset_name):
        if preset_name in self.presets:
            p = self.presets[preset_name]
            self.manual_bpm_spin.setValue(p["bpm"])
            self.manual_colors["strong"] = [float(c) for c in p["strong"]]
            self.manual_colors["mid"] = [float(c) for c in p["mid"]]
            self.manual_colors["high"] = [float(c) for c in p["high"]]
            self._update_btn_color(self.btn_color_strong, p["strong"])
            self._update_btn_color(self.btn_color_mid, p["mid"])
            self._update_btn_color(self.btn_color_high, p["high"])

    def _update_manual_bpm(self, val):
        self.manual_bpm = val

    def _update_btn_color(self, btn, rgb):
        btn.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); color: {'black' if sum(rgb)>380 else 'white'}; font-weight: bold;")

    def _pick_color(self, band, button):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            rgb = [float(color.red()), float(color.green()), float(color.blue())]
            self.manual_colors[band] = rgb
            self._update_btn_color(button, rgb)

    def _update_meta_label(self):
        if self.mode == "MANUAL":
            if not self.meta_label.text().startswith("✓ Manual Track:") and not self.meta_label.text().startswith("⟳ Fetching"):
                self.meta_label.setText("⚙️ MANUAL OVERRIDE ACTIVE")
                self.bpm_source_label.setText("BPM SOURCE: User Interface")
            self.meta_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF; background-color: #8B0000; padding: 4px;")
            return

        state  = self.song_id.state
        status = self.song_id.status_line()
        self.meta_label.setText(status)

        if state == IDState.LOCKED:
            self.meta_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #00FF88; background-color: #0d2b1a; padding: 4px;")
        elif state == IDState.IDENTIFYING:
            self.meta_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFD700; background-color: #2b2500; padding: 4px;")
        elif state == IDState.FALLBACK:
            self.meta_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #FF8844; background-color: #2b1500; padding: 4px;")
        else:
            self.meta_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #CCCCCC; background-color: #1a1a2e; padding: 4px;")

        bpm_override = self.song_id.bpm_override
        if state == IDState.LOCKED:
            genre_str = ", ".join(self.song_id.genres[:2]) if self.song_id.genres else "Unknown Genre"
            if bpm_override is not None:
                self.bpm_source_label.setText(f"DATABASE LOCKED ✓ | Genre: {genre_str}")
                self.bpm_source_label.setStyleSheet("font-size: 11px; color: #00FF88; background-color: #111111;")
            else:
                self.bpm_source_label.setText(f"DATABASE METADATA ONLY | Genre: {genre_str}")
                self.bpm_source_label.setStyleSheet("font-size: 11px; color: #FFD700; background-color: #111111;")
        else:
            self.bpm_source_label.setText("BPM SOURCE: reactive engine")
            self.bpm_source_label.setStyleSheet("font-size: 11px; color: #888888; background-color: #111111;")

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass
        if self.mode == "AUTO":
            self.song_id.feed(chunk)

    def _start_audio_stream(self):
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE,
                device=self.device_id, callback=self._audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Fatal audio stream error: {e}")
            sys.exit(1)

    def _update(self):
        chunk = None
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get_nowait()
        if chunk is None: return

        now = time.monotonic()
        display = np.clip(chunk * WAVEFORM_GAIN, -1.0, 1.0)
        self.waveform_curve.setData(display)

        if self.song_id.use_fallback or self.mode == "MANUAL":
            try:
                self.bpm_send_sock.sendto(chunk.astype(np.float32).tobytes(), self.BPM_SERVER_ADDR)
            except OSError: pass

        feat = self.dsp.process_audio(chunk)
        if not feat: return

        alpha = 0.35
        for k in ('bass', 'mids', 'treble'):
            self.smoothed[k] = alpha * feat.get(k, 0.0) + (1 - alpha) * self.smoothed[k]

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

        raw_bass = float(feat.get('bass', 0.0))
        self.bass_history[self.history_idx] = raw_bass
        self.history_idx = (self.history_idx + 1) % len(self.bass_history)
        valid    = self.bass_history[self.bass_history > 0]
        baseline = float(np.percentile(valid, 50)) if len(valid) > 5 else raw_bass

        if self.mode == "MANUAL":
            self.bpm_label.setText(f"BPM: {self.manual_bpm:.1f} (MANUAL)")
            beat_interval = 60.0 / max(self.manual_bpm, 1.0)
            bpm_beat_on = (now % beat_interval) < self.BPM_FLASH_DURATION
            is_beat = bpm_beat_on  
        else:
            bpm_override = self.song_id.bpm_override
            if bpm_override is not None:
                self.bpm_label.setText(f"BPM: {bpm_override:.1f}")
                beat_interval = 60.0 / bpm_override
                bpm_beat_on = (now % beat_interval) < self.BPM_FLASH_DURATION
                is_beat = bpm_beat_on  
            else:
                self.beat_cooldown = 0.25
                is_beat = (raw_bass > baseline * 1.25 and raw_bass > 0.02 and (now - self.last_beat_time) > self.beat_cooldown)
                if is_beat: self.last_beat_time = now

                try:
                    while True:
                        data, _ = self.bpm_sock.recvfrom(64)
                        try:
                            msg = data.decode('utf-8').strip()
                            if ":" in msg:
                                parsed_val = msg.split(":")[1].strip()
                                if parsed_val.replace('.', '', 1).isdigit():
                                    self.estimated_reactive_bpm = float(parsed_val)
                        except: pass
                        
                        interval = now - self.last_bpm_beat_time
                        if 0.3 <= interval <= 2.0:
                            inst_bpm = 60.0 / interval
                            if self.estimated_reactive_bpm == 0.0:
                                self.estimated_reactive_bpm = inst_bpm
                            else:
                                self.estimated_reactive_bpm = (self.estimated_reactive_bpm * 0.8) + (inst_bpm * 0.2)
                        self.last_bpm_beat_time = now
                except BlockingIOError: pass
                    
                bpm_beat_on = (now - self.last_bpm_beat_time) < self.BPM_FLASH_DURATION
                if self.estimated_reactive_bpm > 0:
                    self.bpm_label.setText(f"BPM: ~{self.estimated_reactive_bpm:.1f} (Live)")
                else:
                    self.bpm_label.setText("BPM: Calculating...")


        # Create a smooth, continuous breathing effect based on the active BPM
        active_bpm = self.song_id.bpm_override if self.song_id.bpm_override else self.estimated_reactive_bpm

        if active_bpm > 0:
            # Convert time and BPM into a continuous moving angle
            time_factor = (now * active_bpm) / 60.0 
    
            # Generate a sine wave that oscillates smoothly between 0.0 and 1.0
            sine_wave = (np.sin(time_factor * np.pi * 2) + 1.0) / 2.0
    
            # Map it to a byte (0-255), but cap the max brightness so it stays in the background
            max_glow = 120 
            beat_byte = int(sine_wave * max_glow)
        else:
            beat_byte = 0
            
        b_f = b_val / 254; m_f = m_val / 254; t_f = t_val / 254
        ambient_r = ambient_g = ambient_b = 0.0
        
        if self.mode == "MANUAL":
            pal_strong = self.manual_colors["strong"]
            pal_mid = self.manual_colors["mid"]
            pal_high = self.manual_colors["high"]
            ambient_r = (pal_strong[0]/255)*b_f + (pal_mid[0]/255)*m_f + (pal_high[0]/255)*t_f
            ambient_g = (pal_strong[1]/255)*b_f + (pal_mid[1]/255)*m_f + (pal_high[1]/255)*t_f
            ambient_b = (pal_strong[2]/255)*b_f + (pal_mid[2]/255)*m_f + (pal_high[2]/255)*t_f
        else:
            active_genres = [g.lower() for g in self.song_id.genres]
            if any(g in ['edm', 'electronic', 'dance', 'house', 'techno'] for g in active_genres):
                if b_f >= m_f and b_f >= t_f: ambient_r = b_f * 0.8; ambient_b = b_f * 0.9 
                elif m_f >= b_f and m_f >= t_f: ambient_b = m_f * 0.9; ambient_r = m_f * 0.4 
                else: ambient_b = t_f * 0.8; ambient_r = t_f * 0.8 
                pal_strong = [255.0, 50.0, 255.0]; pal_mid = [100.0, 0.0, 255.0]; pal_high = [0.0, 150.0, 255.0]
            elif any(g in ['country', 'folk', 'americana'] for g in active_genres):
                if b_f >= m_f and b_f >= t_f: ambient_r = b_f * 0.9; ambient_g = b_f * 0.3 
                elif m_f >= b_f and m_f >= t_f: ambient_r = m_f * 0.8; ambient_g = m_f * 0.1 
                else: ambient_g = t_f * 0.8; ambient_b = t_f * 0.1 
                pal_strong = [255.0, 100.0, 0.0]; pal_mid = [255.0, 0.0, 0.0]; pal_high = [0.0, 255.0, 0.0]
            elif any(g in ['hip hop', 'rap', 'trap'] for g in active_genres):
                if b_f >= m_f and b_f >= t_f: ambient_g = b_f * 0.8; ambient_r = b_f * 0.2 
                elif m_f >= b_f and m_f >= t_f: ambient_r = m_f * 0.6; ambient_b = m_f * 0.8 
                else: ambient_r = t_f * 0.9; ambient_g = t_f * 0.4 
                pal_strong = [50.0, 255.0, 50.0]; pal_mid = [150.0, 0.0, 255.0]; pal_high = [255.0, 100.0, 0.0]
            elif any(g in ['rock', 'metal', 'heavy metal', 'punk'] for g in active_genres):
                if b_f >= m_f and b_f >= t_f: ambient_r = b_f * 0.9 
                elif m_f >= b_f and m_f >= t_f: ambient_r = m_f * 0.6; ambient_g = m_f * 0.6; ambient_b = m_f * 0.6 
                else: ambient_r = t_f * 0.9; ambient_b = t_f * 0.2 
                pal_strong = [255.0, 255.0, 255.0]; pal_mid = [255.0, 0.0, 0.0]; pal_high = [200.0, 50.0, 50.0]
            else:
                if self.song_id.acousticness > 60:
                    if b_f >= m_f and b_f >= t_f: ambient_r = b_f * 0.8;  ambient_g = b_f * 0.3 
                    elif m_f >= b_f and m_f >= t_f: ambient_r = m_f * 0.6;  ambient_g = m_f * 0.4 
                    else: ambient_r = t_f * 0.8;  ambient_g = t_f * 0.2 
                    pal_strong = [255.0, 200.0, 100.0]; pal_mid = [255.0, 150.0, 0.0]; pal_high = [255.0, 100.0, 0.0]
                else:
                    if b_f >= m_f and b_f >= t_f: ambient_r = b_f * 0.8;  ambient_b = b_f * 0.1
                    elif m_f >= b_f and m_f >= t_f: ambient_b = m_f * 0.9;  ambient_g = m_f * 0.2
                    else: ambient_g = t_f * 0.8;  ambient_r = t_f * 0.2
                    pal_strong = [255.0, 255.0, 255.0]; pal_mid = [150.0, 0.0, 255.0]; pal_high = [0.0, 200.0, 255.0]

        if is_beat:
            if raw_bass > baseline * 2.0: self.current_rgb = pal_strong.copy()
            elif m_val > t_val: self.current_rgb = pal_mid.copy()
            else: self.current_rgb = pal_high.copy()
        else:
            drift = 0.3
            target_r = ambient_r * 255.0; target_g = ambient_g * 255.0; target_b = ambient_b * 255.0
            self.current_rgb[0] += (target_r - self.current_rgb[0]) * drift
            self.current_rgb[1] += (target_g - self.current_rgb[1]) * drift
            self.current_rgb[2] += (target_b - self.current_rgb[2]) * drift

        rgb_r = int(np.clip(self.current_rgb[0] * self.global_brightness, 0, 255))
        rgb_g = int(np.clip(self.current_rgb[1] * self.global_brightness, 0, 255))
        rgb_b = int(np.clip(self.current_rgb[2] * self.global_brightness, 0, 255))

        if self.arduino:
            try:
                packet = struct.pack('<BBBBBBBB', 255, b_val, m_val, t_val, rgb_r, rgb_g, rgb_b, beat_byte)
                self.arduino.write(packet)
            except serial.SerialException as e:
                print(f"Arduino communication failure: {e}")
                self.arduino = None

    def closeEvent(self, event):
        self.timer.stop()
        self.meta_timer.stop()
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.bpm_sock.close()
        self.bpm_send_sock.close()
        self.db_conn.close()
        if self.arduino:
            self.arduino.close()
        event.accept()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    visualizer = RealTimeVisualizer()
    visualizer.show()
    sys.exit(app.exec())