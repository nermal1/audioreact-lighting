"""
song_identifier.py — Music Identification & Metadata Cache

State machine:
    LISTENING   → accumulating audio for LISTEN_SECONDS
    IDENTIFYING → background thread calling AudD API
    LOCKED      → song known; BPM comes from DB, not detection
    FALLBACK    → API failed or song unknown; reactive BPM engine takes over
    COOLDOWN    → brief pause between retries when in FALLBACK

API Chaining:
    If AudD identifies the Artist/Title but fails to provide the BPM, 
    the engine will automatically chain a request to the GetSongBPM API 
    to retrieve the Tempo (BPM) and Key.
"""

import io
import wave
import time
import sqlite3
import hashlib
import threading
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
import requests
import os

# ------------------------------------------------------------------ #
#  Configuration — edit these
# ------------------------------------------------------------------ #

AUDD_API_KEY       = os.getenv('AUDD_API_KEY')  
GETSONGBPM_API_KEY = os.getenv('GETSONGBPM_API_KEY')

DB_PATH         = 'music_cache.sqlite'
LISTEN_SECONDS  = 8           
RETRY_COOLDOWN  = 20.0        
SAMPLE_RATE     = 44100


# ------------------------------------------------------------------ #
#  State machine
# ------------------------------------------------------------------ #
class IDState(Enum):
    LISTENING   = auto()   
    IDENTIFYING = auto()   
    LOCKED      = auto()   
    FALLBACK    = auto()   
    COOLDOWN    = auto()   

@dataclass
class SongMetadata:
    title:  str
    artist: str
    bpm:    float
    key:    str  = ''
    source: str  = 'api'   


# ------------------------------------------------------------------ #
#  SQLite cache
# ------------------------------------------------------------------ #
def _init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            fingerprint TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            artist      TEXT NOT NULL,
            bpm         REAL NOT NULL,
            key         TEXT DEFAULT '',
            added_at    REAL NOT NULL
        )
    """)
    conn.commit()
    return conn

def _cache_lookup(conn: sqlite3.Connection, fingerprint: str) -> Optional[SongMetadata]:
    row = conn.execute(
        "SELECT title, artist, bpm, key FROM songs WHERE fingerprint = ?",
        (fingerprint,)
    ).fetchone()
    if row:
        return SongMetadata(title=row[0], artist=row[1], bpm=row[2], key=row[3], source='cache')
    return None

def _cache_insert(conn: sqlite3.Connection, fingerprint: str, meta: SongMetadata) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO songs (fingerprint, title, artist, bpm, key, added_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (fingerprint, meta.title, meta.artist, meta.bpm, meta.key, time.time())
    )
    conn.commit()


# ------------------------------------------------------------------ #
#  Audio fingerprint
# ------------------------------------------------------------------ #
def _fingerprint(audio: np.ndarray) -> str:
    step   = SAMPLE_RATE // 8000
    coarse = (audio[::step] * 1000).astype(np.int16).tobytes()
    return hashlib.md5(coarse).hexdigest()


# ------------------------------------------------------------------ #
#  GetSongBPM API Fallback
# ------------------------------------------------------------------ #
def _fetch_getsongbpm_features(title: str, artist: str, api_key: str) -> tuple[float, str]:
    """Searches GetSongBPM for the track and retrieves its Tempo and Key."""
    if not api_key:
        return 0.0, ""
        
    # Clean the query strings to improve search accuracy
    clean_title = title.split('(')[0].split('-')[0].strip()
    clean_artist = artist.split(',')[0].strip()
    
    # GetSongBPM expects the "both" type query to be formatted specifically
    lookup_str = f"song:{clean_title} artist:{clean_artist}"
    
    try:
        resp = requests.get(
            "https://getsongbpm.com/api/search/", 
            params={
                "api_key": api_key,
                "type": "both",
                "lookup": lookup_str,
                "limit": 1
            },
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        
        # The API can return the list directly or wrapped in a 'search' key
        search_results = data if isinstance(data, list) else data.get("search", [])
        
        if not search_results:
            return 0.0, ""
            
        track = search_results[0]
        bpm = float(track.get("tempo", 0.0))
        key_str = str(track.get("key_of", ""))
            
        return bpm, key_str
        
    except Exception as e:
        print(f"[GetSongBPM] Metadata Retrieval Error: {e}")
        return 0.0, ""


# ------------------------------------------------------------------ #
#  AudD API call
# ------------------------------------------------------------------ #
def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()

def _query_audd(audio: np.ndarray, api_key: str) -> Optional[SongMetadata]:
    wav_bytes = _encode_wav(audio, SAMPLE_RATE)
    try:
        resp = requests.post(
            'https://api.audd.io/',
            data={'api_token': api_key, 'return': 'spotify,apple_music'},
            files={'file': ('audio.wav', wav_bytes, 'audio/wav')},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[SongID] AudD request failed: {e}")
        return None

    if data.get('status') != 'success' or not data.get('result'):
        print("[SongID] No match found.")
        return None

    result = data['result']
    title  = result.get('title',  'Unknown')
    artist = result.get('artist', 'Unknown')
    bpm = 0.0
    key = ''

    # AudD internal routing attempts
    spotify = result.get('spotify') or {}
    if isinstance(spotify, dict):
        features = spotify.get('audio_features') or {}
        if isinstance(features, dict):
            bpm = float(features.get('tempo', 0.0))
            key_num = features.get('key', -1)
            mode    = features.get('mode', 1)
            if isinstance(key_num, int) and key_num >= 0:
                key_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                key = f"{key_names[key_num % 12]} {'Major' if mode else 'Minor'}"

    if bpm == 0.0:
        apple = result.get('apple_music') or {}
        if isinstance(apple, dict):
            bpm = float(apple.get('tempo', 0.0))

    if bpm == 0.0:
        bpm = float(result.get('bpm', 0.0))

    return SongMetadata(title=title, artist=artist, bpm=bpm, key=key, source='api')


# ------------------------------------------------------------------ #
#  SongIdentifier 
# ------------------------------------------------------------------ #
class SongIdentifier:
    def __init__(self, api_key: str = AUDD_API_KEY, db_path: str = DB_PATH,
                 sample_rate: int = SAMPLE_RATE):
        self.api_key     = api_key
        self.sample_rate = sample_rate

        self.db   = _init_db(db_path)
        self.state: IDState                   = IDState.LISTENING
        self.metadata: Optional[SongMetadata] = None

        self._listen_samples = int(LISTEN_SECONDS * sample_rate)
        self._buffer: list[np.ndarray] = []
        self._buffer_len = 0
        self._cooldown_start = 0.0

        self._id_thread: Optional[threading.Thread] = None
        self._result_pending: Optional[Optional[SongMetadata]] = None  
        self._result_ready = False

        print(f"[SongID] Initialised — state: LISTENING "
              f"({'API key set' if api_key else 'NO API KEY — fallback only'})")

    @property
    def bpm_override(self) -> Optional[float]:
        if self.state == IDState.LOCKED and self.metadata and self.metadata.bpm > 0:
            return self.metadata.bpm
        return None

    @property
    def use_fallback(self) -> bool:
        return self.state != IDState.LOCKED

    def feed(self, chunk: np.ndarray) -> None:
        self._check_result()   

        if self.state == IDState.LISTENING:
            self._buffer.append(chunk)
            self._buffer_len += len(chunk)
            if self._buffer_len >= self._listen_samples:
                self._start_identification()

        elif self.state == IDState.COOLDOWN:
            if time.monotonic() - self._cooldown_start >= RETRY_COOLDOWN:
                self._reset_to_listening()

    def reset(self) -> None:
        print("[SongID] Manual reset — returning to LISTENING")
        self.metadata = None
        self._reset_to_listening()

    def status_line(self) -> str:
        if self.state == IDState.LOCKED and self.metadata:
            bpm_str = f"{self.metadata.bpm:.1f} BPM" if self.metadata.bpm > 0 else "BPM unknown"
            key_str = f" · {self.metadata.key}" if self.metadata.key else ""
            src_str = f" [{self.metadata.source}]"
            return f"♪ {self.metadata.artist} — {self.metadata.title}  |  {bpm_str}{key_str}{src_str}"
        elif self.state == IDState.IDENTIFYING:
            return "⟳ Identifying song…"
        elif self.state == IDState.FALLBACK:
            return "? Unrecognised — reactive BPM active"
        elif self.state == IDState.COOLDOWN:
            elapsed  = time.monotonic() - self._cooldown_start
            remaining = max(0, RETRY_COOLDOWN - elapsed)
            return f"↺ Retrying in {remaining:.0f}s — reactive BPM active"
        else:
            return f"◉ Listening… ({self._buffer_len / self.sample_rate:.1f}s / {LISTEN_SECONDS}s)"

    def _reset_to_listening(self) -> None:
        self._buffer     = []
        self._buffer_len = 0
        self.state       = IDState.LISTENING

    def _start_identification(self) -> None:
        audio = np.concatenate(self._buffer)
        fp    = _fingerprint(audio)

        cached = _cache_lookup(self.db, fp)
        if cached:
            print(f"[SongID] Cache hit: {cached.artist} — {cached.title} ({cached.bpm:.1f} BPM)")
            self.metadata = cached
            self.state    = IDState.LOCKED
            return

        if not self.api_key:
            print("[SongID] No API key — entering FALLBACK")
            self.state           = IDState.FALLBACK
            self._cooldown_start = time.monotonic()
            self._reset_to_listening()
            self.state = IDState.COOLDOWN
            return

        self.state          = IDState.IDENTIFYING
        self._result_ready  = False
        self._result_pending = None

        self._id_thread = threading.Thread(
            target=self._identify_worker,
            args=(audio, fp),
            daemon=True,
        )
        self._id_thread.start()

    def _identify_worker(self, audio: np.ndarray, fingerprint: str) -> None:
        meta = _query_audd(audio, self.api_key)
        
        if meta:
            # --- API CHAINING: The GetSongBPM Fallback ---
            if meta.bpm == 0.0 and GETSONGBPM_API_KEY:
                print(f"[SongID] AudD found '{meta.title}', but no BPM. Connecting to GetSongBPM Database...")
                bpm, key = _fetch_getsongbpm_features(meta.title, meta.artist, GETSONGBPM_API_KEY)
                if bpm > 0:
                    meta.bpm = bpm
                    meta.key = key if key else meta.key
                    print(f"[SongID] GetSongBPM Database Chain Success! Found {meta.bpm} BPM.")
            
            _cache_insert(self.db, fingerprint, meta)
            
            if meta.bpm > 0:
                print(f"[SongID] Locked: {meta.artist} — {meta.title} ({meta.bpm:.1f} BPM, {meta.key or '—'})")
            else:
                print(f"[SongID] Locked (no BPM): {meta.artist} — {meta.title} — reactive BPM will be used")
        else:
            print("[SongID] Could not identify song — entering COOLDOWN")
            
        self._result_pending = meta
        self._result_ready   = True

    def _check_result(self) -> None:
        if not self._result_ready:
            return
        self._result_ready = False
        meta = self._result_pending

        if meta:
            self.metadata = meta
            self.state    = IDState.LOCKED
        else:
            self.metadata        = None
            self._cooldown_start = time.monotonic()
            self._reset_to_listening()
            self.state = IDState.COOLDOWN