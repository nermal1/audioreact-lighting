import os
import io
import wave
import time
import threading
import queue
import requests
import numpy as np
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class IDState(Enum):
    LISTENING = 1
    IDENTIFYING = 2
    LOCKED = 3
    FALLBACK = 4

class SongIdentifier:
    def __init__(self, api_key: str, sample_rate: int = 44100):
        self.api_key = api_key
        self.getsongbpm_key = os.getenv('GETSONGBPM_API_KEY')
        self.sample_rate = sample_rate
        
        self.state = IDState.LISTENING
        self.bpm_override = None
        self.use_fallback = True
        
        # Track Metadata
        self.current_song = "Unknown"
        self.current_artist = "Unknown"
        self.key_of = ""
        self.time_sig = ""
        self.danceability = 50
        self.acousticness = 0
        self.genres = []
        
        self.last_state_change = time.time()
        
        # Silence Detection Setup
        self.silence_start_time = None
        self.silence_threshold = 0.005  
        self.silence_duration_needed = 2.5  
        
        self.buffer_lock = threading.Lock()
        self.audio_buffer = []
        self.required_samples = sample_rate * 6 
        
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _set_state(self, new_state: IDState):
        self.state = new_state
        self.last_state_change = time.time()

    def feed(self, chunk: np.ndarray):
        # --- SILENCE DETECTION (The Only Way to Reset) ---
        rms = np.sqrt(np.mean(np.square(chunk)))
        
        if rms < self.silence_threshold:
            if self.silence_start_time is None:
                self.silence_start_time = time.time()
            elif time.time() - self.silence_start_time > self.silence_duration_needed:
                if self.state in (IDState.LOCKED, IDState.FALLBACK, IDState.IDENTIFYING):
                    print("\n[SongID] 🔇 Sustained silence detected! Resetting for new track...")
                    self._reset_metadata()
                    with self.buffer_lock:
                        self.audio_buffer.clear()
                    self._set_state(IDState.LISTENING)
                self.silence_start_time = None 
        else:
            self.silence_start_time = None

        # --- Audio Accumulation ---
        if self.state not in (IDState.LISTENING, IDState.IDENTIFYING):
            return

        with self.buffer_lock:
            self.audio_buffer.extend(chunk.tolist())
            if len(self.audio_buffer) >= self.required_samples and self.state == IDState.LISTENING:
                samples_to_process = np.array(self.audio_buffer[:self.required_samples], dtype=np.float32)
                self.audio_buffer = self.audio_buffer[int(self.sample_rate * 2):]
                print("\n[SongID] 6 seconds of audio captured. Sending to AudD...")
                self._set_state(IDState.IDENTIFYING)
                self.task_queue.put(samples_to_process)

    def status_line(self) -> str:
        if self.state == IDState.LISTENING:
            return "◉ Listening for song…"
        elif self.state == IDState.IDENTIFYING:
            return "⟳ Identifying track via AudD..."
        elif self.state == IDState.LOCKED:
            return f"✓ Playing: {self.current_artist} — {self.current_song}"
        return "⚠️ Unrecognised Audio — Reactive Mode Active"

    def _reset_metadata(self):
        self.current_song = "Unknown"
        self.current_artist = "Unknown"
        self.bpm_override = None
        self.key_of = ""
        self.time_sig = ""
        self.danceability = 50
        self.acousticness = 0
        self.genres = []

    def _convert_to_wav_bytes(self, audio_samples: np.ndarray) -> bytes:
        int_samples = np.clip(audio_samples * 32767, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) 
            wf.setframerate(self.sample_rate)
            wf.writeframes(int_samples.tobytes())
        return buf.getvalue()

    def _fetch_getsongbpm_fallback(self, title: str, artist: str) -> dict:
        if not self.getsongbpm_key:
            return {}
            
        clean_title = title.split('(')[0].split('-')[0].strip()
        clean_artist = artist.split(',')[0].strip()
        lookup_str = f"song:{clean_title} artist:{clean_artist}"
        
        print(f"[SongID] ⟳ Chaining request to GetSongBPM for: {clean_artist} — {clean_title}...")
        try:
            resp = requests.get(
                "https://api.getsong.co/search/",
                params={"api_key": self.getsongbpm_key, "type": "both", "lookup": lookup_str, "limit": 1},
                headers={"User-Agent": "AudioReact-Lighting/1.0", "Accept": "application/json"},
                timeout=5
            )
            resp.raise_for_status()
            data = resp.json()
            
            # SAFE PARSING: Prevent 'KeyError: 0' by ensuring it is a populated list
            search_results = data if isinstance(data, list) else data.get("search", [])
            if not search_results or not isinstance(search_results, list) or len(search_results) == 0:
                print(f"[SongID] ℹ️ Track not found in GetSongBPM database.")
                return {}
                
            track = search_results[0]
            artist_data = track.get("artist", {})
            
            return {
                "bpm": float(track.get("tempo", 0.0)),
                "key_of": str(track.get("key_of", "")),
                "time_sig": str(track.get("time_sig", "")),
                "danceability": int(track.get("danceability", 50)),
                "acousticness": int(track.get("acousticness", 0)),
                "genres": artist_data.get("genres", [])
            }
        except Exception as e:
            print(f"[SongID] ❌ GetSongBPM Fetch Error: {type(e).__name__} - {e}")
            return {}

    def _worker_loop(self):
        while True:
            audio_samples = self.task_queue.get()
            if audio_samples is None:
                break
                
            wav_bytes = self._convert_to_wav_bytes(audio_samples)
            
            try:
                resp = requests.post(
                    'https://api.audd.io/',
                    data={'api_token': self.api_key, 'return': 'spotify,apple_music'},
                    files={'file': ('audio.wav', wav_bytes, 'audio/wav')},
                    timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
                
                if data.get('status') == 'success' and data.get('result'):
                    result = data['result']
                    self.current_song = result.get('title', 'Unknown')
                    self.current_artist = result.get('artist', 'Unknown')
                    
                    bpm = 0.0
                    spotify = result.get('spotify') or {}
                    if isinstance(spotify, dict):
                        features = spotify.get('audio_features') or {}
                        if isinstance(features, dict):
                            bpm = float(features.get('tempo', 0.0))
                            
                    if bpm == 0.0:
                        apple = result.get('apple_music') or {}
                        if isinstance(apple, dict):
                            bpm = float(apple.get('tempo', 0.0))
                            
                    if bpm == 0.0:
                        bpm = float(result.get('bpm', 0.0))
                        
                    bpm_meta = self._fetch_getsongbpm_fallback(self.current_song, self.current_artist)
                    
                    if bpm_meta:
                        self.time_sig = bpm_meta.get("time_sig", "")
                        self.key_of = bpm_meta.get("key_of", "")
                        self.danceability = bpm_meta.get("danceability", 50)
                        self.acousticness = bpm_meta.get("acousticness", 0)
                        self.genres = bpm_meta.get("genres", [])
                        
                        if bpm == 0.0 and bpm_meta.get("bpm", 0.0) > 0:
                            bpm = bpm_meta.get("bpm")

                    if bpm > 0:
                        print(f"[SongID] ✅ Match: {self.current_artist} — {self.current_song} | {bpm:.1f} BPM")
                        print(f"[SongID] 📊 Vibe: Dance: {self.danceability} | Acoustic: {self.acousticness} | Genres: {self.genres}")
                        self.bpm_override = bpm
                    else:
                        print(f"[SongID] ⚠️ Found track, but no BPM available. Defaulting to reactive fallback.")
                        self.bpm_override = None

                    # If we know the song name, we are LOCKED, even if we don't have a BPM.
                    self.use_fallback = (self.bpm_override is None)
                    self._set_state(IDState.LOCKED)
                        
                else:
                    print("[SongID] ❌ No match found by AudD. Defaulting to reactive fallback.")
                    self.bpm_override = None
                    self.use_fallback = True
                    self._set_state(IDState.FALLBACK)
                    
            except Exception as e:
                print(f"[SongID] ⚠️ Error during identification: {e}")
                self.bpm_override = None
                self.use_fallback = True
                self._set_state(IDState.FALLBACK)
                
            self.task_queue.task_done()