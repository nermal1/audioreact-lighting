import os
from dotenv import load_dotenv
from core.song_identifier import _fetch_getsongbpm_features

# Load environment variables
load_dotenv()

def test_integration():
    api_key = os.getenv('GETSONGBPM_API_KEY')
    
    if not api_key:
        print("❌ Error: GETSONGBPM_API_KEY not found in your .env file!")
        return

    print("🛰️ Connecting to GetSongBPM API...")
    
    # Let's test a well-known track: Metallica - Enter Sandman
    test_title = "Enter Sandman"
    test_artist = "Metallica"
    
    bpm, key = _fetch_getsongbpm_features(test_title, test_artist, api_key)
    
    print("\n--- API Response Results ---")
    if bpm > 0:
        print(f"✅ Success! Connection verified.")
        print(f"🎵 Track: {test_artist} — {test_title}")
        print(f"🥁 Detected Tempo: {bpm} BPM")
        print(f"🎹 Detected Key: {key if key else 'Not specified'}")
    else:
        print("❌ Failed: The API connected, but returned 0.0 BPM.")
        print("Check if your API key is pasted correctly or if the account is active.")

if __name__ == "__main__":
    test_integration()