# recognize.py
import json
import os
import sounddevice as sd
import numpy as np
import librosa
import soundfile as sf
import whisper

# ===============================
# Configuration
# ===============================
SAMPLE_RATE = 16000
DURATION = 3  # slightly longer for stability
N_MFCC = 13
THRESHOLD = 0.7
MIN_AUDIO_SEC = 2.0  # minimal audio length for DTW stability
MIN_FRAMES = 10      # minimal MFCC frames for DTW

# Load Whisper once globally
print("🔊 Loading Whisper model (small)...")
whisper_model = whisper.load_model("small")
print("✅ Whisper model ready.")

# ===============================
# Audio recording
# ===============================
def record_audio():
    print(f"\n🎤 Press ENTER to start recording for {DURATION}s...")
    input()
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    if np.max(np.abs(audio)) < 1e-4:
        print("⚠️ Audio too quiet. Please speak louder.")
    return audio

# ===============================
# Audio preprocessing
# ===============================
def preprocess_audio(audio):
    if len(audio) == 0:
        raise ValueError("Empty audio input.")
    # Pad short audio
    target_len = int(MIN_AUDIO_SEC * SAMPLE_RATE)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    # Normalize & trim
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    return audio

# ===============================
# Feature extraction
# ===============================
def extract_features(audio):
    if len(audio) == 0:
        raise ValueError("Cannot extract features from empty audio.")
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    if mfcc.shape[1] < 2:
        # pad tiny audio
        mfcc = np.pad(mfcc, ((0,0), (0, MIN_FRAMES - mfcc.shape[1])), mode='edge')
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    return combined.T  # shape: (frames, features)

# ===============================
# DTW similarity
# ===============================
def dtw_similarity(f1, f2):
    from librosa.sequence import dtw
    f1 = np.atleast_2d(f1)
    f2 = np.atleast_2d(f2)
    # pad frames to avoid interpolation issues
    if f1.shape[0] < MIN_FRAMES:
        f1 = np.pad(f1, ((0, MIN_FRAMES - f1.shape[0]), (0, 0)))
    if f2.shape[0] < MIN_FRAMES:
        f2 = np.pad(f2, ((0, MIN_FRAMES - f2.shape[0]), (0, 0)))
    # match feature dimensions
    min_feat = min(f1.shape[1], f2.shape[1])
    f1 = f1[:, :min_feat]
    f2 = f2[:, :min_feat]
    D, wp = dtw(f1, f2, metric='cosine')
    sim = 1 - D[-1, -1] / max(D.shape)
    return sim

# ===============================
# Whisper recognition
# ===============================
def whisper_recognize(audio):
    tmp_path = "temp_command.wav"
    sf.write(tmp_path, audio, SAMPLE_RATE)
    if os.stat(tmp_path).st_size == 0:
        raise RuntimeError("Recorded audio file is empty.")
    result = whisper_model.transcribe(tmp_path, fp16=False, language="en")
    os.remove(tmp_path)
    text = result["text"].strip().lower()
    print(f"🗣 Whisper heard: '{text}'")
    return text

# ===============================
# Prepare sample MFCC (dynamic expansion)
# ===============================
def prepare_sample_mfcc(sample):
    sample = np.array(sample, dtype=np.float64)
    if sample.shape[1] == 13:
        delta = librosa.feature.delta(sample.T).T
        delta2 = librosa.feature.delta(sample.T, order=2).T
        sample = np.hstack([sample, delta, delta2])
    return sample

# ===============================
# Main recognition function
# ===============================
def recognize(audio=None):
    if audio is None:
        audio = record_audio()

    audio = preprocess_audio(audio)

    # --- Step 1: Whisper ---
    recognized_text = whisper_recognize(audio)

    # Load commands
    with open("commands.json", "r") as f:
        commands = json.load(f)

    # Try exact text match
    for cmd in commands.keys():
        if cmd.lower() in recognized_text:
            print(f"✅ Whisper recognized: {cmd}")
            return cmd

    # --- Step 2: DTW fallback ---
    print("🤔 Whisper uncertain — using DTW...")
    features = extract_features(audio)
    best_match = None
    best_score = -1
    second_best = -1

    for cmd, data in commands.items():
        for sample in data["samples"]:
            sample_mfcc = prepare_sample_mfcc(sample)
            sim = dtw_similarity(features, sample_mfcc)
            if sim > best_score:
                second_best = best_score
                best_score = sim
                best_match = cmd

    if best_score - second_best > 0.05 and best_score >= THRESHOLD:
        print(f"✅ DTW recognized: {best_match} (Confidence: {best_score:.2f})")
        return best_match

    print(f"❌ Command not recognized.")
    return None

# ===============================
# Main loop
# ===============================
if __name__ == "__main__":
    print("🎤 Voice command recognizer initialized. Press Ctrl+C to exit.\n")
    while True:
        try:
            recognize()
            print("---")
        except KeyboardInterrupt:
            print("\n👋 Exiting. Goodbye!")
            break
