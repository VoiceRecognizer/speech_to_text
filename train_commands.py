import sounddevice as sd
import librosa
import numpy as np
import json
import os
import pronouncing

COMMANDS = ["yes", "no"]
SAMPLE_RATE = 16000
DURATION = 2 
COMMANDS_FILE = "commands.json"
SAMPLES = 4

# ------------------------------
# Record audio
# ------------------------------
def record_audio():
    print(f"🎤 Speak now ({DURATION}s)...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten() 

# ------------------------------
# Extract MFCC + delta + delta2
# ------------------------------
def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    return combined.T.astype(float).tolist()  # shape: (frames, 39)

# ------------------------------
# Get phonemes for reference
# ------------------------------
def get_phonemes(word): 
    phones = pronouncing.phones_for_word(word) 
    return phones[0] if phones else ""

# ------------------------------
# Save commands to JSON
# ------------------------------
def save_command(word, features):
    if os.path.exists(COMMANDS_FILE):
        with open(COMMANDS_FILE, "r") as f:
            commands = json.load(f)
    else:
        commands = {}

    if word not in commands:
        commands[word] = {"samples": []}
    commands[word]["samples"].append(features)

    with open(COMMANDS_FILE, "w") as f:
        json.dump(commands, f, indent=2)

# ------------------------------
# Main training loop
# ------------------------------
def main():
    data = {}
    for cmd in COMMANDS:
        print(f"\n--- Recording command: {cmd} ---")
        phonemes = get_phonemes(cmd)
        data[cmd] = {"phonemes": phonemes, "samples": []}
        for i in range(SAMPLES):
            input(f"Press Enter and say '{cmd}' ({i+1}/{SAMPLES})...")
            audio = record_audio()
            features = extract_features(audio)
            data[cmd]["samples"].append(features)

    with open(COMMANDS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print("\n✅ Training data saved to commands.json with MFCC+delta+delta2 features!")

# ------------------------------
# Run script
# ------------------------------
if __name__ == "__main__":
    main()
