import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pydub import AudioSegment
from scipy.spatial.distance import cosine
import sounddevice as sd
import numpy as np
import os


# --- 1. Load YAMNet model ---
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")


# --- 2. Convert ANY audio format to WAV 16k ---
def load_audio_any_format(path, target_sr=16000):
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sr)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.array_type).max
    return samples, target_sr


# --- 3. Extract YAMNet embedding ---
def extract_embedding(audio):
    # TF-Hub YAMNet expects shape [N]
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

    # Run model
    scores, embeddings, spectrogram = yamnet_model(audio_tensor)

    # Average the embeddings across frames
    return tf.reduce_mean(embeddings, axis=0).numpy()


# --- 4. Build database of embeddings ---
def build_database_from_folders(base_folder="samples"):
    db = {}
    for cmd_folder in os.listdir(base_folder):
        cmd_path = os.path.join(base_folder, cmd_folder)
        if not os.path.isdir(cmd_path):
            continue

        db[cmd_folder] = []

        for file in os.listdir(cmd_path):
            file_path = os.path.join(cmd_path, file)
            try:
                audio, sr = load_audio_any_format(file_path)
                emb = extract_embedding(audio)
                db[cmd_folder].append(emb)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
    return db

# --- 5.record from mic
def record_from_mic(seconds=3, sr=16000):
    print(f"🎙 Recording for {seconds} seconds...")

    audio = sd.rec(
        int(seconds * sr),
        samplerate=sr,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    print("✅ Recording done.")
    return audio.flatten(), sr
 
# --- 6. Compare a recording to stored samples ---
def recognize(input_audio, db):
    # using test file
    # audio, sr = load_audio_any_format(input_audio)
    # query_emb = extract_embedding(audio)

    # using audio
    audio, sr = record_from_mic(seconds=3)
    query_emb = extract_embedding(audio)

    best_score = 9999
    best_name = None

    for name, emb in db.items():
        score = cosine(query_emb, emb)
        if score < best_score:
            best_score = score
            best_name = name

    return best_name, 1 - best_score  # confidence


# --- RUN TEST ---
if __name__ == "__main__":
    db = build_database("samples")

    test_file = "./samples/no.wav"   # change if needed
    match, confidence = recognize(test_file, db)

    print("\nBest match:", match)
    print("Confidence:", confidence)
