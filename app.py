import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import os
from scipy.spatial.distance import cosine
from gtts import gTTS
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from pydub import AudioSegment

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 4
SAMPLES_FOLDER = "samples" 

# ---------------- Load YAMNet ----------------
st.info("Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
st.success("YAMNet loaded!")

# ---------------- Helpers ----------------
def load_audio_any_format(path, target_sr=SAMPLE_RATE):
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.array_type).max
    return samples

def extract_embedding(audio):
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(audio_tensor)
    return tf.reduce_mean(embeddings, axis=0).numpy()

def build_database_from_folders(base_folder=SAMPLES_FOLDER):
    db = {}
    for cmd_folder in os.listdir(base_folder):
        cmd_path = os.path.join(base_folder, cmd_folder)
        if not os.path.isdir(cmd_path):
            continue
        db[cmd_folder] = []
        for file in os.listdir(cmd_path):
            file_path = os.path.join(cmd_path, file)
            try:
                audio = load_audio_any_format(file_path)
                emb = extract_embedding(audio)
                db[cmd_folder].append(emb)
            except Exception as e:
                st.warning(f"Failed to process {file_path}: {e}")
    return db

def average_embedding(samples):
    return np.mean(samples, axis=0)

def identify_command(mic_emb, db):
    best_cmd = None
    best_score = 999
    for cmd, samples in db.items():
        if len(samples) == 0:
            continue
        avg_emb = average_embedding(samples)
        score = cosine(mic_emb, avg_emb)
        if score < best_score:
            best_score = score
            best_cmd = cmd
    return best_cmd, best_score

def record_from_mic(seconds=DURATION, sr=SAMPLE_RATE):
    st.info(f"Recording for {seconds} seconds...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording finished!")
    return audio.flatten()

# ---------------- Build database ----------------
st.info("Building command database from files...")
database = build_database_from_folders(SAMPLES_FOLDER)
st.success(f"Database loaded with {len(database)} commands!")

# ---------------- Streamlit UI ----------------
st.title("🎤 Voice Command Interface")
st.markdown(f"**Known commands:** {', '.join(database.keys())}")
st.write("Click the button and speak your command!")

if st.button("Start Listening"):
    audio = record_from_mic()
    mic_emb = extract_embedding(audio)
    command, distance = identify_command(mic_emb, database)

    if command:
        st.write(f"✅ Recognized command: **{command}** (distance: {distance:.4f})")
        # Text-to-speech via gTTS
        tts = gTTS(command)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")
        tmp_file.close()
        os.unlink(tmp_file.name)
    else:
        st.write("❌ Command not recognized.")
