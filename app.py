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
import io
import base64

# ---------------- MARKDOWN -------------
st.markdown("""
<style>
button[data-baseweb="button"] {
    font-size: 32px;
    padding: 30px 60px;
    border-radius: 20px;
    font-weight: bold;
}
button[data-baseweb="button"]:nth-of-type(1) { 
    background-color:#16a34a; color:white; 
}
button[data-baseweb="button"]:nth-of-type(2) { 
    background-color:#dc2626; color:white; 
}
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 4
SAMPLES_FOLDER = "samples" 

# ---------------- CACHING ----------------
@st.cache_resource
def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")

@st.cache_resource
def load_database():
    return build_database_from_folders(SAMPLES_FOLDER)


# ---------------- Load YAMNet ----------------
print("Loading YAMNet model...")
yamnet_model = load_yamnet()
print("YAMNet loaded!")

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
    emb = tf.reduce_mean(embeddings, axis=0).numpy()
    return emb / np.linalg.norm(emb)

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
        # compute distance to each sample
        distances = [cosine(mic_emb, s) for s in samples]
        score = np.mean(sorted(distances)[:3])  # average of 3 closest

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
print("Building command database from files...")
database = load_database()
print(f"Database loaded with {len(database)} commands!")

# ---------------- Streamlit UI ----------------
st.title("🎤 Voice Command Interface")
st.markdown(f"**Known commands:** {', '.join(database.keys())}")
st.write("Click the button and speak your command!")

# Function to play audio immediately using gTTS
def play_audio(text):
    tts = gTTS(text)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    b64 = base64.b64encode(mp3_fp.read()).decode()
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    print(f"Spoken: {text}")  # log to terminal

# ---------------- Voice recognition button ----------------
if st.button("Start Listening"):
    audio = record_from_mic()
    mic_emb = extract_embedding(audio)
    command, distance = identify_command(mic_emb, database)

    if command:
        st.write(f"✅ Recognized command: **{command}** (distance: {distance:.4f})")
        play_audio(command)
    else:
        st.write("❌ Command not recognized.")

# ---------------- Yes/No buttons ----------------
st.markdown("### Quick Responses")
col1, col2 = st.columns(2)
with col1:
    if st.button("Yes"):
        play_audio("Yes")
with col2:
    if st.button("No"):
        play_audio("No")

