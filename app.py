import streamlit as st
import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import tempfile
import base64
import numpy as np
import io
import wave

# Paths
db_path = "faces_db"
attendance_file = "attendance.csv"
os.makedirs(db_path, exist_ok=True)

# Ensure CSV exists and has headers
if not os.path.exists(attendance_file) or os.path.getsize(attendance_file) == 0:
    df = pd.DataFrame(columns=["ID", "Name", "Timestamp"])
    df.to_csv(attendance_file, index=False)

# Session state
if "marked_names" not in st.session_state:
    st.session_state["marked_names"] = set()
if "new_face" not in st.session_state:
    st.session_state["new_face"] = None
if "last_face_img" not in st.session_state:
    st.session_state["last_face_img"] = None


# ----------- Sound Generation -----------
def generate_beep(frequency=440, duration=0.3, volume=0.5, samplerate=44100):
    """Generate a sine wave beep and return base64 WAV string"""
    t = np.linspace(0, duration, int(samplerate * duration), False)
    wave_data = (volume * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    # Save to buffer as WAV
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((wave_data * 32767).astype(np.int16).tobytes())

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64


def play_sound(frequency=440, duration=0.3):
    """Embed sound into Streamlit frontend"""
    sound_b64 = generate_beep(frequency, duration)
    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{sound_b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )


# ----------- Core Functions -----------
def get_next_id():
    existing = [f for f in os.listdir(db_path) if f.endswith(".jpg")]
    return str(len(existing) + 1).zfill(3)


def mark_attendance(face_id, name, face_img):
    if name not in st.session_state["marked_names"]:
        df = pd.read_csv(attendance_file)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [[face_id, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]],
                    columns=["ID", "Name", "Timestamp"],
                ),
            ]
        )
        df.to_csv(attendance_file, index=False)
        st.session_state["marked_names"].add(name)
        st.success(f"Face detected for {name}")
        st.session_state["last_face_img"] = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # 🔊 Play success beep
        play_sound(frequency=880, duration=0.2)


def scan_face():
    # Streamlit camera input (works on mobile & desktop browsers)
    img_file = st.camera_input("📷 Capture your face")

    if img_file is None:
        st.warning("Please take a picture to continue.")
        return

    # Convert to OpenCV image
    bytes_data = img_file.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save temp file for DeepFace
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, frame)

    face_files = [f for f in os.listdir(db_path) if f.endswith(".jpg")]

    try:
        if len(face_files) == 0:
            st.info("🆕 New face detected (database empty).")
            st.session_state["new_face"] = frame
            play_sound(frequency=440, duration=0.3)
            return

        # Run recognition
        result = DeepFace.find(
            img_path=temp_file.name,
            db_path=db_path,
            model_name="Facenet512",
            detector_backend="retinaface",  # better on selfies
            enforce_detection=False,
            distance_metric="cosine",
        )

        if len(result) > 0 and not result[0].empty:
            matched_file = os.path.basename(result[0].iloc[0]["identity"])
            matched_id, matched_name = os.path.splitext(matched_file)[0].split("_", 1)

            st.success(f"✅ Face matched: {matched_name}")
            mark_attendance(matched_id, matched_name, frame)
            st.session_state["new_face"] = None
        else:
            st.info("🆕 New face detected")
            st.session_state["new_face"] = frame
            play_sound(frequency=440, duration=1.0)

    except Exception as e:
        st.error(f"⚠ Error during recognition: {e}")
        st.session_state["new_face"] = None


# ----------- Streamlit UI -------------
st.title("🟢 Face Identification & Attendance")

scan_face()

# If new face detected, prompt for name
if st.session_state["new_face"] is not None:
    name = st.text_input("Enter name for the new face:")
    if name:
        new_id = get_next_id()
        filename = f"{new_id}_{name}.jpg"
        path = os.path.join(db_path, filename)
        cv2.imwrite(path, st.session_state["new_face"])
        st.success(f"✅ New face saved as {filename}")
        mark_attendance(new_id, name, st.session_state["new_face"])
        st.session_state["new_face"] = None

# Display last captured face
if st.session_state["last_face_img"] is not None:
    st.image(
        st.session_state["last_face_img"],
        caption="Last scanned face",
        use_container_width=True,
    )

# Attendance table
if st.checkbox("Show Attendance Records"):
    try:
        df_attendance = pd.read_csv(attendance_file)
        st.dataframe(df_attendance)
    except pd.errors.EmptyDataError:
        st.warning("Attendance file is empty.")
