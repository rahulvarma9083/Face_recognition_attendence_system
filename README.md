# 🟢 Face Recognition Attendance System

A simple **Streamlit + DeepFace** app that marks attendance using face recognition.  
Compatible with **desktop and mobile browsers** (uses `st.camera_input` for camera access).

## 🚀 Features
- Register new faces with names.
- Recognize faces and log attendance with timestamps.
- Works on mobile devices (camera via browser).
- Attendance stored in `attendance.csv`.

## 📂 Project Structure
face-attendance/
│
├── app.py # main Streamlit app
├── requirements.txt # dependencies
├── attendance.csv # auto-created attendance log
├── faces_db/ # database of face images