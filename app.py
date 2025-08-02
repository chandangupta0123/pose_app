# app.py

import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd
from utils.pose_module import PoseDetector

st.set_page_config(page_title="Pose Detection", layout="wide")
st.title("🎥 Human Movement & Suspicious Activity Detection")

with st.sidebar:
    st.header("Settings")
    detection_conf = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
    tracking_conf = st.slider("Tracking Confidence", 0.1, 1.0, 0.5, 0.1)
    skip_frames = st.slider("Skip every N frames", 1, 10, 2)
    show_landmarks = st.checkbox("Show Pose Landmarks", value=True)

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    width, height = 480, 360
    output_path = os.path.join(tempfile.gettempdir(), "output_processed.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    detector = PoseDetector(detection_conf, tracking_conf)
    movement_log = []

    stframe = st.empty()
    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    st.info("⏳ Processing...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_num % skip_frames != 0:
            frame_num += 1
            continue

        frame = cv2.resize(frame, (width, height))
        annotated_frame, landmarks, center, direction, suspicious = detector.detect_pose(
            frame, draw_landmarks=show_landmarks
        )

        if direction:
            cv2.putText(annotated_frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if suspicious:
            cv2.putText(annotated_frame, "⚠ Suspicious Movement", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        movement_log.append({
            "frame": frame_num,
            "center_x": center[0] if center else None,
            "center_y": center[1] if center else None,
            "direction": direction,
            "suspicious": suspicious
        })

        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        out.write(annotated_frame)
        progress.progress(min(frame_num / total_frames, 1.0))
        frame_num += 1

    cap.release()
    out.release()
    st.success("✅ Done!")

    # Report
    df = pd.DataFrame(movement_log)
    st.subheader("📊 Movement Log")
    st.dataframe(df.tail(10))
    st.download_button("📥 Download Log as CSV", df.to_csv(index=False), "movement_log.csv")

    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download Processed Video", f, file_name="output_video.mp4")

