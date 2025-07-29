import streamlit as st
import cv2
from utils.pose_module import PoseDetector
import tempfile

st.set_page_config(page_title="Pose Detection App", layout="centered")
st.title("🕺 Real-Time Pose Detection")

# Pose Detector
pose_detector = PoseDetector()

# Upload video or use webcam
source_type = st.radio("Choose Input", ["Webcam", "Upload Video"])

if source_type == "Webcam":
    st.warning("Live webcam support is limited in Streamlit. Upload a short video instead.")
elif source_type == "Upload Video":
    video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

    if video_file:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            frame = pose_detector.detect_pose(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        cap.release()
        st.success("Pose detection completed!")
