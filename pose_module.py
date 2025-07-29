import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=detection_confidence,
                                           min_tracking_confidence=tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_pose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return img
