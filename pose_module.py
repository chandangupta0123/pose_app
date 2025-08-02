# utils/pose_module.py

import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,  # 👈 CRUCIAL for frame skipping
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.prev_center = None

    def detect_pose(self, img, draw_landmarks=True):
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return img, [], None, None, False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        landmarks = []
        center = None
        direction = None
        suspicious = False

        if results.pose_landmarks:
            if draw_landmarks:
                self.mp_draw.draw_landmarks(
                    img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )

            h, w, _ = img.shape
            for id, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id, cx, cy))

            try:
                left_hip = results.pose_landmarks.landmark[23]
                right_hip = results.pose_landmarks.landmark[24]
                center_x = int((left_hip.x + right_hip.x) / 2 * w)
                center_y = int((left_hip.y + right_hip.y) / 2 * h)
                center = (center_x, center_y)

                if self.prev_center:
                    dx = center[0] - self.prev_center[0]
                    dy = center[1] - self.prev_center[1]
                    direction = self.get_direction(dx, dy)
                    movement = np.hypot(dx, dy)
                    suspicious = movement > 50

                self.prev_center = center
            except:
                pass

        return img, landmarks, center, direction, suspicious

    def get_direction(self, dx, dy, threshold=10):
        if abs(dx) > abs(dy):
            return "Right" if dx > threshold else "Left" if dx < -threshold else "Stationary"
        else:
            return "Down" if dy > threshold else "Up" if dy < -threshold else "Stationary"
