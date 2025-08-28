import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.title("Full-Body Stick Figure Tracker with Hands and Face")

# Sidebar options
show_body = st.sidebar.checkbox("Show Body", True)
show_hands = st.sidebar.checkbox("Show Hands", True)
show_face = st.sidebar.checkbox("Show Face", True)
start_camera = st.sidebar.button("Start Camera")
stop_camera = st.sidebar.button("Stop Camera")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# Connections for body stick figure
body_connections = [
    (0, 11), (0, 12),        # Nose to shoulders
    (11, 13), (13, 15),       # Left arm
    (12, 14), (14, 16),       # Right arm
    (11, 23), (12, 24),       # Shoulders to hips
    (23, 24),                 # Hip line
    (23, 25), (25, 27),       # Left leg
    (24, 26), (26, 28)        # Right leg
]

# Connections for hands
hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# Streamlit image placeholder
frame_placeholder = st.empty()
fps_placeholder = st.empty()

# Only start camera if button clicked
if start_camera:
    cap = cv2.VideoCapture(0)
    prev_time = 0
    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # --- Process Pose ---
        pose_results = pose.process(frame_rgb)
        if show_body and pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            for connection in body_connections:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # --- Process Hands ---
        hand_results = hands.process(frame_rgb)
        if show_hands and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for connection in hand_connections:
                    start = hand_landmarks.landmark[connection[0]]
                    end = hand_landmarks.landmark[connection[1]]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

        # --- Process Face ---
        face_results = face_mesh.process(frame_rgb)
        if show_face and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
                )

        # --- Calculate FPS ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        fps_placeholder.text(f"FPS: {int(fps)}")

        # Stop camera if stop button pressed
        if stop_camera:
            running = False

    cap.release()
    cv2.destroyAllWindows()
