import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
from collections import deque

st.title("Exercise Mirror with Full-Body + Hands Similarity")

# Sidebar options
show_body = st.sidebar.checkbox("Show Body", True)
show_hands = st.sidebar.checkbox("Show Hands", True)
show_face = st.sidebar.checkbox("Show Face", True)
start_camera = st.sidebar.button("Start Camera")
stop_camera = st.sidebar.button("Stop Camera")

# Upload reference video
uploaded_file = st.file_uploader("Upload reference video", type=["mp4", "mov", "avi"])

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# Connections
body_connections = [
    (0, 11), (0, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]

hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

frame_placeholder = st.empty()
fps_placeholder = st.empty()

# --- Similarity function ---
def calculate_similarity(ref_kpts, webcam_kpts):
    """
    Compute similarity between two sets of keypoints.
    - Normalize keypoints by torso (shoulders+hips) if available
    - Use cosine similarity for vectors between keypoints
    """
    ref_kpts = np.array(ref_kpts)
    webcam_kpts = np.array(webcam_kpts)
    
    min_len = min(len(ref_kpts), len(webcam_kpts))
    if min_len < 4:
        return 0  # Not enough points to compare
    
    ref_kpts = ref_kpts[:min_len]
    webcam_kpts = webcam_kpts[:min_len]
    
    # Normalize by torso if there are enough body points
    torso_idx = [11,12,23,24]
    if max(torso_idx) < min_len:
        torso_ref = ref_kpts[torso_idx]
        torso_webcam = webcam_kpts[torso_idx]
        ref_center = np.mean(torso_ref, axis=0)
        webcam_center = np.mean(torso_webcam, axis=0)
        ref_norm = ref_kpts - ref_center
        webcam_norm = webcam_kpts - webcam_center
        ref_size = np.linalg.norm(torso_ref[0] - torso_ref[1])
        webcam_size = np.linalg.norm(torso_webcam[0] - torso_webcam[1])
        if ref_size > 0 and webcam_size > 0:
            ref_norm /= ref_size
            webcam_norm /= webcam_size
    else:
        # Just normalize by mean
        ref_center = np.mean(ref_kpts, axis=0)
        webcam_center = np.mean(webcam_kpts, axis=0)
        ref_norm = ref_kpts - ref_center
        webcam_norm = webcam_kpts - webcam_center

    # Cosine similarity
    dot = np.sum(ref_norm * webcam_norm, axis=1)
    norms = np.linalg.norm(ref_norm, axis=1) * np.linalg.norm(webcam_norm, axis=1)
    cos_sim = np.where(norms > 0, dot / norms, 0)
    return np.clip(np.mean(cos_sim) * 100, 0, 100)

# --- Load reference video frames ---
reference_kpts_list = []

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    ref_path = tfile.name

    cap_ref = cv2.VideoCapture(ref_path)
    st.info("Processing reference video...")

    while cap_ref.isOpened():
        ret, frame = cap_ref.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kpts_frame = []

        pose_results = pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                kpts_frame.append([lm.x, lm.y])

        hand_results = hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    kpts_frame.append([lm.x, lm.y])

        if kpts_frame:
            reference_kpts_list.append(np.array(kpts_frame))
    cap_ref.release()
    st.success(f"Reference video processed: {len(reference_kpts_list)} frames")

# --- Webcam processing ---
if start_camera and reference_kpts_list:
    cap = cv2.VideoCapture(0)
    prev_time = 0
    running = True
    frame_index = 0
    smooth_window = 5
    match_deque = deque(maxlen=smooth_window)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        stick_frame = frame.copy()
        webcam_kpts = []

        # --- Pose ---
        pose_results = pose.process(frame_rgb)
        if show_body and pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            for connection in body_connections:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(stick_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(stick_frame, (cx, cy), 5, (0, 255, 0), -1)
                webcam_kpts.append([lm.x, lm.y])

        # --- Hands ---
        hand_results = hands.process(frame_rgb)
        if show_hands and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for connection in hand_connections:
                    start = hand_landmarks.landmark[connection[0]]
                    end = hand_landmarks.landmark[connection[1]]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(stick_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(stick_frame, (cx, cy), 3, (255, 0, 0), -1)
                    webcam_kpts.append([lm.x, lm.y])

        # --- Similarity calculation ---
        if webcam_kpts:
            ref_kpts_draw = reference_kpts_list[frame_index]
            match_percent = calculate_similarity(ref_kpts_draw, webcam_kpts)
            match_deque.append(match_percent)
            smoothed_match = np.mean(match_deque)
        else:
            smoothed_match = 0

        # --- Draw reference stick figure ---
        ref_frame = np.zeros_like(stick_frame)
        ref_kpts_draw = reference_kpts_list[frame_index]
        for connection in body_connections:
            if connection[0] >= len(ref_kpts_draw) or connection[1] >= len(ref_kpts_draw):
                continue
            start = ref_kpts_draw[connection[0]]
            end = ref_kpts_draw[connection[1]]
            x1, y1 = int(start[0] * w), int(start[1] * h)
            x2, y2 = int(end[0] * w), int(end[1] * h)
            cv2.line(ref_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # Combine reference and webcam
        combined = cv2.hconcat([ref_frame, stick_frame])

        # --- Draw similarity bar ---
        bar_width = int((smoothed_match / 100) * combined.shape[1])
        bar_height = 20
        cv2.rectangle(combined, (0, 0), (combined.shape[1], bar_height), (50, 50, 50), -1)
        cv2.rectangle(combined, (0, 0), (bar_width, bar_height), (0, 255, 0), -1)
        cv2.putText(combined, f"Match: {int(smoothed_match)}%", (10, bar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- FPS ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(combined, f'FPS: {int(fps)}', (10, bar_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display
        frame_placeholder.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), channels="RGB")
        fps_placeholder.text(f"FPS: {int(fps)}")

        # Loop reference video
        frame_index += 1
        if frame_index >= len(reference_kpts_list):
            frame_index = 0

        # Stop camera
        if stop_camera:
            running = False

    cap.release()
    cv2.destroyAllWindows()
