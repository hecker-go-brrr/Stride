import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

st.title("Side-by-Side Full-Body + Hands + Face Stick Figure Video")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    st.video(input_path, start_time=0)
    
    # Button to start processing
    if st.button("Process Video"):
        st.info("Processing video... This may take a while.")

        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_face = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils

        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
        face_mesh = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

        # Stick figure connections
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

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(tempfile.gettempdir(), "output_full.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stick_frame = frame.copy()

            # --- Process Pose ---
            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                for connection in body_connections:
                    start = landmarks[connection[0]]
                    end = landmarks[connection[1]]
                    x1, y1 = int(start.x * width), int(start.y * height)
                    x2, y2 = int(end.x * width), int(end.y * height)
                    cv2.line(stick_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                for lm in landmarks:
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    cv2.circle(stick_frame, (cx, cy), 5, (0, 255, 0), -1)

            # --- Process Hands ---
            hand_results = hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for connection in hand_connections:
                        start = hand_landmarks.landmark[connection[0]]
                        end = hand_landmarks.landmark[connection[1]]
                        x1, y1 = int(start.x * width), int(start.y * height)
                        x2, y2 = int(end.x * width), int(end.y * height)
                        cv2.line(stick_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    for lm in hand_landmarks.landmark:
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        cv2.circle(stick_frame, (cx, cy), 3, (255, 0, 0), -1)

            # --- Process Face ---
            face_results = face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        stick_frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
                    )

            # Combine original + stick figure
            combined = cv2.hconcat([frame, stick_frame])
            out.write(combined)

        cap.release()
        out.release()
        st.success("Processing finished!")

        st.video(output_path)

        # Download button
        with open(output_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="stick_figure_full.mp4", mime="video/mp4")
