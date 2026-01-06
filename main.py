import cv2
import os
import numpy as np
import face_recognition
from tqdm import tqdm
from deepface import DeepFace
import time
import mediapipe as mp

base_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(base_dir, "images")
input_path = os.path.join(base_dir, "videos", "input.mp4")
output_path = os.path.join(base_dir, "output", f"output_{int(time.time())}.mp4")
pose_model_path = os.path.join(base_dir, "models", "pose_landmarker_lite.task")

def load_images(path):
    encodings = []
    names = []
    
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                print(f"Processing image: {filename} in folder {folder}")
                image_path = os.path.join(folder_path, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)

                if encoding:
                    encodings.append(encoding[0])
                    names.append(folder)

    return encodings, names

def create_pose_landmarker():
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_segmentation_masks=False
    )

    return mp.tasks.vision.PoseLandmarker.create_from_options(options)

def main():
    print("OpenCV version:", cv2.__version__)
    print("face_recognition version:", face_recognition.__version__)
    print("DeepFace version:", DeepFace.__version__)
    print("mediapipe version:", mp.__version__)

    # Loading face embeding
    known_encodings, known_names = load_images(images_dir)
    print(f"Loaded {len(known_encodings)} face encodings")
    print(f"Names: {known_names}")

    # Initialize video capture
    capture = cv2.VideoCapture(input_path)

    if not capture.isOpened():
        raise Exception("Could not open video")

    # Video properties
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initializing video writer
    output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Processing video
    for frame_index in tqdm(range(total_frames), desc="Processing video"):

        # Read frame
        ret, frame = capture.read()
        if not ret:
            break

        # Transform frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Current frame face detection
        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        # Match known faces embeddings with current faces
        display_names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "unknown"
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
            display_names.append(name)  

        # DeepFace emotion analysis
        df_result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)

        # Draw emotions and recognized faces
        for face in df_result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            emotion = face['dominant_emotion']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            for (top, _right, _bottom, left), name in zip(locations, display_names):
                if x<= left <= x + w and y <= top <= y + h:
                    cv2.putText(frame, name, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    break

        # mediapipe pose detection
        pose_landmarker = create_pose_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_index / fps) * 1000)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw pose landmarks
        if pose_result.pose_landmarks:
            height, width, _ = frame.shape

            for lm in pose_result.pose_landmarks:
                for p in lm:
                    cx = int(p.x * width)
                    cy = int(p.y * height)
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

        output.write(frame)

    capture.release()
    output.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()