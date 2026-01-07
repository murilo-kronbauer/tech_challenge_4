import cv2
import os
import numpy as np
import face_recognition
from tqdm import tqdm
from deepface import DeepFace
import time
import mediapipe as mp
from collections import deque

base_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(base_dir, "images")
input_path = os.path.join(base_dir, "videos", "unlocking_facial_recognition.mp4")
output_path = os.path.join(base_dir, "output", f"output_{int(time.time())}.mp4")
pose_model_path = os.path.join(base_dir, "models", "pose_landmarker_heavy.task")

COLORS = [
    (255,   0,   0),  # vermelho
    (0,   255,   0),  # verde
    (0,     0, 255),  # azul
    (255, 255,   0),  # ciano
    (255,   0, 255),  # magenta
    (0,   255, 255),  # amarelo
]

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
        output_segmentation_masks=False,
        num_poses=3
    )

    return mp.tasks.vision.PoseLandmarker.create_from_options(options)

def detect_wave(wrist_history, shoulder_y, elbow_y, min_oscillations=2, threshold=0.03):
    """
    Detect waving motion by analyzing wrist vertical movement.
    Returns True if a wave is detected.
    """
    if len(wrist_history) < 10:
        return False
    
    wrist_y_values = [pos[1] for pos in wrist_history]
    
    # Check if hand is raised (wrist above elbow)
    if wrist_y_values[-1] > elbow_y:
        return False
    
    # Count direction changes (oscillations)
    direction_changes = 0
    for i in range(1, len(wrist_y_values) - 1):
        if abs(wrist_y_values[i] - wrist_y_values[i-1]) > threshold:
            curr_direction = wrist_y_values[i] - wrist_y_values[i-1]
            next_direction = wrist_y_values[i+1] - wrist_y_values[i]
            
            if curr_direction * next_direction < 0:
                direction_changes += 1
    
    return direction_changes >= min_oscillations

def detect_handshake(person1_landmarks, person2_landmarks, proximity_threshold=0.15):
    """
    Detect handshake between two people.
    Returns True if hands are close and at similar height.
    """
    # Right wrist indices: 16, Left wrist: 15
    # Right elbow: 14, Left elbow: 13
    
    # Get wrist positions for both people
    p1_right_wrist = person1_landmarks[16]
    p1_left_wrist = person1_landmarks[15]
    p2_right_wrist = person2_landmarks[16]
    p2_left_wrist = person2_landmarks[15]
    
    # Check all hand combinations
    hand_pairs = [
        (p1_right_wrist, p2_right_wrist),
        (p1_right_wrist, p2_left_wrist),
        (p1_left_wrist, p2_right_wrist),
        (p1_left_wrist, p2_left_wrist)
    ]
    
    for hand1, hand2 in hand_pairs:
        # Calculate 3D distance
        distance = np.sqrt(
            (hand1.x - hand2.x)**2 + 
            (hand1.y - hand2.y)**2 + 
            (hand1.z - hand2.z)**2
        )
        
        # Check if hands are close
        if distance < proximity_threshold:
            # Check if at similar height (y-coordinate)
            height_diff = abs(hand1.y - hand2.y)
            if height_diff < 0.1:
                return True
    
    return False

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

    # Initialize pose landmarker
    pose_landmarker = create_pose_landmarker()

    # Track wrist positions for wave detection (per person)
    wrist_histories = {}
    wave_cooldown = {}
    handshake_frames = 0
    handshake_detected = False

    # Processing video
    for frame_index in tqdm(range(total_frames), desc="Processing video"):

        # Read frame
        ret, frame = capture.read()
        if not ret:
            break

        # Skip every other frame to speed up processing
        if frame_index % 2 != 0:
            output.write(frame)
            continue            

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
        df_result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False, detector_backend='retinaface')

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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_index / fps) * 1000)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if pose_result.pose_landmarks:
            # Wave detection for each person
            for person_idx, lm in enumerate(pose_result.pose_landmarks):
                color = COLORS[person_idx % len(COLORS)]
                
                # Initialize history for this person
                if person_idx not in wrist_histories:
                    wrist_histories[person_idx] = {'right': deque(maxlen=15), 'left': deque(maxlen=15)}
                    wave_cooldown[person_idx] = {'right': 0, 'left': 0}
                
                # Get key landmarks (MediaPipe indices)
                right_wrist = lm[16]  # Right wrist
                left_wrist = lm[15]   # Left wrist
                right_shoulder = lm[12]
                left_shoulder = lm[11]
                right_elbow = lm[14]
                left_elbow = lm[13]
                
                # Track wrist positions
                wrist_histories[person_idx]['right'].append((right_wrist.x, right_wrist.y, right_wrist.z))
                wrist_histories[person_idx]['left'].append((left_wrist.x, left_wrist.y, left_wrist.z))
                
                # Detect wave for right hand
                if wave_cooldown[person_idx]['right'] == 0:
                    if detect_wave(wrist_histories[person_idx]['right'], right_shoulder.y, right_elbow.y):
                        cv2.putText(frame, f"Person {person_idx+1}: WAVING (R)", 
                                    (10, 30 + person_idx * 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        wave_cooldown[person_idx]['right'] = fps  # Cooldown for 1 second
                
                # Detect wave for left hand
                if wave_cooldown[person_idx]['left'] == 0:
                    if detect_wave(wrist_histories[person_idx]['left'], left_shoulder.y, left_elbow.y):
                        cv2.putText(frame, f"Person {person_idx+1}: WAVING (L)", 
                                    (10, 60 + person_idx * 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        wave_cooldown[person_idx]['left'] = fps
                
                # Decrease cooldown
                if wave_cooldown[person_idx]['right'] > 0:
                    wave_cooldown[person_idx]['right'] -= 1
                if wave_cooldown[person_idx]['left'] > 0:
                    wave_cooldown[person_idx]['left'] -= 1
                
                # draw landmarks
                for p in lm:
                    cx = int(p.x * width)
                    cy = int(p.y * height)
                    cv2.circle(frame, (cx, cy), 3, color, 1)
            
            # Handshake detection (requires at least 2 people)
            if len(pose_result.pose_landmarks) >= 2:
                handshake_found = False
                for i in range(len(pose_result.pose_landmarks)):
                    for j in range(i + 1, len(pose_result.pose_landmarks)):
                        if detect_handshake(pose_result.pose_landmarks[i], pose_result.pose_landmarks[j]):
                            handshake_found = True
                            handshake_frames += 1
                            break
                    if handshake_found:
                        break
                
                if not handshake_found:
                    handshake_frames = 0
                    handshake_detected = False
                
                # Confirm handshake if detected for multiple consecutive frames
                if handshake_frames > fps * 0.5 and not handshake_detected:
                    handshake_detected = True
                
                if handshake_detected and handshake_frames > 0:
                    cv2.putText(frame, "HANDSHAKE DETECTED!", 
                                (width // 2 - 150, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                handshake_frames = 0
                handshake_detected = False

        output.write(frame)

    capture.release()
    output.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()