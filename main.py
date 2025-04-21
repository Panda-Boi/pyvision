import cv2
from ultralytics import YOLO
import os, random

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

# === Input video file path ===
INPUT_VIDEO_PATH = "input.mp4"  # Replace with your video path

# Check if the input file exists
if not os.path.isfile(INPUT_VIDEO_PATH):
    raise FileNotFoundError(f"Input video file not found: {INPUT_VIDEO_PATH}")

# === Load YOLOv8 model ===
yolo_model = YOLO("yolov8n.pt")  # Swap to yolov8s.pt, yolov8m.pt, etc., for better accuracy

# === Open video file ===
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

OUTPUT_VIDEO_PATH = f"output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

print(f"Processing input: {INPUT_VIDEO_PATH}")
print(f"Saving processed video to: {OUTPUT_VIDEO_PATH}")

tf_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = tf_model.signatures['serving_default']

def detect_pose(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Run detection
    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # 17 keypoints
    return keypoints

def get_keypoints_on_original_image(keypoints, original_shape, target_size=192):
    h, w = original_shape[0], original_shape[1]

    # Compute scale factor and padding used in resize_with_pad
    scale = min(target_size / w, target_size / h)
    new_width = scale * w
    new_height = scale * h
    pad_x = (target_size - new_width) / 2
    pad_y = (target_size - new_height) / 2

    mapped = []
    for y, x, conf in keypoints:
        # Convert from normalized [0,1] coordinates in padded image
        x_px = ((x * target_size - pad_x) / w) / scale
        y_px = ((y * target_size - pad_y) / h)/ scale
        mapped.append([y_px, x_px, conf])

    return np.array(mapped)

def draw_skeleton(frame, keypoints, threshold=0.3):
    h, w, _ = frame.shape
    # Define skeleton connections
    edges = {
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 5), (0, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 6),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    }

    # Draw keypoints
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > threshold:
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)

    # Draw skeleton edges
    for p1, p2 in edges:
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > threshold and c2 > threshold:
            x1_px, y1_px = int(x1 * w), int(y1 * h)
            x2_px, y2_px = int(x2 * w), int(y2 * h)
            cv2.line(frame, (x1_px, y1_px), (x2_px, y2_px), (255, 0, 0), 2)

def is_idle(keypoints_now, keypoints_prev, movement_threshold=0.02):
    diffs = np.linalg.norm(keypoints_now[:, :2] - keypoints_prev[:, :2], axis=1)
    avg_motion = np.mean(diffs)
    return avg_motion < movement_threshold

def is_uniform(person_image) -> bool:
    return 'Wearing Uniform' if (random.randint(0, 1)) else 'Not wearing uniform'

def is_working(person_image) -> bool:
    return 'Working' if (random.randint(0, 1)) else 'Idle'

keypoints_prev = []
keypoints_current = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]

    person_count = 0
    # person_shown = False

    keypoints_prev = keypoints_current

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if yolo_model.names[cls_id] == 'person':
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            person_crop = frame[y1:y2, x1:x2]

            rgb_person = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            keypoints = detect_pose(rgb_person)
            keypoints_mapped = get_keypoints_on_original_image(keypoints, person_crop.shape)
            draw_skeleton(person_crop, keypoints_mapped)

            keypoints_current.append(keypoints_mapped)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img=frame, 
                text=f'Person {person_count}:{conf:.2f}', 
                org=(x1, y1 + 20), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                color=(0, 0, 0), 
                thickness=2
            )
            cv2.putText(
                img=frame, 
                text=is_uniform(person_crop), 
                org=(x1, y1 + 35), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                color=(0, 0, 0), 
                thickness=2
            )
            cv2.putText(
                img=frame, 
                text=is_working(person_crop), 
                org=(x1, y1 + 50), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                color=(0, 0, 0), 
                thickness=2
            )

    count_label = f'People detected: {person_count}'
    cv2.putText(frame, count_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Workplace Analysis', frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:  # ESC to exit early
        break

    # input()

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()