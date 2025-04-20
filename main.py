import cv2
from ultralytics import YOLO
import os, random

# === Input video file path ===
INPUT_VIDEO_PATH = "input.mp4"  # Replace with your video path

# Check if the input file exists
if not os.path.isfile(INPUT_VIDEO_PATH):
    raise FileNotFoundError(f"Input video file not found: {INPUT_VIDEO_PATH}")

# === Load YOLOv8 model ===
model = YOLO("yolov8n.pt")  # Swap to yolov8s.pt, yolov8m.pt, etc., for better accuracy

# === Open video file ===
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

OUTPUT_VIDEO_PATH = f"output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

print(f"Processing input: {INPUT_VIDEO_PATH}")
print(f"Saving processed video to: {OUTPUT_VIDEO_PATH}")

def is_uniform(person_image) -> bool:
    return 'Wearing Uniform' if (random.randint(0, 1)) else 'Not wearing uniform'

def is_working(person_image) -> bool:
    return 'Working' if (random.randint(0, 1)) else 'Idle'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    person_count = 0
    person_shown = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if model.names[cls_id] == 'person':
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f'Person {conf:.2f}'

            person_crop = frame[y1:y2, x1:x2]

            # showing 1 persons video stream
            if not person_shown:
                cv2.imshow("Cropped Person", person_crop)
                person_shown = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img=frame, 
                text=label, 
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

    # cv2.imshow('Workplace Analysis', frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:  # ESC to exit early
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()