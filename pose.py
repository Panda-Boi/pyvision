import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

tf_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = tf_model.signatures['serving_default']

def state(keypoints_now, keypoints_prev, prev_state):

    KEYPOINTS_THRESHOLD = 10

    if len(keypoints_now) < KEYPOINTS_THRESHOLD:
        return prev_state

    if is_idle(keypoints_now, keypoints_prev):
        return 'Idle'
    else:
        return 'Working'

def is_idle(keypoints_now, keypoints_prev, movement_threshold=0.1):
    if not keypoints_prev.all():
        return True

    diffs = np.linalg.norm(keypoints_now[:, :2] - keypoints_prev[:, :2], axis=1)
    avg_motion = np.mean(diffs)
    return avg_motion < movement_threshold

def detect_pose(frame: np.ndarray, target_size=192) -> np.ndarray:
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), target_size, target_size)
    input_img = tf.cast(img, dtype=tf.int32)

    # Run detection
    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # 17 keypoints
    
    # map keypoints to original image
    keypoints = _get_keypoints_on_original_image(keypoints, frame.shape, target_size)
    return keypoints

def _get_keypoints_on_original_image(keypoints: np.ndarray, original_shape, target_size=192) -> np.ndarray:
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