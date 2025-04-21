import time
from typing import List, Tuple, Iterator, Dict
from dataclasses import dataclass

import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from ultralytics import YOLO


@dataclass
class DetectionConfig:
    input_path: str
    output_path: str
    frame_limit: int = 1000
    batch_size: int = 8
    person_confidence: float = 0.5
    helmet_confidence: float = 0.5
    base_model: str = 'yolov8n.pt'
    helmet_model: str = 'best.pt'
    deepsort_max_age: int = 30
    deepsort_n_init: int = 5
    deepsort_confidence: float = 0.5


class BaseDetectionModel:
    def __init__(self, config: DetectionConfig):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(config.base_model).to(self.device)

    def detect(self, frames: List[np.ndarray]):
        return self.model(frames, stream=False, verbose=False)


class HelmetDetectionModel:
    def __init__(self, config: DetectionConfig):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(config.helmet_model).to(self.device)

    def detect(self, image: np.ndarray) -> bool:
        result = self.model.predict(image, stream=False, verbose=False)[0]
        return any(int(b.cls[0]) == 0 and float(b.conf[0]) > config.helmet_confidence for b in result.boxes)


class DeepSortTracker:
    def __init__(self, config: DetectionConfig):
        self.max_age = config.deepsort_max_age
        self.n_init = config.deepsort_n_init
        self.min_confidence = config.deepsort_confidence
        self.tracker = DeepSort(self.max_age, self.n_init, self.min_confidence)


class Visualizer:
    @staticmethod
    def draw_detections(frame: np.ndarray, box: Tuple[int, int, int, int], has_helmet: bool):
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = 'Helmet' if has_helmet else 'No Helmet'

    @staticmethod
    def draw_person_count(frame: np.ndarray, count: int):
        cv2.putText(frame, f'People: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


class VideoProcessor:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.cap = cv2.VideoCapture(config.input_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f'Could not open video file: {config.input_path}')

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.out = cv2.VideoWriter(config.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def read_batches(self) -> Iterator[Tuple[List[np.ndarray], List[int]]]:
        frame_count = 0
        batch_frames = []
        process_indices = []

        with tqdm(total=min(self.total_frames, self.config.frame_limit), desc='Processing frames') as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame_count >= self.config.frame_limit:
                    if batch_frames:
                        yield batch_frames, process_indices
                        pbar.update(len(batch_frames))
                    break

                batch_frames.append(frame)

                if frame_count % 3 == 0:
                    process_indices.append(len(batch_frames) - 1)

                frame_count += 1

                if len(batch_frames) >= self.config.batch_size:
                    yield batch_frames, process_indices
                    pbar.update(len(batch_frames))
                    batch_frames = []
                    process_indices = []
        print()


def interpolate_box(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], factor: float) -> Tuple[int, int, int, int]:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1 = int(x1_1 + factor * (x1_2 - x1_1))
    y1 = int(y1_1 + factor * (y1_2 - y1_1))
    x2 = int(x2_1 + factor * (x2_2 - x2_1))
    y2 = int(y2_1 + factor * (y2_2 - y2_1))
    
    return (x1, y1, x2, y2)


class DetectionSystem:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.person_model = BaseDetectionModel(config)
        self.helmet_model = HelmetDetectionModel(config)
        self.tracker = DeepSortTracker(config).tracker
        self.visualizer = Visualizer()

    def process_video(self):
        print(f'Device: {"cuda" if torch.cuda.is_available() else "cpu"}\n'
              f'Input: {self.config.input_path}\n'
              f'Output: {self.config.output_path}')

        start_time = time.time()
        frame_count = 0

        with VideoProcessor(self.config) as processor:
            for batch, process_indices in processor.read_batches():
                frame_count += len(batch)
                self._process_batch(batch, process_indices, processor)

        total_time = time.time() - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        print(f'FPS: {fps:.2f}')

    def _process_batch(self, batch: List[np.ndarray], process_indices: List[int], processor: VideoProcessor):
        results = self.person_model.detect([batch[i] for i in process_indices])
        person_boxes_per_index = {}
        crops = []
        crop_meta = []

        for idx, (frame_idx, result) in enumerate(zip(process_indices, results)):
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 0 and conf >= self.config.person_confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detections.append((bbox, conf, 'person'))
            person_boxes_per_index[frame_idx] = detections

        tracks_per_index = {}
        for i in process_indices:
            tracks = self.tracker.update_tracks(person_boxes_per_index[i], frame=batch[i])
            track_meta = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(processor.width, x2), min(processor.height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = batch[i][y1:y2, x1:x2]
                crops.append(crop)
                crop_meta.append((i, track_id, (x1, y1, x2, y2)))
                track_meta.append((track_id, (x1, y1, x2, y2)))
            tracks_per_index[i] = track_meta

        helmet_results = self.helmet_model.model(crops, stream=False, verbose=False)
        helmet_flags = []
        for result in helmet_results:
            helmet_flags.append(any(int(b.cls[0]) == 0 and float(b.conf[0]) > self.config.helmet_confidence for b in result.boxes))

        helmet_per_person = {}
        for (frame_idx, track_id, box), has_helmet in zip(crop_meta, helmet_flags):
            if frame_idx not in helmet_per_person:
                helmet_per_person[frame_idx] = {}
            helmet_per_person[frame_idx][track_id] = (box, has_helmet)

        track_history = {}
        
        for frame_idx, detections in helmet_per_person.items():
            for track_id, (box, has_helmet) in detections.items():
                if track_id not in track_history:
                    track_history[track_id] = {}
                track_history[track_id][frame_idx] = (box, has_helmet)
        
        interpolated_detections = {}
        for i in range(len(batch)):
            interpolated_detections[i] = {}
            
            if i in helmet_per_person:
                interpolated_detections[i] = helmet_per_person[i]
                continue
                
            for track_id, frame_data in track_history.items():
                prev_frame = None
                next_frame = None
                
                for proc_idx in sorted(frame_data.keys()):
                    if proc_idx < i:
                        prev_frame = proc_idx
                    elif proc_idx > i:
                        next_frame = proc_idx
                        break
                
                if prev_frame is not None and next_frame is not None:
                    prev_box, prev_helmet = frame_data[prev_frame]
                    next_box, next_helmet = frame_data[next_frame]
                    
                    total_gap = next_frame - prev_frame
                    current_pos = i - prev_frame
                    factor = current_pos / total_gap if total_gap > 0 else 0
                    
                    interpolated_box = interpolate_box(prev_box, next_box, factor)
                    
                    nearest_helmet = prev_helmet if (i - prev_frame < next_frame - i) else next_helmet
                    
                    interpolated_detections[i][track_id] = (interpolated_box, nearest_helmet)
                
                elif prev_frame is not None and prev_frame == max(frame_data.keys()):
                    if len(frame_data) >= 2:
                        sorted_frames = sorted(frame_data.keys())
                        if len(sorted_frames) >= 2:
                            prev_prev_frame = sorted_frames[-2]
                            prev_box, prev_helmet = frame_data[prev_frame]
                            prev_prev_box, _ = frame_data[prev_prev_frame]
                            
                            frame_diff = prev_frame - prev_prev_frame
                            if frame_diff > 0:
                                factor = (i - prev_frame) / frame_diff
                                predicted_box = interpolate_box(prev_prev_box, prev_box, 1 + factor)
                                interpolated_detections[i][track_id] = (predicted_box, prev_helmet)
                            else:
                                interpolated_detections[i][track_id] = (prev_box, prev_helmet)
                        else:
                            interpolated_detections[i][track_id] = frame_data[prev_frame]
                    else:
                        interpolated_detections[i][track_id] = frame_data[prev_frame]

        for i, frame in enumerate(batch):
            detections = interpolated_detections.get(i, {})
            person_count = len(detections)

            for track_id, (box, has_helmet) in detections.items():
                x1, y1, x2, y2 = box
                self.visualizer.draw_detections(frame, box, has_helmet)
                cv2.putText(frame, f'ID: {track_id}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            self.visualizer.draw_person_count(frame, person_count)
            processor.out.write(frame)


config = DetectionConfig(
    input_path='input.mp4',
    output_path='output.mp4',
    batch_size=16,
    person_confidence=0.5,
    helmet_confidence=0.7,
    base_model='yolov8n.pt',
    helmet_model='s640.pt',
    deepsort_max_age=90,
    deepsort_n_init=10,
    deepsort_confidence=0.6
)

if __name__ == '__main__':
    system = DetectionSystem(config)
    system.process_video()
