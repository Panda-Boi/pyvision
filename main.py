import time
from typing import List, Tuple, Iterator
from dataclasses import dataclass

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


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
        return any(int(b.cls[0]) == 0 and float(b.conf[0]) > 0.5 for b in result.boxes)


class Visualizer:
    @staticmethod
    def draw_detections(frame: np.ndarray, box: Tuple[int, int, int, int], has_helmet: bool, confidence: float):
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = 'Helmet' if has_helmet else 'No Helmet'
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

    def read_batches(self) -> Iterator[List[np.ndarray]]:
        frame_count = 0
        batch_frames = []

        with tqdm(total=min(self.total_frames, self.config.frame_limit), desc='Processing frames') as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame_count >= self.config.frame_limit:
                    if batch_frames:
                        yield batch_frames
                        pbar.update(len(batch_frames))
                    break

                batch_frames.append(frame)
                frame_count += 1

                if len(batch_frames) >= self.config.batch_size:
                    yield batch_frames
                    pbar.update(len(batch_frames))
                    batch_frames = []
        print()


class DetectionSystem:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.person_model = BaseDetectionModel(config)
        self.helmet_model = HelmetDetectionModel(config)
        self.visualizer = Visualizer()

    def process_video(self):
        print(f'Device: {"cuda" if torch.cuda.is_available() else "cpu"}\n'
              f'Input: {self.config.input_path}\n'
              f'Output: {self.config.output_path}')

        start_time = time.time()
        frame_count = 0

        with VideoProcessor(self.config) as processor:
            for batch in processor.read_batches():
                frame_count += len(batch)
                self._process_batch(batch, processor)

        total_time = time.time() - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        print(f'FPS: {fps:.2f}')

    def _process_batch(self, batch: List[np.ndarray], processor: VideoProcessor):
        results = self.person_model.detect(batch)

        for frame, result in zip(batch, results):
            person_count = 0

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id != 0 or conf < self.config.person_confidence:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_count += 1
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(processor.width, x2), min(processor.height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                has_helmet = self.helmet_model.detect(crop)
                self.visualizer.draw_detections(frame, (x1, y1, x2, y2), has_helmet, conf)

            self.visualizer.draw_person_count(frame, person_count)
            processor.out.write(frame)


config = DetectionConfig(
    input_path='input.mp4',
    output_path='output.mp4',
    batch_size=16,
    person_confidence=0.5,
    helmet_confidence=0.7,
    base_model='yolov8n.pt',
    helmet_model='s640.pt'
)

if __name__ == '__main__':
    system = DetectionSystem(config)
    system.process_video()
