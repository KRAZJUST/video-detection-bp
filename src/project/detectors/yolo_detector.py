""" Module to perform object detection using YOLOv11. """

from typing import Any, Dict, List
from ultralytics import YOLO
from .color_filter import ColorFilter
from .detection import Detection
import torch
import numpy as np

class YOLODetector:
    """ Class to perform object detection using YOLOv11. """

    def __init__(self, model_path: str = 'yolo11n.pt', device: str = 'auto'):
        self.model = YOLO(model_path)
        self.color_filter = ColorFilter()
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.class_names = self.model.names

    def detect_objects(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results = self.model.predict(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls[0])
                class_name = self.class_names.get(class_idx, "Unknown")
                # Detect only people and vehicles
                if class_name in ['person', 'car', 'truck', 'bus']:
                    confidence = float(box.conf[0])
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(xmin, ymin, xmax, ymax),
                        timestamp=timestamp,
                        dominant_color=self.color_filter.detect_dominant_color(frame, (xmin, ymin, xmax, ymax))
                    ))

        return results[0], detections