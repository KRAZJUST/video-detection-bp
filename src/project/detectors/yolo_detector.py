# This code uses YOLOv11 from Ultralytics for object detection
# Citation: Jocher, G., et al. (2023). Ultralytics YOLOv11. https://docs.ultralytics.com/models/yolo11/

from typing import Any, Dict, List
from ultralytics import YOLO
from .color_filter import ColorFilter
from .detection import Detection
import torch
import numpy as np
from constants.constants import OBJECTS
from parsers.yolo_segmenter import YOLOSegmenter

class YOLODetector:
    """ 
    Module to perform object detection using YOLOv11 model from Ultralytics.
    This class is designed to detect objects in a given frame and return the detection results.
    The model's weights are loaded from a specified file path.
    
    Original YOLO architecture by Joseph Redmon et al.
    Model: yolov11n.pt (YOLOv11)
    Citation: Jocher, G., et al. (2023). Ultralytics YOLOv11. https://docs.ultralytics.com/models/yolo11/
    Repository: https://github.com/ultralytics/ultralytics
    """

    def __init__(self, model_path: str = 'yolo11n.pt', 
                 device: str = 'auto',
                 use_segmentation: bool = False):
        self.model = YOLO(model_path)
        self.color_filter = ColorFilter()
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.class_names = self.model.names
        self.OBJECTS = OBJECTS
        if use_segmentation:
            self.segmenter = YOLOSegmenter()
        else:
            self.segmenter = None

    def detect_objects(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results = self.model.predict(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls[0])
                class_name = self.class_names.get(class_idx, "Unknown")
                # Detect only people and vehicles - NOTE: might add more classes later because it is detecting them anyway and just filtering them out
                if class_name in OBJECTS:
                    confidence = float(box.conf[0])
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                    detections.append(Detection(
                        class_name=class_name,
                        class_id=class_idx,
                        confidence=confidence,
                        bbox=(xmin, ymin, xmax, ymax),
                        timestamp=timestamp,
                        dominant_color=self.color_filter.analyze_object_color(frame, (xmin, ymin, xmax, ymax), self.segmenter)
                    ))

        return results[0], detections