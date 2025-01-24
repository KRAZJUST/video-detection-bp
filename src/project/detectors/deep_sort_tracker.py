from typing import Any, Dict, List
import numpy as np
import torch
from .color_filter import ColorFilter
from deep_sort_realtime.deepsort_tracker import DeepSort
from .base_tracker import BaseTracker

class DeepSortTracker(BaseTracker):
    def __init__(self, output_dir: str, max_age: int = 5, use_gpu: bool = True):
        super().__init__(output_dir)
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            nms_max_overlap=0.8,
            max_cosine_distance=0.2,
            max_iou_distance=0.6,
            nn_budget=100,
            embedder='mobilenet',
            half=True if 'cuda' in self.device else False,
            embedder_gpu=self.device
        )
        self.color_filter = ColorFilter()

    def update_tracks(self, detections, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        # Format detections for DeepSORT
        raw_detections = [
            [(det.bbox[0], det.bbox[1], det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]), det.confidence]
            for det in detections
        ]
        
        # Update tracker
        tracks = self.tracker.update_tracks(raw_detections=raw_detections, frame=frame)
        
        # Process tracked detections
        tracked_detections = []
        for det, track in zip(detections, tracks):
            bbox = [det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]]
            
            # Calculate direction
            direction = self.update_track_history(track.track_id, bbox, frame_number)
            
            # Create detection dictionary
            detection_info = {
                'track_id': track.track_id,
                'bbox': [int(coord) for coord in bbox],
                'confidence': det.confidence,
                'class_id': det.class_id,
                'class_name': det.class_name,
                'direction': direction,
                'dominant_color': self.color_filter.detect_dominant_color(frame, bbox)
            }
            tracked_detections.append(detection_info)

        # Save annotated frame
        self.save_annotated_frame(frame, tracked_detections, frame_number)
        
        return tracked_detections