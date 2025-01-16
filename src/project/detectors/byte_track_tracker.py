"""ByteTrackTracker class for tracking objects using ByteTrack from supervision."""

from typing import Any, Dict, List, Tuple
import numpy as np
import supervision as sv
from .color_filter import ColorFilter
from .base_tracker import BaseTracker


class ByteTrackTracker(BaseTracker):
    def __init__(self, output_dir: str, min_frames_for_averaging: int = 2, frame_width: int = 640, frame_height: int = 374):
        super().__init__(output_dir, min_frames_for_averaging, frame_width, frame_height)
        self.tracker = sv.ByteTrack(track_activation_threshold=0.2)
        self.color_filter = ColorFilter()

    def update_tracks(self, results, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        # Convert YOLO results to Detections format
        sv_detections = sv.Detections.from_ultralytics(results)
        
        # Update ByteTrack with detections
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        # Cleanup old tracks that haven't been seen for a while to save memory
        self.cleanup_old_tracks(frame_number)
        
        # Parse detections
        detections = []
        for i in range(len(tracked_detections.tracker_id)):
            track_id = int(tracked_detections.tracker_id[i])
            bbox = tracked_detections.xyxy[i].tolist()
            
            # Calculate direction
            direction = self.update_track_history(track_id, bbox, frame_number)
            # Process colors with lifetime history and averaging
            dominant_color, color_percentages = self.process_detection_colors(
                track_id, frame, bbox, frame_number
            )
            
            # Create detection dictionary
            detection_info = {
                'track_id': track_id,
                'bbox': [int(coord) for coord in bbox],
                'confidence': float(tracked_detections.confidence[i]),
                'class_id': int(tracked_detections.class_id[i]),
                'class_name': tracked_detections.data['class_name'][i],
                'direction': direction,
                'dominant_color': dominant_color,
            }
            detections.append(detection_info)

        # Save annotated frame
        self.save_annotated_frame(frame, detections, frame_number)
        
        return detections