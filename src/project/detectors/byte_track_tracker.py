# =============================================================================
# File: byte_track_tracker.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# IMPORTANT: This module uses ByteTrack algorithm implemented in the Supervision library.
#   ByteTrack paper: Zhang, Yifu, et al. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box."
#   ECCV 2022. https://arxiv.org/abs/2110.06864
#   Supervision library: https://github.com/roboflow/supervision
#
# Description:
# This module implements the ByteTrackTracker class, which uses the ByteTrack
# algorithm for multi-object tracking. The class is designed to work with
# YOLO detection results and inherits from the BaseTracker class.
#
# =============================================================================

from typing import Any, Dict, List, Tuple
import numpy as np
import supervision as sv
from .color_filter import ColorFilter
from .base_tracker import BaseTracker


class ByteTrackTracker(BaseTracker):
    """
    ByteTrackTracker class for tracking objects using ByteTrack from supervision.

    ByteTrack paper: Zhang, Yifu, et al. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." 
    ECCV 2022. https://arxiv.org/abs/2110.06864

    Supervision implementation: https://github.com/roboflow/supervision
    """

    def __init__(self, output_dir: str, 
                 min_frames_for_averaging: int = 2, 
                 frame_width: int = 640, 
                 frame_height: int = 360):
        super().__init__(output_dir, min_frames_for_averaging, frame_width, frame_height)
        self.tracker = sv.ByteTrack(track_activation_threshold=0.2)
        self.color_filter = ColorFilter()

    def update_tracks(self, results, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """
        Passes the YOLO results to the ByteTrack tracker and updates the tracks.

        Args:
            results: YOLO detection results
            frame: The current frame being processed
            frame_number: The current frame number
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing detection results
        """
        
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
        #self.save_annotated_frame(frame, detections, frame_number)
        
        return detections