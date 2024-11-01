"""ByteTrackTracker class for tracking objects using ByteTrack from supervision."""

import os
from typing import Any, Dict, List
import numpy as np
import cv2
import supervision as sv
from .color_filter import ColorFilter


class ByteTrackTracker:
    def __init__(self, output_dir: str):
        # Initialize ByteTrack from supervision
        self.tracker = sv.ByteTrack(lost_track_buffer=30)
        # Initialize the color filter
        self.color_filter = ColorFilter()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.output_dir = output_dir
        self.annotated_images_dir = os.path.join(self.output_dir, "annotated_frames")
        self.initialize_output_folders()


    def initialize_output_folders(self):
        # Create the main output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create a subdirectory for annotated images
        os.makedirs(self.annotated_images_dir, exist_ok=True)

    def annotate_frame(self, frame: np.ndarray, tracked_detections) -> np.ndarray:
        if len(tracked_detections.xyxy) == 0:
            return frame

        # This method creates annotations on the frame, which includes bounding boxes and labels
        labels = [
            f"#{track_id} {class_name}"
            for track_id, class_name in zip(tracked_detections.tracker_id, tracked_detections.data['class_name'])
        ]

        # Annotate bounding boxes and labels
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=tracked_detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=tracked_detections, labels=labels)
        return annotated_frame

    def parse_tracked_detections(self, tracked_detections) -> List[Dict[str, Any]]:
        """Parse the tracked detections into a list of dictionaries so it can be later saved into detection log."""
        detections = []
        for i in range(len(tracked_detections.tracker_id)):
            detection_info = {
                'track_id': int(tracked_detections.tracker_id[i]),
                'bbox': tracked_detections.xyxy[i].tolist(),            # Bounding box as list
                'confidence': float(tracked_detections.confidence[i]),  # Confidence score
                'class_id': int(tracked_detections.class_id[i]),        # Class ID
                'class_name': tracked_detections.data['class_name'][i]  # Class name
            }
            detections.append(detection_info)
        return detections

    def update_tracks(self, results, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        # Convert YOLO results directly to Detections format
        sv_detections = sv.Detections.from_ultralytics(results)
        
        # Update ByteTrack with detections
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        parsed_detections = self.parse_tracked_detections(tracked_detections)

        # Annotate the frame with bounding boxes and labels for better visualization TODO: remove later for performance
        annotated_frame = self.annotate_frame(frame, tracked_detections)

        # Save the annotated frame
        annotated_frame_path = os.path.join(self.annotated_images_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(annotated_frame_path, annotated_frame)


        return parsed_detections