"""ByteTrackTracker class for tracking objects using ByteTrack from supervision."""

import os
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2
import supervision as sv
from .color_filter import ColorFilter
from math import atan2, degrees


class ByteTrackTracker:
    def __init__(self, output_dir: str):
        # Initialize ByteTrack from supervision
        self.tracker = sv.ByteTrack(track_activation_threshold=0.2)
        # Initialize the color filter
        self.color_filter = ColorFilter()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.output_dir = output_dir
        self.annotated_images_dir = os.path.join(self.output_dir, "annotated_frames")
        self.initialize_output_folders()

        # Track history for calculating direction
        self.track_history = {}

    def initialize_output_folders(self):
        """
        Initialize the output directories for saving annotated images.
        """
        
        # Create the main output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create a subdirectory for annotated images
        os.makedirs(self.annotated_images_dir, exist_ok=True)


    def annotate_frame(self, frame: np.ndarray, tracked_detections) -> np.ndarray:
        """
        Helper method to annotate the frame with bounding boxes and labels.

        Parameters:
            frame (np.ndarray): The current video frame.
            tracked_detections: The tracked detections from ByteTrack.

        Returns:
            np.ndarray: The annotated frame with bounding boxes and labels.
        """

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


    def get_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calculate the centroid of a bounding box.
        
        Parameters:
            bbox (List[float]): The bounding box in [xmin, ymin, xmax, ymax] format.

        Returns:
            Tuple[float, float]: The centroid coordinates (x, y).
        """

        xmin, ymin, xmax, ymax = bbox
        return ((xmin + xmax) / 2, (ymin + ymax) / 2)


    def angle_to_direction(self, angle: float) -> str:
        """
        Convert an angle in degrees to a compass direction within basic 8 directions.

        Parameters:
            angle (float): The angle in degrees, where 0 degrees is "north".

        Returns:
            str: The compass direction corresponding to the angle.
        """

        # Normalize angle to be within 0-360 degrees
        angle = (angle + 360) % 360
        print(f'angle: {angle}')

        # Map angle to 8-point compass directions
        if 337.5 <= angle or angle < 22.5:
            return "E"
        elif 22.5 <= angle < 67.5:
            return "NE"
        elif 67.5 <= angle < 112.5:
            return "N"
        elif 112.5 <= angle < 157.5:
            return "NW"
        elif 157.5 <= angle < 202.5:
            return "W"
        elif 202.5 <= angle < 247.5:
            return "SW"
        elif 247.5 <= angle < 292.5:
            return "S"
        elif 292.5 <= angle < 337.5:
            return "SE"

    
    def calculate_direction(self, start_point: Tuple[float, float], end_point: Tuple[float, float]) -> str:
        """
        Calculate direction based on movement vector between points.

        Parameters:
            start_point (Tuple[float, float]): The starting point (x, y).
            end_point (Tuple[float, float]): The ending point (x, y).

        Returns:
            str: The compass direction of the movement.
        """

        dx = end_point[0] - start_point[0]
        # Invert dy to match the coordinate system
        dy = start_point[1] - end_point[1]

        angle = degrees(atan2(dy, dx))

        # Map angle to compass direction
        return self.angle_to_direction(angle)


    def parse_tracked_detections(self, tracked_detections, frame_number: int) -> List[Dict[str, Any]]:
        """
        Parse tracked detections from ByteTrack into a list of dictionaries.

        Parameters:
            tracked_detections: The tracked detections from ByteTrack.
            frame_number (int): The frame number.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing detection information.
        """

        detections = []
        for i in range(len(tracked_detections.tracker_id)):
            track_id = int(tracked_detections.tracker_id[i])
            bbox = tracked_detections.xyxy[i].tolist()
            class_name = tracked_detections.data['class_name'][i]

            # Compute centroid and store in track history
            centroid = self.get_centroid(bbox)
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((frame_number, centroid))

            # Calculate direction if there is enough history
            direction = None
            if len(self.track_history[track_id]) >= 2:
                start_frame, start_point = self.track_history[track_id][-2]
                end_frame, end_point = self.track_history[track_id][-1]
                direction = self.calculate_direction(start_point, end_point)

            # Append detection info to the list
            detection_info = {
                'track_id': track_id,
                'bbox': bbox,
                'confidence': float(tracked_detections.confidence[i]),
                'class_id': int(tracked_detections.class_id[i]),
                'class_name': class_name,
                'direction': direction
            }
            detections.append(detection_info)
        return detections

    def update_tracks(self, results, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """
        Updates the initial YOLO detections with ByteTrack tracking.

        Parameters:
            results: The YOLO detection results.
            frame (np.ndarray): The current video frame.
            frame_number (int): The frame number.

        Returns:
            List[Dict[str, Any]]: The tracked detections with added track IDs.
        """

        # Convert YOLO results directly to Detections format
        sv_detections = sv.Detections.from_ultralytics(results)
        
        # Update ByteTrack with detections
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        parsed_detections = self.parse_tracked_detections(tracked_detections, frame_number)

        # Annotate the frame with bounding boxes and labels for better visualization TODO: remove later for performance
        annotated_frame = self.annotate_frame(frame, tracked_detections)

        # Save the annotated frame
        annotated_frame_path = os.path.join(self.annotated_images_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(annotated_frame_path, annotated_frame)

        return parsed_detections