import os
from typing import Any, Dict, List
import numpy as np
import cv2
import torch
from .color_filter import ColorFilter
from .detection import Detection
from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortDetector:
    def __init__(self, output_dir, max_age=30, use_gpu: bool = True):
        """
        Initialize the DeepSortDetector.
        
        Parameters:
            model: YOLO detection model.
            max_age (int): Maximum number of missed frames before a track is deleted.
            n_init (int): Number of consecutive detections before a track is confirmed.
        """
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        self.tracker = DeepSort(
            max_age=max_age,                                            # Frames to retain lost track
            n_init=3,                                                   # Number of frames before confirming a track
            nms_max_overlap=1.0,                                        # Maximum allowed overlap for NMS
            max_cosine_distance=0.2,                                    # Cosine distance for feature matching
            max_iou_distance=0.7,
            nn_budget=100,                                              # Maximum size of the appearance descriptor collection
            embedder='mobilenet',                                       # TODO: maybe change later to osnet_x1_0 but for now leave mobilenet for speed
            half=True if 'cuda' in self.device else False,              # Use half precision for GPU
            embedder_gpu=self.device                                    # Device to run the embedder
        )
        self.color_filter = ColorFilter()
        self.output_dir = output_dir
        self.annotated_images_dir = os.path.join(self.output_dir, "annotated_frames")

    def track(self, detections, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """
        Detect objects in the frame and track them with DeepSORT.
        
        Parameters:
            frame (np.ndarray): The current video frame.
            timestamp (float): The timestamp of the frame.
        
        Returns:
            List[Dict[str, Any]]: Tracked detections with added track IDs.
        """
        # Format detections for DeepSORT
        raw_detections = [
            [(det.bbox[0], det.bbox[1], det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]), det.confidence]
        for det in detections]
        
        # Update tracker
        tracks = self.tracker.update_tracks(raw_detections=raw_detections, frame=frame)
        
        # Assign track IDs back to detections
        tracked_detections = []
        for det, track in zip(detections, tracks):
            det.track_id = track.track_id
            tracked_detections.append(det)

        self.save_annotated_frame(frame=frame, tracked_detections=tracked_detections, frame_number=frame_number)
        return tracked_detections

    def save_annotated_frame(self, frame: np.ndarray, frame_number: int, tracked_detections: List[Dict[str, Any]]):
        """
        Annotates the frame with tracked detections and saves it to the specified path.
        
        Parameters:
            frame (np.ndarray): The current video frame to annotate.
            tracked_detections (List[Dict[str, Any]]): List of tracked detections with 'track_id', 'bbox', 'confidence', etc.
            output_path (str): Path to save the annotated frame.
        """
        for det in tracked_detections:
            # Draw bounding box
            xmin, ymin, xmax, ymax = det.bbox
            color = (255, 0, 0)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Display track ID, confidence, and class name
            label = f"ID: {det.track_id}, {det.class_name}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the annotated frame
        annotated_frame_path = os.path.join(self.annotated_images_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(annotated_frame_path, frame)