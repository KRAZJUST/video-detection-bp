from abc import ABC, abstractmethod
import os
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from math import atan2, degrees
from collections import defaultdict
from .color_filter import ColorFilter
from constants.constants import COLOR_MAP

class BaseTracker(ABC):
    def __init__(self, output_dir: str, min_frames_for_averaging: int = 2, frame_width: int = 640, frame_height: int = 360):
        self.output_dir = output_dir
        self.annotated_images_dir = os.path.join(self.output_dir, "annotated_frames")
        self.track_history = {}
        # Store recent directions
        self.direction_history = defaultdict(list)
        # Store where tracks leave the frame
        self.exit_points = {}
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.initialize_output_folders()
        
        # Track ID -> list of color dictionaries
        self.color_history = defaultdict(list)
        # Last frame where the track was seen
        self.last_seen_frame = {}
        # Minimum number of frames before averaging colors to reduce noise caused by false positives and ID switches
        self.min_frames_for_averaging = min_frames_for_averaging
        self.color_filter = ColorFilter()
        
    def initialize_output_folders(self):
        """Initialize output directories for saving annotated images."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        os.makedirs(self.annotated_images_dir, exist_ok=True)
        
    def process_detection_colors(self, track_id: int, frame: np.ndarray, bbox: List[float], frame_number: int) -> Tuple[str, dict]:
        """
        Process color detection for a tracked object using ColorFilter.
        
        Args:
            track_id: The ID of the tracked object
            frame: Current video frame (BGR format)
            bbox: Bounding box coordinates [xmin, ymin, xmax, ymax]
            frame_number: Current frame number
            
        Returns:
            Tuple[str, dict]: Dominant color and complete color percentages
        """
        try:
            # Extract region of interest with boundary checking
            height, width = frame.shape[:2]
            xmin = max(0, int(bbox[0]))
            ymin = max(0, int(bbox[1]))
            xmax = min(width, int(bbox[2]))
            ymax = min(height, int(bbox[3]))
            
            if xmax <= xmin or ymax <= ymin:
                return 'gray', {'gray': 1.0}
                
            roi = frame[ymin:ymax, xmin:xmax]
            
            if roi.size == 0:
                return 'gray', {'gray': 1.0}
                
            # Create appropriate mask
            aspect_ratio = (xmax - xmin) / (ymax - ymin)
            mask = (self.color_filter.create_elliptical_mask(xmax - xmin, ymax - ymin)
                if aspect_ratio <= 0.8 or aspect_ratio >= 1.2
                else self.color_filter.create_circular_mask(xmax - xmin, ymax - ymin))
            
            # Process colors
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            masked_hsv = cv2.bitwise_and(hsv_roi, hsv_roi, mask=mask)
            current_colors = self.color_filter.calculate_color_presence(masked_hsv, mask)
            
            # Update history and get averaged colors
            avg_colors = self.update_color_history(track_id, current_colors, frame_number)
            dominant_color = self.get_dominant_color(avg_colors)
            
            return dominant_color, avg_colors
        
        except Exception as e:
            print(f"Error processing colors for track {track_id}: {str(e)}")
            return 'gray', {'gray': 1.0}  # Safe default
        
    def update_color_history(self, track_id: int, color_data: dict, frame_number: int) -> dict:
        """
        Update color history with exponential moving average.
        
        Args:
            track_id: The ID of the tracked object
            color_data: Dictionary of color percentages from current frame
            frame_number: Current frame number
            
        Returns:
            dict: Averaged color percentages
        """
        # Check if track was missing for too long
        if track_id in self.last_seen_frame:
            frames_missing = frame_number - self.last_seen_frame[track_id]
            if frames_missing > 10:  # Reset if missing for too long
                self.color_history[track_id] = []
                
        # Update last seen frame
        self.last_seen_frame[track_id] = frame_number
        
        # Add new color data to history
        self.color_history[track_id].append(color_data)
        
        # Keep only last N frames to prevent memory growth
        max_history = 30
        if len(self.color_history[track_id]) > max_history:
            self.color_history[track_id] = self.color_history[track_id][-max_history:]
            
        # Calculate exponential moving average
        alpha = 0.3  # Smoothing factor
        avg_colors = {}
        
        if len(self.color_history[track_id]) >= self.min_frames_for_averaging:
            # Initialize with first frame
            for color in self.color_filter.color_ranges.keys():
                avg_colors[color] = self.color_history[track_id][0].get(color, 0.0)
                
            # Apply exponential moving average
            for frame_data in self.color_history[track_id][1:]:
                for color in self.color_filter.color_ranges.keys():
                    current_value = frame_data.get(color, 0.0)
                    avg_colors[color] = (alpha * current_value + 
                                       (1 - alpha) * avg_colors[color])
        else:
            # If not enough frames, use current frame's colors
            avg_colors = color_data.copy()
            
        return avg_colors
        
    def get_dominant_color(self, avg_colors: dict, threshold: float = 0.02) -> str:
        """
        Get dominant color from averaged percentages with noise filtering.
        
        Args:
            avg_colors: Dictionary of averaged color percentages
            threshold: Minimum threshold for color presence
            
        Returns:
            str: Name of the dominant color
        """
        # Filter out colors below threshold
        filtered_colors = {color: presence for color, presence 
                         in avg_colors.items() if presence >= threshold}
        
        if not filtered_colors:
            # If no colors detected, return the color with the highest presence
            if avg_colors:
                return max(avg_colors.items(), key=lambda x: x[1])[0]

        # Return the color with the highest presence
        dominant_color = max(filtered_colors, key=filtered_colors.get)
        return dominant_color

    def cleanup_old_tracks(self, current_frame: int, max_frames_missing: int = 10):
        """
        Clean up color history for tracks that haven't been seen for a while.
        
        Args:
            current_frame: Current frame number
            max_frames_missing: Maximum number of frames a track can be missing before cleanup
        """
        tracks_to_remove = []
        for track_id, last_frame in self.last_seen_frame.items():
            if current_frame - last_frame > max_frames_missing:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.color_history[track_id]
            del self.last_seen_frame[track_id]

    def get_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate the centroid of a bounding box."""
        xmin, ymin, xmax, ymax = bbox
        return ((xmin + xmax) / 2, (ymin + ymax) / 2)

    def angle_to_direction(self, angle: float, confidence: float = 1.0) -> Tuple[str, float]:
        """
        Convert angle to direction with confidence score.
        Higher confidence means we're more certain about the direction.
        """
        angle = (angle + 360) % 360
        
        # Define direction ranges with centers
        directions = [
            ("east", 0, 337.5, 22.5),
            ("north-east", 45, 22.5, 67.5),
            ("north", 90, 67.5, 112.5),
            ("north-west", 135, 112.5, 157.5),
            ("west", 180, 157.5, 202.5),
            ("south-west", 225, 202.5, 247.5),
            ("south", 270, 247.5, 292.5),
            ("south-east", 315, 292.5, 337.5)
        ]
        
        # Find the matching direction
        for name, center, start, end in directions:
            if name == "east":
                # Special handling for east beucase it wraps around 360
                if angle >= start or angle < end:
                    # Calculate angle difference considering wrap-around
                    if angle >= start:
                        angle_diff = min(abs(angle - 360), abs(angle - center))
                    else:
                        angle_diff = abs(angle - center)
                    direction_confidence = 1.0 - (angle_diff / 22.5)  # 22.5 is max possible difference
                    return name, direction_confidence * confidence
            else:
                # Normal range check for other directions
                if start <= angle < end:
                    angle_diff = abs(angle - center)
                    direction_confidence = 1.0 - (angle_diff / 22.5)
                    return name, direction_confidence * confidence

        return "unknown", 0.0

    def is_leaving_frame(self, centroid: Tuple[float, float], bbox: List[float]) -> bool:
        """
        Check if object is leaving the frame.
        TODO: Implement more robust logic for edge cases and take into account the whole bounding box not just the centroid.
        """
        x, y = centroid
        xmin, ymin, xmax, ymax = bbox
        # margin to consider object leaving the frame (in pixels)
        margin = 20
        
        return (x <= margin or x >= self.frame_width - margin or 
                y <= margin or y >= self.frame_height - margin)

    def get_exit_direction(self, centroid: Tuple[float, float]) -> Tuple[str, float]:
        """Get direction based on where object exits frame."""
        x, y = centroid
        # 10% of frame dimension
        margin = 0.1
        
        # Calculate margin distances in pixels
        margin_x = self.frame_width * margin
        margin_y = self.frame_height * margin
        
        # Check if within margin of edges
        in_left_margin = x <= margin_x
        in_right_margin = x >= self.frame_width - margin_x
        in_top_margin = y <= margin_y
        in_bottom_margin = y >= self.frame_height - margin_y
        
        # Calculate confidence based on how close to edge
        # (closer to edge = higher confidence)
        if in_left_margin:
            confidence = 1.0 - (x / margin_x)
            return "west", confidence
        elif in_right_margin:
            confidence = 1.0 - ((self.frame_width - x) / margin_x)
            return "east", confidence
        elif in_top_margin:
            confidence = 1.0 - (y / margin_y)
            return "north", confidence
        elif in_bottom_margin:
            confidence = 1.0 - ((self.frame_height - y) / margin_y)
            return "south", confidence
        
        # If not in any margin zone
        return "unknown", 0.0

    def calculate_direction(self, start_point: Tuple[float, float], 
                          end_point: Tuple[float, float]) -> Tuple[str, float]:
        """Calculate movement direction between points with confidence."""
        dx = end_point[0] - start_point[0]
        dy = start_point[1] - end_point[1]
        
        # Calculate magnitude of movement
        magnitude = (dx**2 + dy**2)**0.5
        # minimum pixels to consider movement significant
        min_movement = 5
        
        if magnitude < min_movement:
            return "stationary", 0.0
            
        angle = degrees(atan2(dy, dx))
        
        # Higher confidence for larger movements - scale confidence with movement
        confidence = min(1.0, magnitude / 50)
        return self.angle_to_direction(angle, confidence)

    def update_track_history(self, track_id: int, bbox: List[float], frame_number: int) -> str:
        """
            Update tracking history and calculate direction 
            based on recent movement and exit points.
        """
        centroid = self.get_centroid(bbox)
        
        if track_id not in self.track_history:
            self.track_history[track_id] = []
            
        self.track_history[track_id].append((frame_number, centroid))
        
        # Keep only last 30 frames of history
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id] = self.track_history[track_id][-30:]
            
        final_direction = None
        if len(self.track_history[track_id]) >= 2:
            # Get immediate direction
            start_frame, start_point = self.track_history[track_id][-2]
            end_frame, end_point = self.track_history[track_id][-1]
            immediate_direction, confidence = self.calculate_direction(start_point, end_point)
            
            # Check if object is leaving frame
            if self.is_leaving_frame(end_point, bbox):
                exit_direction, exit_conf = self.get_exit_direction(end_point)
                self.exit_points[track_id] = (end_point, exit_direction)
                
                # Blend immediate direction with exit direction based on confidence
                if confidence < exit_conf:
                    final_direction = exit_direction
                else:
                    final_direction = immediate_direction
            else:
                # Use historical directions for smoothing
                self.direction_history[track_id].append(immediate_direction)
                if len(self.direction_history[track_id]) > 5:
                    self.direction_history[track_id] = self.direction_history[track_id][-5:]
                
                # Get most common recent direction
                if confidence < 0.5:  # Low confidence in immediate direction
                    from collections import Counter
                    direction_counts = Counter(self.direction_history[track_id])
                    if direction_counts:
                        final_direction = direction_counts.most_common(1)[0][0]
                    else:
                        final_direction = immediate_direction
                else:
                    final_direction = immediate_direction
                    
        return final_direction

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Helper function to annotate the frame with bounding boxes, labels, and colors.

        Args:
            frame (np.ndarray): The input image in BGR format.
            detections (List[Dict[str, Any]]): List of detection dictionaries containing 'bbox', 'class_name', etc.
        
        Returns:
            np.ndarray: Annotated image with bounding boxes and labels.
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            xmin, ymin, xmax, ymax = det['bbox']
            color = COLOR_MAP[det['dominant_color']] if det.get('dominant_color') in COLOR_MAP else (0, 0, 0) 
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            
            # Create label with track ID, class name, and direction if available
            label_parts = [f"#{det['track_id']}", det['class_name']]
            if det.get('direction'):
                label_parts.append(f"→{det['direction']}")
            if det.get('dominant_color'):
                label_parts.append(f"({det['dominant_color']})")
                
            label = " ".join(label_parts)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, 
                         (int(xmin), int(ymin) - 20),
                         (int(xmin) + label_size[0], int(ymin)),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label,
                       (int(xmin), int(ymin) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        return annotated_frame

    def save_annotated_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]], frame_number: int):
        """
        Helper function to save the annotated frame to disk.
        
        Args:
            frame (np.ndarray): The input image in BGR format.
            detections (List[Dict[str, Any]]): List of detection dictionaries containing 'bbox', 'class_name', etc.
            frame_number (int): The current frame number.

        Returns:
            None
        """
        annotated_frame = self.annotate_frame(frame, detections)
        output_path = os.path.join(self.annotated_images_dir, f"frame_{frame_number:05d}.jpg")
        cv2.imwrite(output_path, annotated_frame)

    @abstractmethod
    def update_tracks(self, detections, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """Abstract method different for the ByteTrack and deepSORT."""
        pass