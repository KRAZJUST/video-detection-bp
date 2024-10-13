import os
import cv2
import torch
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO
from PIL import Image

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    timestamp: float

class YOLODetector:
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'auto'):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path (str): Path to the YOLO model weights.
            device (str): Device to run the model on ('cpu' or 'cuda'). 'auto' selects automatically.
        """
        self.model = YOLO(model_path)
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        # Dictionary mapping class indices to names
        self.class_names = self.model.names

    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame (np.ndarray): The video frame in BGR format.
        
        Returns:
            List[Detection]: A list of detected objects.
        """
        results = self.model.predict(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls[0])
                class_name = self.class_names.get(class_idx, "Unknown")
                confidence = float(box.conf[0])
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(xmin, ymin, xmax, ymax),
                    # Placeholder - will be set in VideoProcessor
                    timestamp=0.0
                ))
        return detections

class ColorFilter:
    def __init__(self):
        """
        Initialize the color filter with predefined HSV ranges for colors.
        """
        # Defines HSV ranges for colors
        self.color_ranges = {
            'red': [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
            'blue': [((100, 150, 0), (140, 255, 255))],
            'green': [((40, 70, 70), (80, 255, 255))],
            'yellow': [((20, 100, 100), (30, 255, 255))],
            # TODO: add more options
        }

    def is_color_present(self, image: np.ndarray, bbox: Tuple[int, int, int, int], color: str) -> bool:
        """
        Check if the specified color is present within the bounding box of the image.
        
        Args:
            image (np.ndarray): The original frame in BGR format.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (xmin, ymin, xmax, ymax).
            color (str): The color to filter by.
        
        Returns:
            bool: True if the color is present, False otherwise.
        """
        if color not in self.color_ranges:
            raise Exception(f"Color '{color} not supported.'")

        xmin, ymin, xmax, ymax = bbox
        roi = image[ymin:ymax, xmin:xmax]

        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = None
        for lower, upper in self.color_ranges[color]:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            current_mask = cv2.inRange(hsv, lower_np, upper_np)
            if mask is None:
                mask = current_mask
            else:
                mask = cv2.bitwise_or(mask, current_mask)

        # Calculate the percentage of the ROI that matches the color
        color_presence = np.sum(mask) / 255 / mask.size
        # Treshold (set on the precision the user requires TODO: maybe add it as cli argument)
        return color_presence > 0.05

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, query: str, interval: int = 2):
        """
        Initialize the VideoProcessor.
        
        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory to save output frames and logs.
            query (str): Query string in the format 'color object' (e.g., 'red car').
            interval (int): Interval in seconds to extract frames.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.query = query.lower().split()
        if len(self.query) < 2:
            raise ValueError("Query should be in the format 'color object', e.g., 'red car'")
        self.color_query = self.query[0]
        self.object_query = ' '.join(self.query[1:])
        self.interval = interval
        self.detector = YOLODetector()
        self.color_filter = ColorFilter()
        self.log_entries = []

        # If the output directory doesn't exist, create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create directory for frames where the query was detected
        self.frames_output_dir = os.path.join(self.output_dir, "found_frames")
        os.makedirs(self.frames_output_dir, exist_ok=True)

        # Get the text file where the detected objects will be stored with confidence score and timestamps
        self.log_file_path = os.path.join(self.output_dir, "detection_log.txt")

    def process_video(self):
        """
        Process the video: extract frames, detect objects, filter by query, save frames, and log detections.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception(f"ERROR: Unable to open video file '{self.video_path}'.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        print(f"Video FPS: {fps}")
        print(f"Total frames: {frame_count}")
        print(f"Video duration (s): {duration:.2f}")

        current_time = 0.0
        frame_number = 0

        while cap.isOpened() and current_time <= duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            detections = self.detector.detect_objects(frame)
            timestamp = current_time

            # Update timestamp in detections
            for det in detections:
                det.timestamp = timestamp

            # Check for the query in detections
            found = False
            for det in detections:
                if det.class_name == self.object_query:
                    if self.color_filter.is_color_present(frame, det.bbox, self.color_query):
                        found = True
                        break

            if found:
                # Save the frame
                frame_filename = os.path.join(self.frames_output_dir, f"frame_{int(timestamp)}s.png")
                cv2.imwrite(frame_filename, frame)
                print(f"Saved frame at {timestamp:.2f}s: {frame_filename}")

                # Log all detections in this frame
                log_entry = f"Frame {frame_number}, Timestamp: {timestamp:.2f}s\n"
                for det in detections:
                    log_entry += f" - Detected: {det.class_name}, Confidence: {det.confidence:.2f}, Timestamp: {det.timestamp:.2f}s\n"
                log_entry += "\n"
                self.log_entries.append(log_entry)

            current_time += self.interval

        cap.release()

        # Write log file
        with open(self.log_file_path, 'w') as log_file:
            log_file.writelines(self.log_entries)
        print(f"Detection log saved to '{self.log_file_path}'.")

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection with Color Filtering')
    parser.add_argument('--video_path', type=str, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, help='Directory to save output frames and logs')
    parser.add_argument('--query', type=str, help='Search query in the format "color object", e.g., "red car"')
    parser.add_argument('--interval', type=int, default=2, help='Time interval in seconds to extract frames. Default is one frame each two seconds.')

    args = parser.parse_args()

    processor = VideoProcessor(
        video_path=args.video_path,
        output_dir=args.output_dir,
        query=args.query,
        interval=args.interval
    )

    processor.process_video()

if __name__ == '__main__':
    main()
