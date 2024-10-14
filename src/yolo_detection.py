import os
import cv2
import torch
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO
from PIL import Image
import subprocess
import json
import glob

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    timestamp: float

class YOLODetector:
    def __init__(self, model_path: str = 'yolo11n.pt', device: str = 'auto'):
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
                if class_name in ['person', 'car']:
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
        NOTE: Not used in the current implementation on the initial indexing.

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
    def __init__(self, video_path: str, output_dir: str, query: str, interval: int = 30):
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
        
        # Create directory for frames
        self.frames_output_dir = os.path.join(self.output_dir, "found_frames")
        os.makedirs(self.frames_output_dir, exist_ok=True)

        # Get the text file where the detected objects will be stored with confidence score and timestamps
        self.log_file_path = os.path.join(self.output_dir, "detection_log.txt")

    def get_video_info(self, video_path):
        """ Function to extract video information using FFmpeg. """

        # ffprobe command to extract video information in JSON format
        ffprobe_command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames,duration', '-of', 'json', video_path
        ]
        
        # Execute the ffprobe command
        result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Parse the result as JSON
        video_info = json.loads(result.stdout)
        
        # Extract required information
        if 'streams' in video_info and len(video_info['streams']) > 0:
            stream_info = video_info['streams'][0]
            width = stream_info.get('width', 'Unknown')
            height = stream_info.get('height', 'Unknown')
            duration = stream_info.get('duration', 'Unknown')
            nb_frames = stream_info.get('nb_frames', 'Unknown')
            avg_frame_rate = stream_info.get('avg_frame_rate', 'Unknown')
            
            # Parse frame rate (if it's available as a fraction)
            fps = eval(avg_frame_rate) if '/' in avg_frame_rate else avg_frame_rate
            
            return {
                'width': width,
                'height': height,
                'duration': float(duration),
                'nb_frames': int(nb_frames),
                'fps': fps
            }
        
        return None
    
    def print_video_info(self, video_info):
        """ Helper function to print video information. """

        print(f"Video Info:")
        print(f" - Resolution: {video_info['width']}x{video_info['height']}")
        print(f" - Duration: {video_info['duration']} seconds")
        print(f" - Frame Rate: {video_info['fps']} fps")
        print(f" - Total Frames: {video_info['nb_frames']}")
        print(f" - Frames to Process: {video_info['nb_frames'] / self.interval}")

    def process_video(self):
        """
        Process the video: extract frames with FFmpeg, detect objects, filter by query, save frames, and log detections.
        """

        # Get video information
        video_info = self.get_video_info(self.video_path)
        if video_info is None:
            print("Failed to get video information.")
        else: 
            self.print_video_info(video_info)

        # Defines FFmpeg command to extract frames at regular intervals (self.interval in seconds)
        ffmpeg_command = [
            'ffmpeg', '-i', self.video_path,                               # path to input video
            '-vf', f'select=not(mod(n\\,{self.interval}))',                # Selects every nth frame (interval)
            '-vsync', 'vfr',
            f'{self.frames_output_dir}/frame_%04d.jpg'                     # Output frame path
        ]
        
        # Start the FFmpeg process
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for the process to finish
        ffmpeg_process.wait()

         # Read the saved frames for processing
        frame_files = sorted(glob.glob(os.path.join(self.frames_output_dir, 'frame_*.jpg')))

        frame_number = 0

        for frame_number, frame_file in enumerate(frame_files):
            print(f"Processing frame {frame_file}")
            
            # Read the saved frame
            frame = cv2.imread(frame_file)

            # Run YOLO detection on the frame
            detections = self.detector.detect_objects(frame)
            # Calculate timestamp by multiplying frame number by interval
            timestamp = frame_number * (1 / video_info['fps']) * self.interval

            # Update timestamp in detections
            for det in detections:
                det.timestamp = timestamp

            # Log detections
            for det in detections:
                log_entry = {
                    "frame_number": frame_number,
                    "frame_file": frame_file,
                    "timestamp": timestamp,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox": det.bbox
                }
                self.log_entries.append(log_entry)

        # Clean up the FFmpeg process
        ffmpeg_process.stdout.close()
        ffmpeg_process.stderr.close()
        ffmpeg_process.wait()

        # Write log file in JSON format for better readability and later processing
        with open(self.log_file_path, 'w') as log_file:
            json.dump(self.log_entries, log_file, indent=4)
        print(f"Metadata log saved to '{self.log_file_path}'.")

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection with Color Filtering')
    parser.add_argument('--video_path', type=str, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, help='Directory to save output frames and logs')
    parser.add_argument('--query', type=str, help='Search query in the format "color object", e.g., "red car"')
    parser.add_argument('--interval', type=int, default=30, help='Time interval in seconds to extract frames. Default is one frame each two seconds.')

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
