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
import shutil
import time

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
                if class_name in ['person', 'car', 'truck', 'bus', 'motorcycle']:
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
            'white': [((0, 0, 200), (180, 25, 255))],
            'orange': [((10, 100, 100), (20, 255, 255))],
            'purple': [((140, 50, 50), (160, 255, 255))],
            # TODO: add more options
        }

    def detect_dominant_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Detect the most prominent color within the bounding box of the image.

        Args:
            image (np.ndarray): The original frame in BGR format.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (xmin, ymin, xmax, ymax).

        Returns:
            str: The name of the most prominent color, or 'none' if no color is prominent.
        """
        xmin, ymin, xmax, ymax = bbox
        roi = image[ymin:ymax, xmin:xmax]

        if roi.size == 0:
            return 'none'

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        color_presence = {}
        # Loops through each color and calculate the percentage of that color in the ROI
        for color, ranges in self.color_ranges.items():
            mask = None
            for lower, upper in ranges:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                current_mask = cv2.inRange(hsv, lower_np, upper_np)
                if mask is None:
                    mask = current_mask
                else:
                    mask = cv2.bitwise_or(mask, current_mask)

            # Calculates the percentage of the ROI that matches the color
            percentage = np.sum(mask) / 255 / mask.size
            color_presence[color] = percentage

        # Finds the color with the highest percentage presence
        dominant_color = max(color_presence, key=color_presence.get)

        # Returns the dominant color if it's presence is above threshold, otherwise 'none' TODO: add threshold as parameter and CLI argument
        if color_presence[dominant_color] > 0.05:
            return dominant_color
        return 'none'

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
        
        # Creates directory for frames
        self.frames_output_dir = os.path.join(self.output_dir, "extracted_frames")
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
        
        # Executes the ffprobe command
        result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Parse the result as JSON
        video_info = json.loads(result.stdout)
        
        # Extracts required information
        if 'streams' in video_info and len(video_info['streams']) > 0:
            stream_info = video_info['streams'][0]
            width = stream_info.get('width', 'Unknown')
            height = stream_info.get('height', 'Unknown')
            duration = stream_info.get('duration', 'Unknown')
            nb_frames = stream_info.get('nb_frames', 'Unknown')
            avg_frame_rate = stream_info.get('avg_frame_rate', 'Unknown')
            
            # Parses frame rate (if it's available as a fraction)
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
        Process the video: extract frames with FFmpeg, detect objects, and log them with necessary informations and dominant color.
        """

        # Gets video information
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
            '-start_number', '0',                                          # Starts frame numbering from 0
            f'{self.frames_output_dir}/frame_%04d.jpg'                     # Output frame path
        ]
        
        # Starts the FFmpeg process
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for the process to finish
        ffmpeg_process.wait()

        # Reads the saved frames for processing
        frame_files = sorted(glob.glob(os.path.join(self.frames_output_dir, 'frame_*.jpg')))

        # Initialize a dictionary to store log entries
        self.log_entries = {}

        for frame_number, frame_file in enumerate(frame_files):
            print(f"Processing frame {frame_file}")
            
            # Reads the saved frame
            frame = cv2.imread(frame_file)

            # Runs YOLO detection on the frame
            detections = self.detector.detect_objects(frame)
            # Calculates timestamp by multiplying frame number by interval
            timestamp = frame_number * (1 / video_info['fps']) * self.interval

             # Initialize a list to hold detections for this frame
            frame_detections = []

            # Logs detections
            for det in detections:
                # Gets the dominant color within the bounding box
                dominant_color = self.color_filter.detect_dominant_color(frame, det.bbox)

                # Create a log entry for the detection
                detection_entry = {
                    "frame_number": frame_number,
                    "frame_file": frame_file,
                    "timestamp": timestamp,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "bbox": det.bbox,
                    "dominant_color": dominant_color
                }
                frame_detections.append(detection_entry)

            # Store all detections for this frame under its frame number
            if frame_detections:
                self.log_entries[frame_number] = frame_detections

        # Cleans up the FFmpeg process
        ffmpeg_process.stdout.close()
        ffmpeg_process.stderr.close()
        ffmpeg_process.wait()

        # Save the log entries to a text file
        self.save_log()

    def save_log(self):
        """
        Save the log entries to a text file.
        """
        with open(self.log_file_path, 'w') as log_file:
            json.dump(self.log_entries, log_file, indent=4)
        print(f"Log saved to '{self.log_file_path}'.")


class DetectionParser:
    def __init__(self, log_entries, query, output_dir):
        self.log_entries = log_entries
        self.queries = self.parse_complex_query(query)
        self.found_log_entries = {}
        self.output_dir = output_dir
        # Create a directory to save the found frames
        self.found_dir = os.path.join(output_dir, 'found_frames')
        os.makedirs(self.found_dir, exist_ok=True)

    def parse_complex_query(self, query):
        """
        Parse multiple color-object pairs with support for AND/OR conditions.
        Example input: 'red car and blue truck or yellow bus'
        Output: [{'logic': 'AND', 'conditions': [('red', 'car'), ('blue', 'truck')]}, {'logic': 'OR', 'conditions': [('yellow', 'bus')]}]
        """
        # Split by 'or' first, then within each group split by 'and'
        or_groups = [group.strip() for group in query.lower().split('or')]
        parsed_queries = []
        
        for group in or_groups:
            and_conditions = [condition.strip() for condition in group.split('and')]
            conditions = []
            for condition in and_conditions:
                parts = condition.split(' ')
                if len(parts) == 2:
                    color, obj = parts
                    conditions.append((color, obj))
                else:
                    raise ValueError("Query must be in the format 'color object', for example 'red car'")
            parsed_queries.append({'logic': 'AND', 'conditions': conditions})

        print(parsed_queries)
        return parsed_queries
    
    def filter_log(self):
        """Filter the detection log based on the query."""

        # Iterate through the log entries organized by frame number
        for frame_number, detections in self.log_entries.items():
            for entry in detections:
                if self.matches_complex_query(entry):
                    if frame_number not in self.found_log_entries:
                        self.found_log_entries[frame_number] = []
                    self.found_log_entries[frame_number].append(entry)
                    
                    # Copy the image to the found folder
                    frame_file = entry['frame_file']
                    shutil.copy(frame_file, self.found_dir)

        self.save_found_log()
    
    def matches_complex_query(self, entry):
        """
        Check if the log entry matches any of the complex queries.
        A query can contain multiple conditions connected by 'AND' or 'OR'.
        """
        vehicle_query = ['car', 'truck', 'bus', 'motorcycle']

        # Go through each 'OR' group
        for query_group in self.queries:
            if query_group['logic'] == 'AND':
                # All conditions in this group must be satisfied
                if all(self.matches_condition(entry, condition, vehicle_query) for condition in query_group['conditions']):
                    return True
            else:  # It's an OR logic
                # At least one condition in this group must be satisfied
                if any(self.matches_condition(entry, condition, vehicle_query) for condition in query_group['conditions']):
                    return True
                
        return False

    def matches_condition(self, entry, condition, vehicle_query):
        """Check if a single condition (color-object pair) matches the log entry."""
        query_color, query_object = condition
        
        # Object matching logic
        if query_object == 'vehicle':
            object_matches = entry['class_name'].lower() in vehicle_query
        else:
            object_matches = entry['class_name'].lower() == query_object
        
        # Color matching logic
        color_matches = entry['dominant_color'].lower() == query_color
        
        confidence = entry['confidence']

        return object_matches and color_matches and confidence > 0.4

    def save_found_log(self):
        """Save the filtered log entries to a new JSON file."""
        found_log_path = os.path.join(self.output_dir, 'found_log.txt')
        with open(found_log_path, 'w') as found_log_file:
            json.dump(self.found_log_entries, found_log_file, indent=4)
        print(f"Found log saved to '{found_log_path}'.")


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='YOLO Object Detection with Color Filtering')
    parser.add_argument('--video_path', type=str, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, help='Directory to save output frames and logs')
    parser.add_argument('--query', type=str, help='Search query in the format "color object", e.g., "red car"')
    parser.add_argument('--interval', type=int, default=30, help='Time interval in which to extract frames (default: every 30th frame)')

    args = parser.parse_args()
    argument_parsing_time = time.time()

    # Run detection on the video
    processor = VideoProcessor(
        video_path=args.video_path,
        output_dir=args.output_dir,
        query=args.query,
        interval=args.interval
    )
    processor.process_video()

    # Parse the log file to filter the results based on the query
    log_parser = DetectionParser(
        log_entries=processor.log_entries,
        query=args.query,
        output_dir=args.output_dir
    )
    log_parser.filter_log()

    print(f"Total time taken: {time.time() - start_time:.2f} seconds (Argument Parsing: {argument_parsing_time - start_time:.2f} seconds)")

if __name__ == '__main__':
    main()
