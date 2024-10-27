import os
import cv2
import torch
import argparse
import numpy as np
import subprocess
import json
import glob
import shutil
import time
from typing import Dict, Any, List, Tuple

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

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

class Detection:
    """ Data class to store detection information. """

    def __init__(self, class_name: str, confidence: float, bbox: tuple, timestamp: float, track_id: int = None, dominant_color: str = None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = timestamp
        self.track_id = track_id
        self.dominant_color = dominant_color

class YOLODetector:
    """ Class to perform object detection using YOLOv11. """

    def __init__(self, model_path: str = 'yolo11n.pt', device: str = 'auto'):
        self.model = YOLO(model_path)
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.class_names = self.model.names

    def detect_objects(self, frame: np.ndarray, timestamp: float) -> List[Detection]:
        # Run YOLO prediction on the frame
        results = self.model.predict(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls[0])
                class_name = self.class_names.get(class_idx, "Unknown")
                # Detect only people and vehicles
                if class_name in ['person', 'car', 'truck']:
                    confidence = float(box.conf[0])
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(xmin, ymin, xmax, ymax),
                        timestamp=timestamp
                    ))
        return detections


class DeepSortTracker:
    # TODO FIX THIS CLASS
    """ Class to perform object tracking using DeepSort. """

    def __init__(self, max_age: int = 3, nn_budget: int = 100, use_gpu: bool = True):

        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        self.color_filter = ColorFilter()
        # Initialize DeepSort tracker
        self.tracker = DeepSort(
            max_age=max_age,                                            # Frames to retain lost track
            n_init=3,                                                   # Number of frames before confirming a track
            nms_max_overlap=1.0,                                        # Maximum allowed overlap for NMS
            max_cosine_distance=0.3,                                    # Cosine distance for feature matching
            nn_budget=nn_budget,                                        # Maximum size of the appearance descriptor collection
            embedder='mobilenet',                                       # TODO: maybe change later to osnet_x1_0 but for now leave mobilenet for speed
            half=True if 'cuda' in self.device else False,              # Use half precision for GPU
            embedder_gpu=self.device                                    # Device to run the embedder
        )

    def update_tracks(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
        """ Updates tracking with Deep SORT, assigning track IDs to detections. """
        
        # Prepare bounding boxes and confidence scores as required by Deep SORT
        raw_detections = [
            [(det.bbox[0], det.bbox[1], det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]), det.confidence]
            for det in detections
        ]

        # Update tracker with formatted detections and the frame
        tracks = self.tracker.update_tracks(raw_detections=raw_detections, frame=frame)
        
        # Assign track IDs back to detections
        tracked_detections = []
        for det, track in zip(detections, tracks):
            det.track_id = track.track_id
            det.dominant_color = self.color_filter.detect_dominant_color(frame, det.bbox)
            tracked_detections.append(det)
        
        return tracked_detections


class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, interval: int = 30):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_output_dir = os.path.join(self.output_dir, "extracted_frames")
        os.makedirs(self.frames_output_dir, exist_ok=True)
        self.detector = YOLODetector()
        self.tracker = DeepSortTracker()
        self.log_entries = {}
        self.video_info = self.get_video_info()
        self.interval = interval

    def get_video_info(self) -> Dict[str, Any]:
        """ Function to extract video information using FFmpeg. """

        # ffprobe command to extract video information in JSON format
        ffprobe_command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames,duration', '-of', 'json', self.video_path
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

    def extract_frames(self):
        """ Function to extract frames from a video using FFmpeg. """

        ffmpeg_command = [
            'ffmpeg', '-fflags', '+genpts', '-i', self.video_path,
            '-vf', f"select='not(mod(n\\,{self.interval}))'",
            '-fps_mode', 'vfr',
            # Sets pixel format to yuvj420p for better accuracy in object detection
            '-pix_fmt', 'yuvj420p',
            '-start_number', '0',
            # Output frame path
            f'{self.frames_output_dir}/frame_%04d.jpg'
        ]

        print("Extracting frames with FFmpeg...")
        subprocess.run(ffmpeg_command, check=True)
        print("Frame extraction complete.")

    def process_video(self):
        """ Function to process video frames for object detection and tracking. """

        # Run FFmpeg extraction before processing frames
        self.extract_frames()
        
        # Load frames generated by FFmpeg
        frame_files = sorted(glob.glob(os.path.join(self.frames_output_dir, 'frame_*.jpg')))
        

        for frame_number, frame_file in enumerate(frame_files):
            print(f"Processing frame {frame_file}")
            frame = cv2.imread(frame_file)
            # Adjust timestamp for every 30th frame
            timestamp = frame_number * (30 / self.video_info['fps'])
            
            # Initial YOLO detection
            detections = self.detector.detect_objects(frame, timestamp)
            # Refine detection with tracking
            tracked_detections = self.tracker.update_tracks(detections, frame)

            self.log_entries[frame_number] = [vars(det) for det in tracked_detections]
        self.save_log()

    def save_log(self):
        log_file_path = os.path.join(self.output_dir, "detection_log.json")
        with open(log_file_path, 'w') as log_file:
            json.dump(self.log_entries, log_file, indent=4)
        print(f"Log saved to '{log_file_path}'.")
    

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='YOLO Object Detection with Color Filtering')
    parser.add_argument('--input', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Directory to save output frames and logs')
    parser.add_argument('--query', type=str, help='Search query in the format "color object", e.g., "red car"')
    parser.add_argument('--interval', type=int, default=30, help='Time interval in which to extract frames (default: every 30th frame)')

    args = parser.parse_args()
    argument_parsing_time = time.time()

    # Run detection on the video
    processor = VideoProcessor(
        video_path=args.input,
        output_dir=args.output,
        interval=args.interval
    )
    processor.process_video()

    print(f"Total time taken: {time.time() - start_time:.2f} seconds (Argument Parsing: {argument_parsing_time - start_time:.2f} seconds)")

if __name__ == '__main__':
    main()