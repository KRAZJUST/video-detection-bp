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
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort


class Detection:
    """ Data class to store detection information. """
    def __init__(self, class_name: str, confidence: float, bbox: tuple, timestamp: float, track_id: int = None, dominant_color: str = None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = timestamp
        self.track_id = track_id
        self.dominant_color = dominant_color

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


class YOLODetector:
    """ Class to perform object detection using YOLOv11. """

    def __init__(self, model_path: str = 'yolo11n.pt', device: str = 'auto'):
        self.model = YOLO(model_path)
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.class_names = self.model.names

    def detect_objects(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        results = self.model.predict(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls[0])
                class_name = self.class_names.get(class_idx, "Unknown")
                # Detect only people and vehicles
                if class_name in ['person', 'car', 'truck', 'bus']:
                    confidence = float(box.conf[0])
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(xmin, ymin, xmax, ymax),
                        timestamp=timestamp
                    ))

        return result[0], detections


class ByteTrackTracker:
    def __init__(self, output_dir: str):
        # Initialize ByteTrack from supervision
        self.tracker = sv.ByteTrack()
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
        print('Parsed YOLO Detections:', sv_detections)
        print('-' * 50)
        
        # Update ByteTrack with detections
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        print('Tracked Detections:', tracked_detections)
        print('-' * 50)
        
        parsed_detections = self.parse_tracked_detections(tracked_detections)

        # Annotate the frame with bounding boxes and labels for better visualization TODO: remove later for performance
        annotated_frame = self.annotate_frame(frame, tracked_detections)

        # Save the annotated frame
        annotated_frame_path = os.path.join(self.annotated_images_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(annotated_frame_path, annotated_frame)


        return parsed_detections


class DeepSortDetector:
    def __init__(self, output_dir, max_age=30, nn_budget: int = 100, use_gpu: bool = True):
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
            max_cosine_distance=0.3,                                    # Cosine distance for feature matching
            nn_budget=nn_budget,                                        # Maximum size of the appearance descriptor collection
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
            det.dominant_color = self.color_filter.detect_dominant_color(frame, det.bbox)
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
            label = f"ID: {det.track_id}, Class: {det.class_name}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the annotated frame
        annotated_frame_path = os.path.join(self.annotated_images_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(annotated_frame_path, frame)

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, interval: int = 30, tracker_arg: str = 'bytetrack'):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_output_dir = os.path.join(self.output_dir, "extracted_frames")
        os.makedirs(self.frames_output_dir, exist_ok=True)
        self.detector = YOLODetector()
        self.tracker = ByteTrackTracker(self.output_dir)
        self.deepsort_tracker = DeepSortDetector(self.output_dir)
        self.log_entries = {}
        self.video_info = self.get_video_info()
        self.interval = interval
        self.tracker_arg = tracker_arg

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
            '-vf', f"scale=640:-1, select='not(mod(n\\,{self.interval}))'",
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
            results, detections = self.detector.detect_objects(frame, timestamp)
            # Refine detection with tracking, choose the tracker based on the argument
            if(self.tracker_arg == 'bytetrack'):
                tracked_detections = self.tracker.update_tracks(results, frame, frame_number)
                self.log_entries[frame_number] = tracked_detections
            elif(self.tracker_arg == 'deepsort'):
                tracked_detections = self.deepsort_tracker.track(detections, frame, frame_number=frame_number)
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
    parser.add_argument('--tracker', type=str, default='bytetrack', help='Which tracker to use - deepsort or bytetrack')

    args = parser.parse_args()
    argument_parsing_time = time.time()

    # Run detection on the video
    processor = VideoProcessor(
        video_path=args.input,
        output_dir=args.output,
        interval=args.interval,
        tracker_arg=args.tracker
    )
    processor.process_video()

    print(f"Total time taken: {time.time() - start_time:.2f} seconds (Argument Parsing: {argument_parsing_time - start_time:.2f} seconds)")

if __name__ == '__main__':
    main()