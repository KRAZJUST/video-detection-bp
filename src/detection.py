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
from typing import Dict, Any, List

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class Detection:
    """ Data class to store detection information. """

    def __init__(self, class_name: str, confidence: float, bbox: tuple, timestamp: float, track_id: int = None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = timestamp
        self.track_id = track_id

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
                if class_name in ['person', 'car', 'truck', 'bus', 'motorcycle']:
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
    # TODO: FIX THIS CLASS
    """ Class to perform object tracking using DeepSort. """

    def __init__(self, max_age: int = 30, nn_budget: int = 100):
        # Initialize DeepSort tracker
        self.tracker = DeepSort(max_age=max_age, nn_budget=nn_budget)

    def refine_tracking(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        # Prepare detections in the format expected by DeepSort
        formatted_detections = [
            (det.bbox, det.confidence, det.class_name) for det in detections
        ]
        
        # Perform tracking
        tracked_objects = self.tracker.update_tracks(
            formatted_detections, frame=frame
        )

        # Assign tracking IDs to detections
        tracked_detections = []
        for track in tracked_objects:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            # bounding box in top-left to bottom-right format
            bbox = track.to_tlbr()
            bbox_int = np.round(bbox).astype(int)
            class_name = track.det_class

            # Debugging: Print detections and bounding box for comparison
            print("Current bounding box:", bbox)
            print("Current bbxos as int:", bbox_int)
            print("Current detections:", [(det.bbox, det.timestamp) for det in detections])

            timestamp = next((det.timestamp for det in detections if np.array_equal(det.bbox, tuple(bbox_int))), None)
            if timestamp is None:
                print("Timestamp not found for detection.")
                continue

            # Add tracking ID and bounding box to detection data
            tracked_detections.append(Detection(
                class_name=class_name,
                confidence=track.det_conf,
                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                timestamp=timestamp,
                track_id=track_id
            ))
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
        self.video_info = self.get_video_info(self.video_path)
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
        
        # Obtain video info
        video_info = self.get_video_info(self.video_path)
        
        for frame_number, frame_file in enumerate(frame_files):
            print(f"Processing frame {frame_file}")
            frame = cv2.imread(frame_file)
            # Adjust timestamp for every 30th frame
            timestamp = frame_number * (30 / video_info['fps'])
            
            # Initial YOLO detection
            detections = self.detector.detect_objects(frame, timestamp)
            # Refine detection with tracking
            #tracked_detections = self.tracker.refine_tracking(frame, detections)

            self.log_entries[frame_number] = [vars(det) for det in detections]
        self.save_log()

    def save_log(self):
        log_file_path = os.path.join(self.output_dir, "detection_log.json")
        with open(log_file_path, 'w') as log_file:
            json.dump(self.log_entries, log_file, indent=4)
        print(f"Log saved to '{log_file_path}'.")

    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        # Extract FPS and other video metadata
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        return {'fps': fps, 'duration': duration, 'frame_count': frame_count}
    

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
        interval=args.interval
    )
    processor.process_video()

    print(f"Total time taken: {time.time() - start_time:.2f} seconds (Argument Parsing: {argument_parsing_time - start_time:.2f} seconds)")

if __name__ == '__main__':
    main()