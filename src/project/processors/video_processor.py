import os
import json
import torch
import subprocess
import glob
from PIL import Image
from typing import Any, Dict, List
import cv2
import time
from detectors.yolo_detector import YOLODetector
from detectors.byte_track_tracker import ByteTrackTracker
from database.sqlite_database import Database
from xclip.xclip_model import XClipModel
from database.vector_database import VectorDatabaseManager


class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, database_path: str, interval: int = 30, tracker_arg: str = 'bytetrack'):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_output_dir = os.path.join(self.output_dir, "extracted_frames")
        os.makedirs(self.frames_output_dir, exist_ok=True)
        self.database_path = database_path
        self.db = Database(self.database_path)
        self.vector_db = VectorDatabaseManager(
            database_path="vector_database",
            collection_name=f"embeddings_{os.path.basename(video_path)}",
            reset_database=True
        )
        self.detector = YOLODetector()
        self.tracker = ByteTrackTracker(self.output_dir)
        self.xclip = XClipModel()
        self.log_entries = {}
        self.initial_yolo_results_log = {}
        self.video_info = self.get_video_info()
        self.interval = interval
        self.tracker_arg = tracker_arg
        self.processing_level = 1 if self.tracker_arg == '-' or self.tracker_arg == 'bytetrack' else 2
        self.temp_embeddings = []

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
    

    def calculate_expected_frames(self) -> int:
        """Calculate the expected number of frames to be extracted."""

        total_frames = self.video_info['nb_frames']
        return total_frames // self.interval
    

    def frames_already_extracted(self) -> bool:
        """Check if the required number of frames has already been extracted."""

        # Count the number of frame files in the output directory
        existing_frames_count = len([
            f for f in os.listdir(self.frames_output_dir) 
            if f.startswith("frame_") and f.endswith(".jpg")
        ])
        
        # Calculate the expected number of frames
        expected_frames_count = self.calculate_expected_frames()
        
        # Return True if the expected frames are already extracted, False otherwise
        return existing_frames_count + 3 >= expected_frames_count and existing_frames_count - 3 <= expected_frames_count

    def extract_frames(self):
        """ Function to extract frames from a video using FFmpeg. """
        print(self.video_info)

        # Check if frames already extracted
        if self.frames_already_extracted():
            print("Frames are already extracted. Skipping extraction.")
            return

        ffmpeg_command = [
            'ffmpeg', '-fflags', '+genpts', '-i', self.video_path,
            '-vf', f"scale=640:-2, select='not(mod(n\\,{self.interval}))', format=yuvj420p",
            '-fps_mode', 'vfr',
            # Set higher quality for better object detection
            '-q:v', '2',
            # Sets pixel format to yuvj420p for better accuracy in object detection
            '-pix_fmt', 'yuvj420p',
            '-to', str(self.video_info['duration']),
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

        start_time = time.time()
        # Load frames generated by FFmpeg
        frame_files = sorted(glob.glob(os.path.join(self.frames_output_dir, 'frame_*.jpg')))
        
        if self.processing_level == 1:
            self._process_video_yolo(frame_files)
        elif self.processing_level == 2:
            self._process_video_xclip(frame_files)
        else:
            raise ValueError("Invalid tracker argument. Please use either '-', 'bytetrack' or 'xclip'.")

        print(f"Processing complete. Total time taken: {time.time() - start_time:.2f} seconds.")

    def _process_video_yolo(self, frame_files):
        """
        YOLO and ByteTrack processing method
        """
        for frame_number, frame_file in enumerate(frame_files):
            print(f"Processing frame {frame_file}")
            
            # Calculate timestamp
            timestamp = frame_number * (self.interval / self.video_info['fps'])
            
            # Insert frame into the database
            self.db.insert_frame(frame_number, timestamp)
            
            # Read frame
            frame = cv2.imread(frame_file)
            
            # Detect objects
            results, detections = self.detector.detect_objects(frame, timestamp)
            self.initial_yolo_results_log[frame_number] = [vars(det) for det in detections]
            
            # Handle tracking
            if self.tracker_arg == '-':
                self.add_detections_in_db(detections, tracker=self.tracker_arg, frame_number=frame_number)
            elif self.tracker_arg == 'bytetrack':
                tracked_detections = self.tracker.update_tracks(results, frame, frame_number)
                self.log_entries[frame_number] = tracked_detections
                self.add_detections_in_db(tracked_detections, tracker=self.tracker_arg, frame_number=frame_number)

        self.save_log(self.initial_yolo_results_log, name='initial_yolo_results_log')
        self.save_log(self.log_entries)

    def _process_video_xclip(self, frame_files):
        """
        X-CLIP processing method
        """
        batch_size = 8
        frame_batches = self.create_frame_batches(frame_files, batch_size)
        # Reset temp_embeddings
        self.temp_embeddings = []

        for batch_number, frame_batch in enumerate(frame_batches):
            # Load frames for the batch
            frames = self.load_frames_as_clip(frame_batch)

            # Pad the batch if the number of frames is less than the batch size
            if len(frames) < batch_size:
                # Repeat the last frame to fill the batch
                padding_needed = batch_size - len(frames)
                frames.extend([frames[-1]] * padding_needed)
                
            # Generate embeddings for the frames using X-CLIP
            embeddings = self.xclip.extract_embeddings(frames)
            print(f"Batch {batch_number}: Extracted embeddings: {embeddings.shape}")

            # Prepare metadata for each embedding
            metadata = [
                {
                    "frame_path": frame_path,
                    "batch_number": batch_number,
                    "frame_number": frame_files.index(frame_path)
                } for frame_path in frame_batch
            ]

            # Add embeddings to the vector database
            self.vector_db.add_batch_embeddings(batch_embeddings=embeddings, batch_metadata=metadata, embedding_strategy='mean')

        print("Embeddings stored in vector database successfully.")

    def add_detections_in_db(self, detections, tracker: str, frame_number: int):
        """Accumulate detections and insert in bulk into the database."""
        # List to hold all detections for the current frame
        bulk_detections = []

        for detection in detections:
            detection_data = {
                'frame_number': frame_number,
                'track_id': detection.get('track_id', None),
                'direction': detection.get('direction', None),
                'class_name': detection.get('class_name', None),
                'class_id': detection.get('class_id', None),
                'bbox': detection.get('bbox', None),
                'confidence': detection.get('confidence', None),
                'timestamp': detection.get('timestamp', None),
                'dominant_color': detection.get('dominant_color', None),
            }
            # Append to the bulk list
            bulk_detections.append(detection_data)

        # Insert all detections for this frame in bulk to the database
        # Choose the table based on the tracker argument
        if tracker == '-':
            self.db.bulk_insert_detections(bulk_detections)
        elif tracker == 'bytetrack':
            self.db.bulk_insert_refined_detections(bulk_detections)

    def save_log(self, log, name: str = 'detection_log'):
        log_file_path = os.path.join(self.output_dir, f"{name}.json")
        with open(log_file_path, 'w') as log_file:
            json.dump(log, log_file, indent=4)
        print(f"Log saved to '{log_file_path}'.")


    def create_frame_batches(self, frame_files, batch_size):
        """
        Organize extracted frames into batches for X-CLIP processing.
        Args:
            frame_files (list): List of frame file paths, sorted by frame number.
            batch_size (int): Number of frames in each batch/clip.
        Returns:
            List of frame batches (each batch is a list of frame paths).
        """
        batches = [frame_files[i:i + batch_size] for i in range(0, len(frame_files), batch_size)]
        return batches

    def load_frames_as_clip(self, frame_paths):
        """
        Load frames from file paths and prepare them as an X-CLIP input.
        Args:
            frame_paths (list): List of frame paths for a single clip.
        Returns:
            List of PIL.Image objects.
        """
        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
        return frames
