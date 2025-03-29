import os
import json
import torch
import subprocess
import glob
from PIL import Image
from typing import Any, Dict, List
import cv2
from detectors.yolo_detector import YOLODetector
from detectors.byte_track_tracker import ByteTrackTracker
from database.sqlite_database import Database
from xclip.xclip_model import XClipModel
from database.vector_database import VectorDatabaseManager
from profiling_utils.profiling_utils import profile_time_usage
from interface.video_info_utils import VideoInfoUtils


class VideoProcessor:
    def __init__(self, video_path: str, 
                 output_dir: str, 
                 database_path: str,  
                 interval: int = 30, 
                 tracker_arg: str = 'bytetrack'):

        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_output_dir_yx = os.path.join(self.output_dir, "extracted_frames_yx")
        self.frames_output_dir_b = os.path.join(self.output_dir, "extracted_frames_b")
        self.frames_output_dir = self.frames_output_dir_b if tracker_arg == 'bytetrack' else self.frames_output_dir_yx
        # Create output directories if they do not exist
        os.makedirs(self.frames_output_dir_yx, exist_ok=True)
        os.makedirs(self.frames_output_dir_b, exist_ok=True)

        self.database_path = database_path
        self.db = Database(self.database_path)
        self.vector_db = VectorDatabaseManager(
            database_path="vector_database",
            collection_name=f"embeddings_{os.path.basename(video_path)}",
            reset_database=True
        )
        self.detector = YOLODetector()
        if tracker_arg == 'bytetrack':
            self.tracker = ByteTrackTracker(output_dir, min_frames_for_averaging=2, frame_width=640, frame_height=360)
        
        self.xclip = XClipModel()
        self.log_entries = {}
        self.initial_yolo_results_log = {}
        self.interval = interval
        self.tracker_arg = tracker_arg
        self.processing_level = 1 if self.tracker_arg in ['-', 'bytetrack'] else 2
        self.temp_embeddings = []

        # Get the video informations
        video_metadata = VideoInfoUtils.get_video_info(video_path)
        # Convert the video metadata to a dictionary
        self.video_info = vars(video_metadata) if video_metadata else {
            'width': None,
            'height': None,
            'duration': None,
            'frame_count': None,
            'fps': None
        }
        print(f"Video Info: {self.video_info}")

    def calculate_expected_frames(self) -> int:
        """Calculate the expected number of frames to be extracted."""
        if self.video_info.get('frame_count') is None:
            return None
        total_frames = self.video_info['frame_count']
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

    @profile_time_usage
    def extract_frames(self):
        # Check if frames are already extracted
        if self.frames_already_extracted():
            print("Frames already extracted")
            return True
        # Delete existing frames
        for f in os.listdir(self.frames_output_dir):
            if f.startswith("frame_") and f.endswith(".jpg"):
                os.remove(os.path.join(self.frames_output_dir, f))
        
        # Check if the CUDA is available
        use_cuda = torch.cuda.is_available()
        # Validate video information
        if not self.video_info['duration']:
            print("Could not determine video duration")
            return False
        
        # Prepare base FFmpeg command
        base_command = [
            'ffmpeg',
            '-fflags', '+genpts',
            '-i', self.video_path,
            '-vf', f"scale=640:-2, select='not(mod(n\\,{self.interval}))', format=yuvj420p",
            '-fps_mode', 'vfr',
            '-q:v', '2',
            '-pix_fmt', 'yuvj420p',
        ]
        
        # Add duration only if available as it can cause issues if not set properly
        if self.video_info['duration']:
            base_command.extend(['-to', str(self.video_info['duration'])])
        
        # Add output pat
        base_command.extend([
            '-start_number', '0',
            f'{self.frames_output_dir}/frame_%05d.jpg'
        ])
        
        # CUDA acceleration command
        if use_cuda:
            cuda_command = [
                'ffmpeg',
                '-hwaccel', 'cuda',
            ] + base_command[1:]
            ffmpeg_command = cuda_command
        else:
            ffmpeg_command = base_command
        
        # Print debug information
        print("FFmpeg Command:", " ".join(ffmpeg_command))
        
        try:
            # Execute FFmpeg command
            result = subprocess.run(
                ffmpeg_command,  
                text=True,
                check=True
            )
            
            # Check for errors
            if result.returncode != 0:
                print("FFmpeg Error Output:", result.stderr)
                return False
            
            print("Frames extracted successfully")
            return True
        
        except Exception as e:
            print(f"Execution error: {e}")
            return False

    @profile_time_usage
    def process_video(self):
        """ Function to process video frames for object detection and tracking. """

        # Run FFmpeg extraction before processing frames
        self.extract_frames()

        # Load frames generated by FFmpeg
        frame_files = sorted(glob.glob(os.path.join(self.frames_output_dir, 'frame_*.jpg')))
        
        if self.processing_level == 1:
            self._process_video_yolo(frame_files)
        elif self.processing_level == 2:
            self._process_video_xclip(frame_files)
        else:
            raise ValueError("Invalid tracker argument. Please use either '-', 'bytetrack' or 'xclip'.")

    def _process_video_yolo(self, frame_files):
        """
        YOLO and ByteTrack processing method
        """
        for frame_number, frame_file in enumerate(frame_files):
            #print(f"Processing frame {frame_file}")
            
            # Check if the fps is set as it can be 'unknown' in some cases and cause division by zero
            if self.video_info['fps'] != 'unknown':
                timestamp = frame_number * (self.interval / self.video_info['fps'])
            else:
                # If the fps is unknown, use the frame number as the timestamp
                timestamp = frame_number * self.interval

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
            else:
                tracked_detections = self.tracker.update_tracks(results, frame, frame_number)
                self.log_entries[frame_number] = tracked_detections
                self.add_detections_in_db(tracked_detections, tracker=self.tracker_arg, frame_number=frame_number)

        #self.save_log(self.initial_yolo_results_log, name='initial_yolo_results_log')
        #self.save_log(self.log_entries)

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
