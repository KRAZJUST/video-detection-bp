import os
import torch
import subprocess
import glob
import time
import math
from PIL import Image
from typing import Any, Dict, List
import cv2
from detectors.yolo_detector import YOLODetector
from detectors.byte_track_tracker import ByteTrackTracker
from database.sqlite_database import Database
from xclip.xclip_model import XClipModel
from database.vector_database import VectorDatabaseManager
from profiling_utils.profiling_utils import profile_time_usage, detailed_profile
from interface.video_info_utils import VideoInfoUtils
from siglip.siglip_model import SigLIPModel


class VideoProcessor:
    def __init__(self, video_path: str, 
                 output_dir: str, 
                 database_path: str,  
                 interval: int = 30, 
                 model_name: str = 'yolo',
                 use_segmentation: bool = False,
                 skip_siglip_with_yolo: bool = False,
                 progress_callback: Any = None):

        self.video_path = video_path
        self.output_dir = output_dir
        self.frames_output_dir_yx = os.path.join(self.output_dir, "extracted_frames_yx")
        self.frames_output_dir_b = os.path.join(self.output_dir, "extracted_frames_b")
        self.frames_output_dir = self.frames_output_dir_b if model_name == 'bytetrack' else self.frames_output_dir_yx
        # Create output directories if they do not exist
        os.makedirs(self.frames_output_dir_yx, exist_ok=True)
        os.makedirs(self.frames_output_dir_b, exist_ok=True)

        self.database_path = database_path
        self.db = Database(self.database_path)
        self.vector_db = VectorDatabaseManager(
            database_path="vector_database",
            collection_name=f"{model_name}_embeddings_{os.path.basename(video_path)}",
            reset_database=True
        )
        self.skip_siglip_with_yolo = skip_siglip_with_yolo
        self.model_name = model_name
        # Initialize the model based on the tracker argument
        if self.model_name in ['yolo', 'bytetrack']:
            # Use segmentation only if the model is YOLO
            if use_segmentation and self.model_name == 'yolo':
                print("Using YOLO with segmentation")
                self.detector = YOLODetector(use_segmentation=use_segmentation)
            else:
                if self.model_name == 'yolo':
                    self.detector = YOLODetector()
                elif self.model_name == 'bytetrack':
                    self.detector = YOLODetector(skip_color_analysis=True)
        if self.model_name == 'bytetrack':
            self.tracker = ByteTrackTracker(output_dir, min_frames_for_averaging=2, 
                                            frame_width=640, frame_height=360)
        if self.model_name == 'xclip-32' or self.model_name == 'xclip-16':
            self.xclip = XClipModel(self.model_name)
        if self.model_name == 'siglip':
            self.siglip_model = SigLIPModel()

        self.interval = interval
        self.temp_embeddings = []
        # Set the processing level based on the tracker argument
        if self.model_name in ['yolo', 'bytetrack']:
            self.processing_level = 1
        elif self.model_name in ['xclip-32', 'xclip-16']:
            self.processing_level = 2
        elif self.model_name == 'siglip':
            self.processing_level = 3
        else:
            raise ValueError("Invalid tracker argument. Please use either 'yolo', 'bytetrack' or 'xclip'.")

        # Initialize the flag to check if processing should be terminated
        self.should_terminate = False
        # Initialize the progress callback
        self.progress_callback = progress_callback
        # processing stages and their relative weights in overall progress
        self.stages = {
            'initialization': {'weight': 5, 'completed': 0},
            'frame_extraction': {'weight': 25, 'completed': 0},
            'object_detection': {'weight': 70, 'completed': 0}
        }
        
        # Get the video informations
        self.update_progress("Getting video information",
                             stage='initialization', progress=30)
        video_metadata = VideoInfoUtils.get_video_info(video_path)
        # Convert the video metadata to a dictionary
        self.video_info = vars(video_metadata) if video_metadata else {
            'width': None,
            'height': None,
            'duration': None,
            'frame_count': None,
            'fps': None
        }
        self.update_progress("Video information retrieved",
                             stage='initialization', progress=100)
        print(f"Video Info: {self.video_info}")


        # Calculate the expected number of frames
        self.expected_frames_count = self.calculate_expected_frames()

    def calculate_overall_progress(self):
        """Calculate overall progress based on weighted stages"""
        total_progress = 0
        total_weight = sum(stage['weight'] for stage in self.stages.values())
        
        for stage_info in self.stages.values():
            stage_contribution = (stage_info['completed'] * stage_info['weight']) / 100
            total_progress += stage_contribution
            
        return int(total_progress * 100 / total_weight)

    def update_progress(self, message, stage=None, progress=None):
        """
        Update progress with a status message and percentage
        
        Args:
            message (str): Status message to display
            stage (str): Current processing stage
            progress (int): Progress percentage for the current stage (0-100)
        """
        # Update stage progress if provided
        if stage is not None and progress is not None:
            if stage in self.stages:
                self.stages[stage]['completed'] = progress
        
        # Calculate overall progress
        overall_progress = self.calculate_overall_progress()
        
        # Call the progress callback with message and percentage
        if self.progress_callback:
            result = self.progress_callback(message, overall_progress)
            if result is False:
                self.should_terminate = True
                print("Processing terminated by user.")

    def calculate_expected_frames(self) -> int:
        """Calculate the expected number of frames to be extracted."""
        if self.video_info.get('frame_count') is None:
            return None
        total_frames = self.video_info['frame_count']
        return math.ceil(total_frames / self.interval)
    
    def frames_already_extracted(self) -> bool:
        """Check if the required number of frames has already been extracted."""
        self.update_progress("Checking existing frames",
                             stage='frame_extraction', progress=10)
        # Count the number of frame files in the output directory
        existing_frames_count = len([
            f for f in os.listdir(self.frames_output_dir) 
            if f.startswith("frame_") and f.endswith(".jpg")
        ])
        
        # Return True if the expected frames are already extracted, False otherwise
        return abs(existing_frames_count - self.expected_frames_count) < 2

    @profile_time_usage
    @detailed_profile
    def extract_frames(self):
        # Check if frames are already extracted
        if self.frames_already_extracted():
            self.update_progress("Frames already extracted",
                                    stage='frame_extraction', progress=100)
            print("Frames already extracted")
            return True
        
        # Delete existing frames
        self.update_progress("Deleting existing frames",
                             stage='frame_extraction', progress=20)
        for f in os.listdir(self.frames_output_dir):
            if f.startswith("frame_") and f.endswith(".jpg"):
                os.remove(os.path.join(self.frames_output_dir, f))
        
        # Check if the CUDA is available
        use_cuda = torch.cuda.is_available()

        # Validate video information
        if not self.video_info['duration']:
            self.update_progress("Could not determine video duration",
                                 stage='frame_extraction', progress=30)
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
            f'{self.frames_output_dir}/frame_%06d.jpg'
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
            self.update_progress("Starting frame extraction", stage='frame_extraction', progress=30)
            # Check if the extraction should be terminated
            if self.should_terminate:
                self.update_progress("Frame extraction terminated by user", stage='frame_extraction', progress=90)
                print("Frame extraction terminated by user")
                return False

            # Start ffmpeg in a separate process
            process = subprocess.Popen(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )
            
            # Progress update variables
            start_progress = 30
            end_progress = 90
            progress_range = end_progress - start_progress
            
            # Rough estimate total extraction time based on video duration 
            # duration / 12x real time processing
            estimated_total_seconds = float(self.video_info['duration']) / 12.0 
            update_interval = 2.0
            
            # Start time for progress calculation
            start_time = time.time()
            
            # Update progress in a loop while process is running
            while process.poll() is None:
                # Check if the extraction should be terminated
                if self.should_terminate:
                    process.terminate()
                    process.wait()
                    self.update_progress("Frame extraction terminated by user", stage='frame_extraction', progress=90)
                    print("Frame extraction terminated by user")
                    return False
                elapsed_time = time.time() - start_time
                
                # Calculate progress as a percentage of estimated time
                progress_percent = min(elapsed_time / estimated_total_seconds, 1.0)
                current_progress = start_progress + int(progress_percent * progress_range)
                
                # Update progress
                self.update_progress(f"Extracting frames", 
                                stage='frame_extraction', 
                                progress=current_progress)
                
                # Sleep for the update interval
                time.sleep(update_interval)
            
            # Check if process completed successfully
            if process.returncode != 0:
                self.update_progress("Error during frame extraction", stage='frame_extraction', progress=90)
                print("FFmpeg Error")
                return False
            
            self.update_progress("Frames extracted successfully", stage='frame_extraction', progress=100)
            print("Frames extracted successfully")
            return True
        
        except Exception as e:
            self.update_progress(f"Error during frame extraction: {str(e)}")
            print(f"Execution error: {e}")
            return False

    @profile_time_usage
    @detailed_profile
    def process_video(self):
        """ Function to process video frames for object detection and tracking. """
        self.update_progress("Starting video processing",
                             stage='initialization', progress=50)
        # Check if processing should be terminated
        if self.should_terminate:
            self.update_progress("Video processing terminated by user", stage='initialization', progress=0)
            return False
         # Close and reopen the database connection to ensure a fresh state
        self.db.close()
        self.db.connect()
        
        # Add video information to the database
        self.update_progress("Adding video information to the database",
                             stage='initialization', progress=80)
        # Check if processing should be terminated
        if self.should_terminate:
            self.update_progress("Video processing terminated by user", stage='initialization', progress=0)
            return False
        # Check if the video is already processed and the frames are already extracted
        # if the extraction interval is different from the one used in the database
        # so the number of frames is different, delete the frames before reprocessing
        if self.model_name in ['yolo', 'xclip-32', 'xclip-16', 'siglip'] and \
           abs((self.db.get_number_of_frames_yolo(self.video_path) - self.expected_frames_count)) > 1:
                print(f"got {self.db.get_number_of_frames_yolo(self.video_path)} frames")
                print(f"expected {self.expected_frames_count} frames")
                self.update_progress("Resetting YOLO database for this video...",
                                    stage='initialization', progress=70)
                self.db.reset_video_yolo(self.video_path)
        elif self.model_name == 'bytetrack' and \
             abs((self.db.get_number_of_frames_bytetrack(self.video_path) - self.expected_frames_count)) > 1:
                self.update_progress("Resetting ByteTrack database for this video...",
                                    stage='initialization', progress=70)
                self.db.reset_video_bytetrack(self.video_path)
        
        self.update_progress("Initialization completed",
                             stage='initialization', progress=100)
        # Check if processing should be terminated
        if self.should_terminate:
            self.update_progress("Video processing terminated by user", stage='initialization', progress=0)
            return False
        
        # Run FFmpeg extraction before processing frames
        if not self.extract_frames():
            self.update_progress("Frame extraction failed", stage='initialization', progress=0)
            return False

        # Load frames generated by FFmpeg
        self.update_progress("Loading extracted frames", 
                             stage='object_detection', progress=5)
        frame_files = sorted(glob.glob(os.path.join(self.frames_output_dir, 'frame_*.jpg')))
        
        if self.processing_level == 1:
            self.update_progress(f"Processing frames with {self.model_name}",
                                 stage='object_detection', progress=10)
            self._process_video_yolo(frame_files)
        elif self.processing_level == 2:
            self.update_progress(f"Processing frames with {self.model_name}",
                                 stage='object_detection', progress=10)
            self._process_video_xclip(frame_files)
        elif self.processing_level == 3:
            self.update_progress(f"Processing frames with {self.model_name}",
                                 stage='object_detection', progress=10)
            self._process_video_siglip(frame_files)
        else:
            self.update_progress("Invalid processing configuration")
            raise ValueError("Invalid tracker argument. Please use either 'yolo', 'bytetrack' or 'xclip'.")
        
        # Finalize processing
        self.update_progress("Completed processing frames",
                                 stage='object_detection', progress=100)

    def _process_video_yolo(self, frame_files):
        """
        YOLO and ByteTrack processing method
        """
        total_frames = len(frame_files)
        if total_frames < 1000:
            update_interval = 20
        elif total_frames < 3000:
            update_interval = 100
        else:
            update_interval = 200 

        for frame_number, frame_file in enumerate(frame_files):
            # Check if processing should be terminated every 10 frames
            if frame_number % 10 == 0 and self.should_terminate:
                self.update_progress("Video processing terminated by user", stage='object_detection', progress=0)
                return False

            # Calculate progress percentage (10-90% of object_detection stage)
            progress_percent = 10 + int((frame_number / total_frames) * 80)
            # Update progress every update_interval frames
            if frame_number == 0 or frame_number == total_frames - 1 or frame_number % update_interval == 0:
                self.update_progress(
                    f"Processing frames ({frame_number + 1}/{total_frames})",
                    stage='object_detection', 
                    progress=progress_percent
                )
            
            # Check if the fps is set as it can be 'unknown' in some cases and cause division by zero
            if self.video_info['fps'] != 'unknown':
                timestamp = frame_number * (self.interval / self.video_info['fps'])
            else:
                # If the fps is unknown, use the frame number as the timestamp
                timestamp = frame_number * self.interval

            # Insert frame into the database
            if self.model_name == 'yolo':
                self.db.insert_yolo_frame(self.video_path, frame_number, timestamp)
            elif self.model_name == 'bytetrack':
                self.db.insert_bytetrack_frame(self.video_path, frame_number, timestamp)
            
            # Read frame
            frame = cv2.imread(frame_file)
            
            # Detect objects
            results, detections = self.detector.detect_objects(frame, timestamp)
            
            # Handle tracking
            if self.model_name == 'yolo':
                self.add_detections_in_db(detections, tracker=self.model_name, frame_number=frame_number)
                self.processed_with_yolo = True
            else:
                tracked_detections = self.tracker.update_tracks(results, frame, frame_number)
                self.add_detections_in_db(tracked_detections, tracker=self.model_name, frame_number=frame_number)

    def _process_video_xclip(self, frame_files):
        """
        X-CLIP processing method
        """
        batch_size = 8
        
        # mapping of frame paths to their indices for lookups used later
        # This is a dictionary comprehension to create a mapping of frame paths to their indices
        # This allows for O(1) lookups instead of O(n) using list.index() 
        # so it is more efficient in our case when working with large number of frames
        frame_index_map = {frame_path: idx for idx, frame_path in enumerate(frame_files)}
        
        frame_batches = self.create_frame_batches(frame_files, batch_size)
        total_batches = len(frame_batches)
        
        # Reset temp_embeddings
        self.temp_embeddings = []

        for batch_number, frame_batch in enumerate(frame_batches):
            # Check if processing should be terminated every 5 batches
            if batch_number % 5 == 0 and self.should_terminate:
                self.update_progress("Video processing terminated by user", stage='object_detection', progress=0)
                return False
            
            # Calculate progress percentage (10-90% of object_detection stage)
            progress_percent = 10 + int((batch_number / total_batches) * 80)
            
            self.update_progress(
                f"Processing batches ({batch_number + 1}/{total_batches})",
                stage='object_detection', 
                progress=progress_percent
            )

            # Load frames for the batch
            frames = self.load_frames_as_clip(frame_batch)

            # Pad the batch if the number of frames is less than the batch size
            if len(frames) < batch_size:
                # Repeat the last frame to fill the batch
                padding_needed = batch_size - len(frames)
                frames.extend([frames[-1]] * padding_needed)
                
            # Generate embeddings for the frames using X-CLIP
            embeddings = self.xclip.extract_embeddings(frames)
          
            # Prepare metadata for each embedding with O(1) lookups
            metadata = []
            for frame_path in frame_batch:
                # Use the pre-computed index from the mapping
                frame_idx = frame_index_map[frame_path]
                
                # Calculate timestamp
                if self.video_info['fps'] != 'unknown':
                    timestamp = frame_idx * (self.interval / self.video_info['fps'])
                else:
                    timestamp = frame_idx * self.interval
                    
                metadata.append({
                    "frame_path": frame_path,
                    "batch_number": batch_number,
                    "frame_number": frame_idx,
                    "timestamp": timestamp,
                })

            # Add embeddings to the vector database
            self.vector_db.add_batch_embeddings(batch_embeddings=embeddings, batch_metadata=metadata, embedding_strategy='mean')

        print("Embeddings stored in vector database successfully.")

    @profile_time_usage
    def _process_video_siglip(self, frame_files):
        """Process individual video frames with SigLIP model"""
        total_frames = len(frame_files)
        if total_frames < 1000:
            update_interval = 20
        elif total_frames < 3000:
            update_interval = 100
        else:
            update_interval = 200

        if self.skip_siglip_with_yolo:
            # pre-fetch all frames with detections
            frames_with_detections = self.db.get_frames_with_yolo_detections(self.video_path)
            print(len(frames_with_detections), "frames with detections")
            if len(frames_with_detections) == 0:
                # no frames with detections, process all frames
                frames_with_detections = set(range(total_frames))
        else:
            # process all frames
            frames_with_detections = set(range(total_frames))

        for frame_idx, frame_path in enumerate(frame_files):
            # Check if processing should be terminated every 10 frames
            if frame_idx % 10 == 0 and self.should_terminate:
                self.update_progress("Video processing terminated by user", stage='object_detection', progress=0)
                return False
            
            # Skip frames without YOLO detections
            if frame_idx not in frames_with_detections:
                continue

            # Calculate progress percentage (10-90% of object_detection stage)
            progress_percent = 10 + int((frame_idx / total_frames) * 80)
            # Update progress every update_interval frames
            if frame_idx == 0 or frame_idx == total_frames - 1 or frame_idx % update_interval == 0:
                self.update_progress(
                    f"Processing frames ({frame_idx + 1}/{total_frames})",
                    stage='object_detection', 
                    progress=progress_percent
                )
            
            # Load the frame
            image = Image.open(frame_path)
            
            # Generate embedding with SigLIP
            embedding = self.siglip_model.extract_embedding(image)
            
             # Calculate timestamp
            if self.video_info['fps'] != 'unknown':
                timestamp = frame_idx * (self.interval / self.video_info['fps'])
            else:
                timestamp = frame_idx * self.interval
            
            # Add embedding and metadata for a single frame into the vector database
            self.vector_db.add_single_frame_embedding(
                embedding=embedding,
                frame_path=frame_path,
                frame_number=frame_idx,
                timestamp=timestamp
            )
        
        print("SigLIP embeddings stored in vector database successfully.")

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
        if tracker == 'yolo':
            self.db.bulk_insert_detections(self.video_path, bulk_detections)
        elif tracker == 'bytetrack':
            self.db.bulk_insert_refined_detections(self.video_path, bulk_detections)

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
