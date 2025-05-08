# =============================================================================
# File: video_processing_worker.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module provides a worker class for processing video files in a separate
# thread. It uses the VideoProcessor class to handle the video processing
# tasks. The worker emits signals to indicate progress, completion, or errors.
#
# =============================================================================

from profiling_utils.profiling_utils import profile_time_usage
from PyQt6.QtCore import QThread, pyqtSignal
from processors.video_processor import VideoProcessor

class VideoProcessingWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str, int)

    def __init__(self, app, video_path, database_path, output_dir, 
                 interval, tracker):
        """
        Initialize the worker with the necessary parameters.

        Args:
            app: Reference to the main application
            video_path: Path to the video file
            database_path: Path to the database file
            output_dir: Directory for output files
            interval: Interval for processing frames
            tracker: Tracker model name
        """
        super().__init__()
        self.app = app
        self.video_path = video_path
        self.database_path = database_path
        self.output_dir = output_dir
        self.interval = interval
        self.tracker = tracker
        self.terminate_processing = False

    def run(self):
        """
        Run the worker thread to process the video.
        This method is executed in a separate thread.
        It emits the finished signal when processing is complete,
        or the error signal if an error occurs.
        """
        try:
            # Progress fallback that emits message and perecentage
            def progress_callback(message, percentage):
                self.progress.emit(message, percentage)
                # Check termination flag during progress updates
                return not self.terminate_processing

            processor = VideoProcessor(
                video_path=self.video_path,
                database_path=self.database_path,
                output_dir=self.output_dir,
                interval=self.interval,
                model_name=self.tracker,
                use_segmentation=self.app.processing_segmentation_value,
                skip_siglip_with_yolo=self.app.skip_siglip_with_yolo_value,
                progress_callback=progress_callback,
            )
            
            # Check if processing was terminated
            if not self.terminate_processing:
                processor.process_video()

            # Emit finished signal
            self.finished.emit()
            
        except Exception as e:
            import traceback
            traceback.print_exc()

            self.error.emit(str(e))