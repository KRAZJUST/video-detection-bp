from profiling_utils.profiling_utils import profile_time_usage
from PyQt6.QtCore import QThread, pyqtSignal
from processors.video_processor import VideoProcessor

class VideoProcessingWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str, int)

    def __init__(self, app, video_path, database_path, output_dir, interval, tracker):
        super().__init__()
        self.app = app
        self.video_path = video_path
        self.database_path = database_path
        self.output_dir = output_dir
        self.interval = interval
        self.tracker = tracker

    @profile_time_usage
    def run(self):
        try:
            # Progress fallback that emits message and perecentage
            def progress_callback(message, percentage):
                self.progress.emit(message, percentage)

            processor = VideoProcessor(
                video_path=self.video_path,
                database_path=self.database_path,
                output_dir=self.output_dir,
                interval=self.interval,
                model_name=self.tracker,
                progress_callback=progress_callback,
            )
            processor.process_video()
            
            # Emit finished signal
            self.finished.emit()
            
        except Exception as e:
            import traceback
            traceback.print_exc()

            self.error.emit(str(e))