import time
from PyQt6.QtCore import QThread, pyqtSignal
from processors.video_processor import VideoProcessor

class VideoProcessingWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, app, video_path, database_path, output_dir, interval, tracker):
        super().__init__()
        self.app = app
        self.video_path = video_path
        self.database_path = database_path
        self.output_dir = output_dir
        self.interval = interval
        self.tracker = tracker

    def run(self):
        try:
            start_time = time.time()
            processor = VideoProcessor(
                video_path=self.video_path,
                database_path=self.database_path,
                output_dir=self.output_dir,
                interval=self.interval,
                tracker_arg=self.tracker
            )
            processor.process_video()
            print(f'Time taken to process video: {time.time() - start_time}')
            
            # Store results in the main app
            if self.tracker == "-":
                self.app.log_results = processor.initial_yolo_results_log
            elif self.tracker in ["bytetrack"]:
                self.app.log_results = processor.log_entries
            elif self.tracker == "xclip":
                self.app.temp_embeddings = processor.temp_embeddings
                
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))