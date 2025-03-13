from PyQt6.QtCore import QThread, pyqtSignal
from .video_info_utils import VideoInfoUtils

class VideoInfoWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            video_metadata = VideoInfoUtils.get_video_info(self.video_path)
            if video_metadata:
                self.finished.emit(vars(video_metadata))
            else:
                self.error.emit("Error reading video information")
        except Exception as e:
            self.error.emit(str(e))