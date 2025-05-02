# =============================================================================
# File: video_info_worker.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module provides a worker class for retrieving video information
# using VideoInfoUtils. It runs in a separate thread to avoid blocking
# the main UI thread. The worker emits signals when the video information
# is successfully retrieved or if an error occurs.
#
# =============================================================================

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