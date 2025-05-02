# =============================================================================
# File: detection.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module defines the Detection class, which is used to format information
# about detected objects.
#
# =============================================================================

class Detection:
    """ Data class to store detection information. """
    def __init__(self, class_name: str, class_id: int, confidence: float, bbox: tuple, timestamp: float, track_id: int = None, dominant_color: str = None):
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = timestamp
        self.track_id = track_id
        self.dominant_color = dominant_color

    def get(self, key, default=None):
        return getattr(self, key, default)