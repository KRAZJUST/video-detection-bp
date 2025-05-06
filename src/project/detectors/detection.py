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
    def __init__(self, class_name: str, class_id: int, confidence: float,
                 bbox: tuple, timestamp: float, track_id: int = None, 
                 dominant_color: str = None):
        """
        Initialize the Detection object.
        Args:
            class_name (str): Name of the detected class
            class_id (int): ID of the detected class
            confidence (float): Confidence score of the detection
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
            timestamp (float): Timestamp of the detection
            track_id (int): ID of the track (optional)
            dominant_color (str): Dominant color of the detected object (optional)
        """
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = timestamp
        self.track_id = track_id
        self.dominant_color = dominant_color

    def get(self, key, default=None):
        """
        Get the value of a specific attribute.
        Args:
            key: Attribute name
            default: Default value if attribute does not exist
        Returns:
            The value of the attribute or the default value if it does not exist.
        """
        return getattr(self, key, default)