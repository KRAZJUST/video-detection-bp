class Detection:
    """ Data class to store detection information. """
    def __init__(self, class_name: str, confidence: float, bbox: tuple, timestamp: float, track_id: int = None, dominant_color: str = None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = timestamp
        self.track_id = track_id
        self.dominant_color = dominant_color