from PyQt6.QtCore import QThread, pyqtSignal
from parsers.detection_parser import DetectionParser
from xclip.xclip_parser import XClipParser

class QueryWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, app, query):
        super().__init__()
        self.app = app
        self.query = query
        self.deduplicate = True

    def run(self):
        try:
            if self.app.tracker_combo.currentText() in ["-", "bytetrack"]:
                log_parser = DetectionParser(
                    log_entries=self.app.log_results,
                    query=self.query,
                    output_dir=self.app.output_dir,
                    database_path=self.app.database_path,
                    tracker=self.app.tracker_combo.currentText(),
                    use_segmentation=self.app.use_segmentation.isChecked()
                )
                log_parser.parse_detections()

                if self.deduplicate:
                    # Deduplicate results based on tracked object IDs and classes
                    if self.app.tracker_combo.currentText() == "bytetrack":
                        results = self.deduplicate_tracker_results(log_parser.found_log_entries)
                    # Deduplicate results based just on object classes and counts
                    elif self.app.tracker_combo.currentText() == "-":
                        results = self.deduplicate_yolo_results(log_parser.found_log_entries)
                else:
                    results = log_parser.found_log_entries
                
            elif self.app.tracker_combo.currentText() == "xclip":
                xclip_parser = XClipParser(
                    video_path=self.app.video_path,
                    query=self.query,
                    output_dir=self.app.output_dir,
                )
                similarities, metadata = xclip_parser.search_embeddings(top_k=5)
                results = xclip_parser.top_frames
                
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))

    def deduplicate_tracker_results(self, results):
        """Deduplicate results based on tracked object IDs and classes"""
        unique_results = {}
        previous_object_set = None
        
        # Sort frames by timestamp/frame number to process in order
        sorted_items = sorted(results.items(), 
                              key=lambda x: x[1].get('frame_number', 0) 
                                  if isinstance(x[1], dict) else 0)
        
        for frame_path, metadata in sorted_items:
            # Extract object IDs and classes from metadata
            current_objects = set()
            
            if 'detections' in metadata:
                for detection in metadata['detections']:
                    # If we have tracking IDs
                    if 'id' in detection and 'class' in detection:
                        # Using tuple of (id, class) as the identifier
                        current_objects.add((detection['id'], detection['class']))
                    # Fall back to just using class and bounding box if no tracking ID
                    elif 'class' in detection and 'bbox' in detection:
                        bbox = tuple(detection['bbox'])  # Convert list to tuple for hashability
                        current_objects.add((detection['class'], bbox))
            
            # Only include this frame if the object set differs from previous
            if previous_object_set is None or current_objects != previous_object_set:
                unique_results[frame_path] = metadata
                previous_object_set = current_objects
                
        return unique_results
    
    def deduplicate_yolo_results(self, results):
        """Deduplicate results based just on object classes and counts"""
        unique_results = {}
        previous_class_counts = None
        
        # Sort frames by timestamp/frame number
        sorted_items = sorted(results.items(), 
                              key=lambda x: x[1].get('frame_number', 0) 
                                  if isinstance(x[1], dict) else 0)
        
        for frame_path, metadata in sorted_items:
            # Count occurrences of each class
            current_class_counts = {}
            
            if 'detections' in metadata:
                for detection in metadata['detections']:
                    if 'class' in detection:
                        cls = detection['class']
                        current_class_counts[cls] = current_class_counts.get(cls, 0) + 1
            
            # Only include this frame if the class counts differ from previous
            if previous_class_counts is None or current_class_counts != previous_class_counts:
                unique_results[frame_path] = metadata
                previous_class_counts = current_class_counts
                
        return unique_results