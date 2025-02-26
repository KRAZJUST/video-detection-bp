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
        """Deduplicate results based on tracked object IDs and classes, but keep frames with new objects"""
        unique_results = {}
        # Objects seen so far, stored as (track_id, class_name) tuples
        seen_objects = set() 

        # Sort frames by timestamp/frame number to process in order
        sorted_items = sorted(results.items(), 
                            key=lambda x: int(x[0].split('_')[-1].split('.')[0]) 
                                if isinstance(x[0], str) else 0)
        
        for frame_path, detection_list in sorted_items:
            new_objects_found = False
            print(detection_list)
            
            # Go through each detection in the frame
            for detection in detection_list:
                # Check if this detection has a track ID and class name
                if 'track_id' in detection and 'class_name' in detection:
                    # Create a unique identifier for this object
                    obj_identifier = (detection['track_id'], detection['class_name'])
                    
                    # Check if this is a new object we haven't seen before
                    if obj_identifier not in seen_objects:
                        new_objects_found = True
                        seen_objects.add(obj_identifier)
                        print(f"New object found: {obj_identifier} in frame {frame_path}")
                    
            # Include this frame if it contains any new objects
            if new_objects_found:
                unique_results[frame_path] = detection_list
                print(f"Added frame {frame_path} to results")
        
        print(f"Total unique frames: {len(unique_results)}")
        return unique_results

    def deduplicate_yolo_results(self, results):
        """Deduplicate results based on object classes, keeping frames with new classes
        TODO: Implement a more sophisticated deduplication strategy for YOLO results
        """
        unique_results = {}
        # Classes seen so far
        seen_classes = set()
        
        # Sort frames by frame number if possible
        sorted_items = sorted(results.items(), 
                            key=lambda x: int(x[0].split('_')[-1].split('.')[0]) 
                                if isinstance(x[0], str) else 0)
        
        for frame_path, detection_list in sorted_items:
            new_classes_found = False
            
            for detection in detection_list:
                if 'class_name' in detection:
                    class_name = detection['class_name']
                    
                    # Check if this is a new class we haven't seen before
                    if class_name not in seen_classes:
                        new_classes_found = True
                        seen_classes.add(class_name)
                        print(f"New class found: {class_name} in frame {frame_path}")
            
            # Include this frame if it contains any new classes
            if new_classes_found:
                unique_results[frame_path] = detection_list
                print(f"Added frame {frame_path} to results")
        
        print(f"Total unique frames: {len(unique_results)}")
        return unique_results