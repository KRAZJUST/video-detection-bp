from PyQt6.QtCore import QThread, pyqtSignal
from parsers.detection_parser import DetectionParser
from xclip.xclip_parser import XClipParser

class QueryWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, app, query, area_of_interest=None):
        super().__init__()
        self.app = app
        self.query = query
        self.deduplicate = True
        self.area_of_interest = area_of_interest

    def run(self):
        try:
            if self.app.tracker_combo.currentText() in ["-", "bytetrack"]:
                log_parser = DetectionParser(
                    query=self.query,
                    output_dir=self.app.output_dir,
                    database_path=self.app.database_path,
                    tracker=self.app.tracker_combo.currentText(),
                    use_segmentation=self.app.use_segmentation.isChecked(),
                    area_of_interest=self.area_of_interest,
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
                
            elif self.app.tracker_combo.currentText() == "xclip-32" or self.app.tracker_combo.currentText() == "xclip-16":
                xclip_parser = XClipParser(
                    video_path=self.app.video_path,
                    query=self.query,
                    output_dir=self.app.output_dir,
                )
                similarities, metadata = xclip_parser.search_embeddings(top_k=5)
                results = xclip_parser.top_frames
                print(type(results))
                print(f"Top frames: {results}")
                
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

    def deduplicate_yolo_results(self, results, min_count_change=1, max_frames_interval=30):
        """
        Deduplicate results based on object classes and counts.

        Filter out frames that don't contain any new classes or significant count changes.
        Keep only the frames that contain new objects or have significant count changes compared to previous frames.

        Args:
            results (dict): Dictionary of frame paths and detection lists
            min_count_change (int): Minimum count change to consider a frame significant
            max_frames_interval (int): Maximum number of frames to include between significant frames

        Returns:
            dict: Dictionary of unique frames with detections
        """
        unique_results = {}
        previous_class_counts = {}
        last_frame_added = -1
        
        sorted_items = sorted(results.items(),
                            key=lambda x: int(x[0].split('_')[-1].split('.')[0])
                            if isinstance(x[0], str) else 0)
        
        for i, (frame_path, detection_list) in enumerate(sorted_items):
            frame_num = int(frame_path.split('_')[-1].split('.')[0]) if isinstance(frame_path, str) else i
            current_class_counts = {}
            
            # Process all detections in current frame
            for detection in detection_list:
                if 'class_name' in detection:
                    class_name = detection['class_name']
                    
                    # Update class counts
                    current_class_counts[class_name] = current_class_counts.get(class_name, 0) + 1
                    
            # Check for significant changes in the frame compared to previous frames to decide if we should keep it
            keep_frame = False
            reason = ""
            
            # Always keep the first frame
            if not unique_results:
                keep_frame = True
                reason = "first frame"
            
            # Check for new classes
            elif set(current_class_counts.keys()) != set(previous_class_counts.keys()):
                keep_frame = True
                reason = "new classes detected"
            
            # Check for significant count changes
            elif any(abs(current_class_counts.get(cls, 0) - previous_class_counts.get(cls, 0)) >= min_count_change 
                    for cls in set(current_class_counts) | set(previous_class_counts)):
                keep_frame = True
                reason = "object count changed"
             
            # Force inclusion of a frame if it's been too long since the last one
            elif last_frame_added >= 0 and (frame_num - last_frame_added) >= max_frames_interval:
                keep_frame = True
                reason = "max interval reached"
            
            # Keep the frame if any criteria was met
            if keep_frame:
                unique_results[frame_path] = detection_list
                previous_class_counts = current_class_counts
                last_frame_added = frame_num
                print(f"Added frame {frame_path} - reason: {reason}")

            print(f"Frame {frame_path} - counts: {current_class_counts},")
        
        print(f"Total unique frames: {len(unique_results)}")
        return unique_results