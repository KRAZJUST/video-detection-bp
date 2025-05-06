# =============================================================================
# File: query_worker.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module provides a worker thread for querying video frames based on
# user input query. It handles different types of queries, including object
# detection and embedding-based searches. The results are emitted back to
# the main thread for further processing. It also includes methods for
# deduplicating result frames.
#
# =============================================================================

from PyQt6.QtCore import QThread, pyqtSignal
from parsers.detection_parser import DetectionParser
from xclip.xclip_parser import XClipParser
from siglip.siglip_parser import SigLIPParser

class QueryWorker(QThread):
    finished = pyqtSignal(dict, list)
    error = pyqtSignal(str)
    feedback = pyqtSignal(str)

    def __init__(self, app, query, area_of_interest=None):
        """
        Initialize the QueryWorker with the application instance, query string,
        and optional area of interest.

        Args:
            app: Application instance
            query: Query string to search for in the video
            area_of_interest: Optional area of interest for object detection
        """
        super().__init__()
        self.app = app
        self.query = query
        self.deduplicate = True
        self.area_of_interest = area_of_interest

    def run(self):
        """
        Run the query worker thread. This method is executed in a separate thread
        when the worker is started. It handles the query processing and emits
        results or errors back to the main thread.
        """
        try:
            # Feedback fallback that emits message
            def feedback_callback(message):
                self.feedback.emit(message)

            if self.app.tracker_combo.currentText() in ["yolo", "bytetrack"]:
                log_parser = DetectionParser(
                    query=self.query,
                    input_video=self.app.video_path,
                    output_dir=self.app.output_dir,
                    database_path=self.app.database_path,
                    tracker=self.app.tracker_combo.currentText(),
                    use_segmentation=self.app.use_segmentation_value,
                    area_of_interest=self.area_of_interest,
                    feedback_callback=feedback_callback,
                )
                log_parser.parse_detections()

                if self.deduplicate:
                    # Deduplicate results based on tracked object IDs and classes
                    if self.app.tracker_combo.currentText() == "bytetrack":
                        results = self.deduplicate_tracker_results(log_parser.found_log_entries)
                    # Deduplicate results based just on object classes and counts
                    elif self.app.tracker_combo.currentText() == "yolo":
                        results = self.deduplicate_yolo_results(log_parser.found_log_entries)
                else:
                    results = log_parser.found_log_entries

                # empty metadata for YOLO and ByteTrack
                metadata = {}
                
            elif self.app.tracker_combo.currentText() in ["xclip-32"]:
                xclip_parser = XClipParser(
                    video_path=self.app.video_path,
                    query=self.query,
                    output_dir=self.app.output_dir,
                    model_name=self.app.tracker_combo.currentText()
                )
                similarities, metadata = xclip_parser.search_embeddings(top_k=self.app.results_batch_count)
                results = xclip_parser.top_frames
            
            elif self.app.tracker_combo.currentText() == "siglip":
                siglip_parser = SigLIPParser(
                    video_path=self.app.video_path,
                    query=self.query,
                    output_dir=self.app.output_dir,
                    model_name=self.app.tracker_combo.currentText()
                )
                similarities, metadata = siglip_parser.search_embeddings(n_results=self.app.results_frames_count)
                results = siglip_parser.top_frames
                
            self.finished.emit(results, metadata)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def deduplicate_tracker_results(self, results, max_frames_interval=15):
        """
        Deduplicate results based on tracked object IDs and classes, but keep 
        frames with new objects

        Args:
            results (dict): Dictionary of frame paths and detection lists
            max_frames_interval (int): Maximum number of frames to skip before forcing inclusion
        Returns:
            dict: Dictionary of unique frames with detections
        """
        unique_results = {}
        # Objects seen so far, stored as (track_id, class_name) tuples
        seen_objects = set()
        # Keep track of the last frame added to results
        frames_since_last_added = 0

        # Sort frames by timestamp/frame number to process in order
        sorted_items = sorted(results.items(), 
                            key=lambda x: int(x[0].split('_')[-1].split('.')[0]) 
                                if isinstance(x[0], str) else 0)
        
        for frame_path, detection_list in sorted_items:
            new_objects_found = False
            print(detection_list)
            frames_since_last_added += 1
            
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
                    
            # If no frame has been added for a while, force inclusion of this frame
            if frames_since_last_added >= max_frames_interval:
                new_objects_found = True
                print(f"Max interval reached, adding frame {frame_path}")

            # Include this frame if it contains any new objects
            if new_objects_found:
                unique_results[frame_path] = detection_list
                frames_since_last_added = 0
                print(f"Added frame {frame_path} to results")
        
        print(f"Total unique frames: {len(unique_results)}")
        return unique_results

    def deduplicate_yolo_results(self, results, min_count_change=1, max_frames_interval=8, min_confidence=0.1):
        """
        Deduplicate results based on object classes, counts, and color names.
        Filter out frames that don't contain significant changes, but keep important frames.
        
        Args:
            results (dict): Dictionary of frame paths and detection lists
            min_count_change (int): Minimum count change to consider a frame significant
            max_frames_interval (int): Maximum number of frames to skip before forcing inclusion
            min_confidence (float): Minimum confidence score to consider a detection valid
            
        Returns:
            dict: Dictionary of unique frames with detections
        """
        unique_results = {}
        previous_class_counts = {}
        previous_color_distribution = {}  # Track color distributions by class
        last_frame_added = -1
        frames_since_last_add = 0
        
        sorted_items = sorted(results.items(),
                            key=lambda x: int(x[0].split('_')[-1].split('.')[0])
                            if isinstance(x[0], str) else 0)
        
        for i, (frame_path, detection_list) in enumerate(sorted_items):
            frame_num = int(frame_path.split('_')[-1].split('.')[0]) if isinstance(frame_path, str) else i
            current_class_counts = {}
            current_color_distribution = {}
            
            # Filter detections by confidence threshold
            valid_detections = [d for d in detection_list if d.get('confidence', 1.0) >= min_confidence]
            
            # Process all detections in current frame
            for detection in valid_detections:
                if 'class_name' in detection:
                    class_name = detection['class_name']
                    # Update class counts
                    current_class_counts[class_name] = current_class_counts.get(class_name, 0) + 1
                    
                    # Track color names if available
                    if 'dominant_color' in detection:
                        color_name = detection['dominant_color']
                        if isinstance(color_name, str):
                            if class_name not in current_color_distribution:
                                current_color_distribution[class_name] = {}
                            
                            # Update color count for this class
                            current_color_distribution[class_name][color_name] = current_color_distribution[class_name].get(color_name, 0) + 1
            
            # Check for significant changes in the frame compared to previous frames
            keep_frame = False
            reason = ""
            frames_since_last_add += 1
            
            # Always keep the first frame
            if not unique_results:
                keep_frame = True
                reason = "first frame"
            
            # Check for new classes
            elif set(current_class_counts.keys()) != set(previous_class_counts.keys()):
                keep_frame = True
                reason = "new classes detected"
                
            # Check for significant count changes with lower threshold as frames pass
            elif any(abs(current_class_counts.get(cls, 0) - previous_class_counts.get(cls, 0)) >= 
                    max(1, min_count_change - frames_since_last_add // 5)  # Gradually reduce threshold
                    for cls in set(current_class_counts) | set(previous_class_counts)):
                keep_frame = True
                reason = "object count changed"
                
            # Check for significant color distribution changes
            elif self._check_color_distribution_change(current_color_distribution, previous_color_distribution):
                keep_frame = True
                reason = "color distribution changed"
                
            # Force inclusion of a frame if it's been too long since the last one
            elif frames_since_last_add >= max(3, max_frames_interval - frames_since_last_add // 2):
                keep_frame = True
                reason = f"max interval reached ({frames_since_last_add} frames)"
                
            # Keep the frame if any criteria was met
            if keep_frame:
                unique_results[frame_path] = detection_list
                previous_class_counts = current_class_counts
                previous_color_distribution = current_color_distribution
                last_frame_added = frame_num
                frames_since_last_add = 0
                
        print(f"Total unique frames: {len(unique_results)} out of {len(results)}")
        return unique_results

    def _check_color_distribution_change(self, current_colors, previous_colors):
        """
        Check if there's a significant change in color distribution for any object class.
        
        Args:
            current_colors (dict): Dictionary of class_name -> dict of color_name -> count
            previous_colors (dict): Dictionary of class_name -> dict of color_name -> count
            
        Returns:
            bool: True if there's a significant color change, False otherwise
        """
        # no previous colors to compare with
        if not previous_colors:
            return False
            
        # Check classes that exist in both frames
        for class_name in set(current_colors.keys()) & set(previous_colors.keys()):
            current_dist = current_colors[class_name]
            previous_dist = previous_colors[class_name]
            
            # If we have new colors appearing
            if set(current_dist.keys()) != set(previous_dist.keys()):
                return True
                
            # If the distribution of colors has changed significantly
            for color in set(current_dist.keys()) & set(previous_dist.keys()):
                # If any color's count changes by more than 25% of the total objects
                curr_count = current_dist[color]
                prev_count = previous_dist[color]
                
                curr_total = sum(current_dist.values())
                prev_total = sum(previous_dist.values())
                
                # Calculate percentage change
                curr_pct = curr_count / curr_total if curr_total > 0 else 0
                prev_pct = prev_count / prev_total if prev_total > 0 else 0
                
                # 25% threshold for percentage change
                if abs(curr_pct - prev_pct) > 0.25:
                    return True
                    
        return False