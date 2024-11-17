import os
import json
import cv2
import shutil
from .query_parser import QueryParser
from .query_matcher import QueryMatcher

class DetectionParser:
    def __init__(self, log_entries, query, output_dir):
        self.log_entries = log_entries
        self.query_parser = QueryParser(query)
        self.query_matcher = QueryMatcher(self.query_parser.get_parsed_queries())
        self.found_log_entries = {}

        self.output_dir = output_dir
        # Create a directory to save the found frames
        self.found_dir = os.path.join(output_dir, 'found_frames')
        os.makedirs(self.found_dir, exist_ok=True)

        self.filter_colors = []
        self.filter_objects = []
 
    def parse_log_entries(self):
        """
        Parse log entries and filter based on the query.
        """

        # Reset found log entries and directories
        if os.path.exists(self.found_dir):
            shutil.rmtree(self.found_dir)
        os.makedirs(self.found_dir, exist_ok=True)
        self.found_log_entries = {}
                
        for frame_num, detections in self.log_entries.items():
            # Check if the frame's detections match the query
            if self.query_matcher.check_query(detections):
                self.found_log_entries[frame_num] = detections
                self.save_frame(frame_num, detections)
        self.save_found_log()

        # Get the parsed colors and objects from the query
        self.filter_colors = self.query_matcher.parsed_colors
        print(f"Filter colors: {self.filter_colors}")
        self.filter_objects = self.query_matcher.parsed_objects
        print(f"Filter objects: {self.filter_objects}")

    def save_frame(self, frame_num, detections):
        """
        Save annotated frame with detections.
        """
        frame_file_name = f'frame_{frame_num:04d}.jpg'
        frame_file_path = os.path.join(self.output_dir, 'extracted_frames', frame_file_name)

        if os.path.exists(frame_file_path):
            self.annotate_image(frame_file_path, detections, frame_num)
        else:
            print(f"Warning: Frame file {frame_file_name} does not exist in {self.output_dir}")

    def save_found_log(self):
        """Save the filtered log entries to a new JSON file."""
        found_log_path = os.path.join(self.output_dir, 'found_log.json')
        with open(found_log_path, 'w') as found_log_file:
            json.dump(self.found_log_entries, found_log_file, indent=4)
        print(f"Found log saved to '{found_log_path}'.")

    def matches_condition(self, detection):
        """
        Check if a detection matches the parsed color and object filters.
        """
        class_name = detection['class_name'].lower()
        dominant_color = detection['dominant_color'].lower()

        return class_name in self.filter_objects and dominant_color in self.filter_colors


    def annotate_image(self, frame_file_path, detections, frame_num):
        """
        Annotate the image with bounding boxes and labels for detections that meet query conditions.
        """

        # Read the image
        image = cv2.imread(frame_file_path)
        if image is None:
            print(f"Failed to read image: {frame_file_path}")
            return

        # Filter detections
        matching_detections = [
            detection for detection in detections if self.matches_condition(detection)
        ]

        # Annotate only the detections that matched
        for detection in matching_detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            dominant_color = detection.get('dominant_color', 'unknown')

            xmin, ymin, xmax, ymax = bbox
            color = (0, 255, 0)  # Green bounding box for matched detections
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw the label text
            label = f"{class_name} ({dominant_color}, {confidence:.2f})"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the annotated image
        output_path = os.path.join(self.found_dir, f"frame_{frame_num:04d}_annotated.jpg")
        cv2.imwrite(output_path, image)
