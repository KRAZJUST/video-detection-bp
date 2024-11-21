import os
import json
import cv2
import shutil
import sqlite3
from .query_parser import QueryParser
from database.database import Database

class DetectionParser:
    def __init__(self, log_entries, query, output_dir, tracker: str, database_path: str):
        self.log_entries = log_entries
        self.database_path = database_path
        self.db = Database(self.database_path)
        self.tracker = tracker
        self.connection = None
        self.query_parser = QueryParser(query)
        self.found_log_entries = {}

        self.output_dir = output_dir
        # Create a directory to save the found frames
        self.found_dir = os.path.join(output_dir, 'found_frames')
        os.makedirs(self.found_dir, exist_ok=True)

        self.filter_colors = []
        self.filter_objects = []
        self.filter_logic = []
        self.filter_directions = []
 
    def parse_detections(self):
        """
        Parse detections in the database and filter based on the query.
        Annotate and save frames without duplicating saves for multiple detections.
        """

        # Parse the query and extract the filter conditions
        parsed_elements = self.query_parser.get_parsed_queries()
        for element in parsed_elements:
            for label, value in element.items():
                if label == 'color':
                    self.filter_colors.append(value)
                elif label == 'object':
                    if value == 'vehicle':
                        self.filter_objects.extend(['car', 'truck', 'bus'])
                    else:
                        self.filter_objects.append(value)
                elif label == 'logic':
                    self.filter_logic.append(value)
                elif label == 'direction':
                    self.filter_directions.append(value)

                
        # If no colors are specified, set it to None
        if len(self.filter_colors) == 0:
            self.filter_colors = None
        if len(self.filter_directions) == 0:
            self.filter_directions = None
        if len(self.filter_logic) == 0:
            self.filter_logic = None

        # Reset found log entries and directories
        if os.path.exists(self.found_dir):
            shutil.rmtree(self.found_dir)
        os.makedirs(self.found_dir, exist_ok=True)
        self.found_log_entries = {}

        print(f"Filtering by objects: {self.filter_objects}")
        print(f"Filtering by colors: {self.filter_colors}")

        # Fetch either initial detections or refined detections based on the tracker
        if self.tracker == '-':
            frames = self.db.get_frames_with_detections('detections', self.filter_objects, self.filter_colors, self.filter_directions)
        elif self.tracker == 'bytetrack':
            frames = self.db.get_frames_with_detections('refined_detections', self.filter_objects, self.filter_colors, self.filter_directions)

        if self.filter_logic and self.filter_logic[0] == 'and':
            print(f"Filtering by logic: {self.filter_logic}")
            frames = self.filter_detections_in_frames(frames, logic_operator=self.filter_logic[0], expected_conditions=parsed_elements)

        # Loop through each frame and its detections
        for frame_number, frame_data in frames.items():
            detections = frame_data['detections']
            self.found_log_entries[frame_number] = detections
            self.save_frame(frame_number, detections)

        self.save_found_log()

    def filter_detections_in_frames(self, frames, logic_operator=None, expected_conditions=None):
        """
        Filter frames based on detection logic (and/or).
        Args:
            frames: Dictionary of frames and their detections.
            logic_operator: 'and' or 'or' for filtering logic.
            expected_conditions: List of conditions to match (e.g., [{'object': 'car', 'color': 'red'}]).
        """

        filtered_frames = {}
        print(f"Filtering frames based on logic: {logic_operator} and conditions: {expected_conditions}")

        # Handle 'vehicle' expansion to specific vehicle types
        for condition in expected_conditions:
            if 'object' in condition and condition['object'] == 'vehicle':
                expected_conditions.remove(condition)
                expected_conditions.extend([{'object': 'car'}, {'object': 'truck'}, {'object': 'bus'}])
            if 'logic' in condition:
                expected_conditions.remove(condition)

        # Loop through each frame
        for frame_number, frame_data in frames.items():
            detections = frame_data['detections']
            
            # Check conditions based on logic
            if logic_operator == "and":
                # For AND, we need at least one match for each condition
                matched_conditions = set()
                for condition in expected_conditions:
                    for detection in detections:
                        if self.check_detection_condition(detection, condition):
                            # Add the matched condition to the set
                            matched_conditions.add(frozenset(condition.items()))
                            # Once condition is matched, no need to check the same detection again
                            break
                # If all conditions are matched, keep the frame
                if len(matched_conditions) == len(expected_conditions):
                    filtered_frames[frame_number] = frame_data

        return filtered_frames

    def check_detection_condition(self, detection, condition):
        """
        Helper function to check if a detection meets a specific condition.
        Args:
            detection: Detection dictionary (e.g., {'class_name': 'car', 'dominant_color': 'yellow'}).
            condition: Condition dictionary (e.g., {'object': 'car', 'color': 'yellow'}).
        """
        if 'object' in condition and detection['class_name'] != condition['object']:
            return False
        if 'color' in condition and detection['dominant_color'] != condition['color']:
            return False
        if 'direction' in condition and detection['direction'] != condition['direction']:
            return False
        return True



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

    def annotate_image(self, frame_file_path, detections, frame_num):
        """
        Annotate the image with bounding boxes and labels for detections.
        The frame is saved only once even if it has multiple detections.
        """
        # Read the image
        image = cv2.imread(frame_file_path)
        if image is None:
            print(f"Failed to read image: {frame_file_path}")
            return

        # Annotate the detections
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            dominant_color = detection.get('dominant_color', 'unknown')

            xmin, ymin, xmax, ymax = bbox
            color = (255, 69, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw the label text
            label = f"{class_name} ({dominant_color}, {confidence:.2f})"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the annotated image if it hasn't been saved already
        output_path = os.path.join(self.found_dir, f"frame_{frame_num:04d}_annotated.jpg")
        if not os.path.exists(output_path):
            cv2.imwrite(output_path, image)
            print(f"Annotated frame saved to '{output_path}'.")

