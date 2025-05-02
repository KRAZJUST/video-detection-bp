# =============================================================================
# File: detection_parser.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module is responsible for parsing the detection results from the database
# and filtering them based on the query. It then saves the annotated frames found 
# based on the query.
#
# =============================================================================

import os
import cv2
import shutil
from typing import Any
from .query_parser import QueryParser
from database.sqlite_database import Database
from constants.constants import COLOR_MAP
from .yolo_segmenter import YOLOSegmenter
from profiling_utils.profiling_utils import detailed_profile


class DetectionParser:
    def __init__(self, 
                 query: str,
                 input_video: str,
                 output_dir: str, 
                 tracker: str, 
                 database_path: str, 
                 use_segmentation: bool = True,
                 frame_width: int = 640, 
                 frame_height: int = 360, 
                 area_of_interest: tuple = None,
                 feedback_callback: Any = None):
        self.db = Database(database_path)
        self.tracker = tracker
        self.use_segmentation = use_segmentation
        self.query_parser = QueryParser(query)
        self.segmenter = YOLOSegmenter()
        self.found_log_entries = {}
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.area_of_interest = area_of_interest

        self.output_dir = output_dir
        self.input_video = input_video
        self.feedback_callback = feedback_callback
        
        # Create or clear the found frames directory
        self.found_dir = os.path.join(output_dir, 'found_frames')
        if not os.path.exists(self.found_dir):
            os.makedirs(self.found_dir, exist_ok=True)
        else:
            shutil.rmtree(self.found_dir)
            os.makedirs(self.found_dir, exist_ok=True)

        self.filter_colors = []
        self.filter_objects = []
        self.filter_logic = []
        self.filter_directions = []
        self.filter_interactions = []
        self.filter_quadrants = []
        self.unknown_conditions = []
 
    def calculate_interaction(self, bbox1, bbox2) -> str:
        """
        Calculate the interaction between two bounding boxes.
        
        Parameters:
            bbox1 (tuple): Bounding box coordinates (x1, y1, x2, y2)
            bbox2 (tuple): Bounding box coordinates (x1, y1, x2, y2)
        
        Returns:
            - 'overlapping' if the bounding boxes overlap
            - 'touching' if the bounding boxes are touching
            - 'near' if the bounding boxes are near each other
            - 'far' if the bounding boxes are far apart
            
        TODO: Implement the interaction calculation based on the bounding box coordinates, make sure to consider the distance between the centers of the bounding boxes
              and make the thresholds configurable.
              REFACTOR THIS!
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate centers
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        # Calculate distance between centers
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        # Check for overlap
        if (x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2):
            return "overlapping"
        
        # Check for touching (within a threshold)
        threshold = 20
        if distance < threshold:
            return "touching"
            
        # Check for near (within a larger threshold)
        if distance < 100:
            return "near"
            
        return "far"

    def get_frame_quadrant(self, bbox, frame_size) -> str:
        """
        Determine which quadrant(s) the object is in.
        
        Parameters:
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
            frame_size (tuple): Frame size (width, height)
            
        Returns:
            str: Quadrant (top-left, top-right, bottom-left, bottom-right)
        """
        x1, y1, x2, y2 = bbox
        frame_width, frame_height = frame_size
        
        # Calculate center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine quadrant
        if center_x < frame_width/2:
            if center_y < frame_height/2:
                return "top-left"
            else:
                return "bottom-left"
        else:
            if center_y < frame_height/2:
                return "top-right"
            else:
                return "bottom-right"
 
    @detailed_profile
    def parse_detections(self):
        """
        Parse detections in the database and filter based on the query.
        Annotate and save frames without duplicating saves for multiple detections.
        
        Returns:
            None
        """

        # Parse the query and extract the filter conditions
        parsed_elements = self.query_parser.get_parsed_queries()
        for element in parsed_elements:
            for label, value in element.items():
                if label == 'color':
                    self.filter_colors.append(value)
                elif label == 'object':
                    # If the object is 'truck' or 'bus', add both to the filter list because the model may not distinguish them correctly a lot of the time
                    #if value == 'truck':
                    #    self.filter_objects.extend(['truck', 'bus'])
                    #elif value == 'bus':
                    #    self.filter_objects.extend(['bus', 'truck'])
                    #else:
                    self.filter_objects.append(value)
                elif label == 'logic':
                    self.filter_logic.append(value)
                elif label == 'direction':
                    self.filter_directions.append(value)
                elif label == 'interaction':
                    self.filter_interactions.append(value)
                elif label == 'quadrant':
                    self.filter_quadrants.append(value)

        print(f"Unknown conditions: {self.unknown_conditions}")
        # If multiple objects or colors are provided and interaction filter is set, add 'and' logic by default
        if (self.filter_interactions or self.filter_quadrants) \
                and (len(self.filter_objects) > 1 or len(self.filter_colors) > 1) \
                and not self.filter_logic:
                    self.filter_logic.append('and')
                    
        # If no filters for a group are provided set them to None
        if len(self.filter_colors) == 0:
            self.filter_colors = None
        if len(self.filter_directions) == 0:
            self.filter_directions = None
        if len(self.filter_logic) == 0:
            self.filter_logic = None
        if len(self.filter_interactions) == 0:
            self.filter_interactions = None
        if len(self.filter_quadrants) == 0:
            self.filter_quadrants = None

        # Reset found log entries and directories
        if os.path.exists(self.found_dir):
            shutil.rmtree(self.found_dir)
        os.makedirs(self.found_dir, exist_ok=True)
        self.found_log_entries = {}

        print(f"Filtering by objects: {self.filter_objects}")
        print(f"Filtering by colors: {self.filter_colors}")
    
        # Get initial frames based on basic filters (objects/colors/directions), choose detections or refined detections based on the level of processing
        if self.tracker == 'yolo':
            frames = self.db.get_yolo_frames_with_detections(
                self.input_video,
                self.filter_objects, 
                self.filter_colors, 
                self.filter_directions,
                self.area_of_interest
            )
        elif self.tracker == 'bytetrack':
            frames = self.db.get_bytetrack_frames_with_detections(
                self.input_video,
                self.filter_objects, 
                self.filter_colors, 
                self.filter_directions,
                self.area_of_interest
            )
        print(f"Number of frames with detections: {len(frames)}")
        
        # Apply spatial filters (interactions and quadrants) if specified, apply interactions only if there are multiple objects
        # to prevent unnecessary calculations and errors if the query is not parsed correctly
        if (self.filter_interactions and len(self.filter_objects) > 1) or self.filter_quadrants:
            print("Applying spatial filters")
            frames = self.apply_spatial_filters(frames)
        print(f"Number of frames after spatial filters: {len(frames)}")
        
        # Filter frames based on logic (and/or) and expected conditions
        if self.filter_logic and self.filter_logic[0] == 'and':
            print(f"Filtering by logic: {self.filter_logic}")
            frames = self.filter_detections_in_frames(frames, logic_operator=self.filter_logic[0], expected_conditions=parsed_elements)
        print(f"Number of frames after logic filter: {len(frames)}")

        # Build a list of words used for filtering to display in the feedback
        # callaback in GUI
        words_used_for_filtering = []
        if self.filter_objects:
            words_used_for_filtering.extend(self.filter_objects)
        if self.filter_colors:
            words_used_for_filtering.extend(self.filter_colors)
        if self.filter_directions:
            words_used_for_filtering.extend(self.filter_directions)
        if self.filter_interactions:
            words_used_for_filtering.extend(self.filter_interactions)
        if self.filter_quadrants:
            words_used_for_filtering.extend(self.filter_quadrants)
        if self.filter_logic:
            words_used_for_filtering.extend(self.filter_logic)
        self.feedback_callback(f"Words used for filtering: {', '.join(words_used_for_filtering)}")
        
        ### Save the filtered frames ###
        # Clear the found frames directory
        if os.path.exists(self.found_dir):
            shutil.rmtree(self.found_dir)
        os.makedirs(self.found_dir, exist_ok=True)

        # Determine the directory for saving frames based on the tracker type
        if self.tracker == 'bytetrack':
            frame_file_dir = os.path.join(self.output_dir, 'extracted_frames_b')
        else:
            frame_file_dir = os.path.join(self.output_dir, 'extracted_frames_yx')

        # Loop through each frame and its detections
        for frame_number, frame_data in frames.items():
            detections = frame_data['detections']
            self.found_log_entries[frame_number] = detections
            self.save_frame(frame_number, detections, frame_file_dir)

        #self.save_found_log()

    def apply_spatial_filters(self, frames) -> dict:
        """
        Apply spatial filters (interactions and quadrants) to frames.
        First, filter frames based on quadrants if specified.
        Then, filter frames based on interactions if specified by checking pairs of objects that should interact.
        Considers both object class and color when specified in queries.
        
        Args:
            frames (dict): Frames and their detections.
        Returns:
            dict: Filtered frames based on spatial filters.
        """
        filtered_frames = {}
        for frame_number, frame_data in frames.items():
            detections = frame_data['detections']
            
            # Quadrant filtering
            if self.filter_quadrants:
                quadrant_match = any(
                    self.get_frame_quadrant(detection['bbox'], (self.frame_width, self.frame_height)) in self.filter_quadrants
                    for detection in detections
                )
                if not quadrant_match:
                    continue
            
            # Interaction filtering
            if self.filter_interactions:
                # Group detections by class name AND color when color is specified
                class_color_detections = {}
                for detection in detections:
                    class_name = detection['class_name']
                    
                    # Only process objects that are in the filter list
                    if class_name in self.filter_objects:
                        # Check if color is available and consider it in grouping
                        color = None
                        if 'dominant_color' in detection:
                            color = detection['dominant_color']
                        
                        # Get expected color for this object from query
                        expected_color = self.get_color_for_object(class_name)
                        
                        # Create key that includes both class and color (if specified)
                        if expected_color and color:
                            key = f"{class_name}_{color}"
                        else:
                            key = class_name
                        
                        # Group by class+color or just class
                        class_color_detections.setdefault(key, []).append(detection)
                
                # Find interactions based on query requirements
                if not self._check_color_aware_interactions(class_color_detections):
                    continue
            
            # If frame passed all filters, add to results
            filtered_frames[frame_number] = frame_data
        
        return filtered_frames

    def _check_color_aware_interactions(self, class_color_detections):
        """
        Helper method to check if there are required interactions between objects,
        considering both class and color when specified.
        
        Args:
            class_color_detections (dict): Dictionary of class_color keys to lists of detections.
        Returns:
            bool: True if required interactions are found.
        """
        # Get all unique base classes (without color)
        base_classes = set()
        for key in class_color_detections.keys():
            base_class = key.split('_')[0]  # Extract base class from key
            base_classes.add(base_class)
        
        # If more than one unique class is specified, focus on the interactions between them
        if len(base_classes) > 1:
            # Look for interactions between different base classes
            for key1, detections1 in class_color_detections.items():
                for key2, detections2 in class_color_detections.items():
                    # Skip if same key (same class+color combination)
                    if key1 == key2:
                        continue
                    
                    # Extract base classes to check if they're different
                    base_class1 = key1.split('_')[0]
                    base_class2 = key2.split('_')[0]
                    
                    # Check if these are the same class but different colors
                    same_class_diff_color = base_class1 == base_class2 and key1 != key2
                    
                    # Either different classes or same class with different colors
                    if base_class1 != base_class2 or same_class_diff_color:
                        # Check interactions between these groups
                        if self._find_interactions_between_groups(detections1, detections2):
                            return True
        else:
            # If only one base class is specified, check interactions within that class
            # but only between different colors if colors are specified
            keys = list(class_color_detections.keys())
            
            # Multiple color variants of the same class
            if len(keys) > 1:
                # Check interactions between different color variants
                for i in range(len(keys)):
                    for j in range(i+1, len(keys)):
                        key1, key2 = keys[i], keys[j]
                        if self._find_interactions_between_groups(
                            class_color_detections[key1], 
                            class_color_detections[key2]
                        ):
                            return True
            else:
                # If only one class+color or just one class, check interactions within that group
                for detections_list in class_color_detections.values():
                    if len(detections_list) >= 2 and self._find_interactions_within_group(detections_list):
                        return True
        
        return False

    def _extract_query_color_objects(self):
        """
        Extract object-color pairs from the query.
        Returns a dictionary mapping objects to their specified colors.
        """
        color_objects = {}
        for element in self.query_parser.get_parsed_queries():
            if 'object' in element and 'color' in element:
                obj_class = element['object']
                color = element['color']
                color_objects[obj_class] = color
        return color_objects

    def _find_interactions_between_groups(self, group1, group2):
        """
        Find interactions between two groups of detections.
        Returns True on first matching interaction.
        """
        for det1 in group1:
            for det2 in group2:
                interaction_type = self.calculate_interaction(det1['bbox'], det2['bbox'])
                if interaction_type in self.filter_interactions:
                    return True
        return False

    def _find_interactions_within_group(self, detections):
        """
        Find interactions within the same group of detections.
        Returns True on first matching interaction.
        """
        for i in range(len(detections)):
            for j in range(i+1, len(detections)):
                interaction_type = self.calculate_interaction(
                    detections[i]['bbox'],
                    detections[j]['bbox']
                )
                if interaction_type in self.filter_interactions:
                    return True
        return False

    def get_color_for_object(self, obj):
        """
        Helper method to get color for a specific object from query parser.
        Returns None if no color specified.
        """
        for element in self.query_parser.get_parsed_queries():
            if 'object' in element and element['object'] == obj and 'color' in element:
                return element['color']
        return None

    def filter_detections_in_frames(self, frames, logic_operator=None, expected_conditions=None):
        """
        Filter frames based on detection logic (and/or).
        Args:
            frames: Dictionary of frames and their detections.
            logic_operator: 'and' or 'or' for filtering logic.
            expected_conditions: List of conditions to match (e.g., [{'object': 'car', 'color': 'red'}]).
        """

        filtered_frames = {}

        # Loop through each frame
        for frame_number, frame_data in frames.items():
            detections = frame_data['detections']
            
            if logic_operator == "and":
                # Track which conditions are matched
                matched_conditions = []
                
                # For each condition, find at least one matching detection
                for condition in expected_conditions:
                    condition_match = any(
                        self.check_detection_condition(detection, condition) 
                        for detection in detections
                    )
                    matched_conditions.append(condition_match)
                
                # Only keep frame if ALL conditions are matched
                if all(matched_conditions):
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

    def save_frame(self, frame_num, detections, frame_file_dir):
        """
        Save annotated frame with detections.
        """
        frame_file_name = f'frame_{frame_num:06d}.jpg'
        frame_file_path = os.path.join(frame_file_dir, frame_file_name)

        # Check if the frame file exists in the specified directory
        if os.path.exists(frame_file_path):
            self.annotate_image(frame_file_path, detections, frame_num)
        else:
            print(f"Warning: Frame file {frame_file_name} does not exist in {self.output_dir}")

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

        # Visualize detections with segmentation masks if the flag is set or with bounding boxes otherwise
        if self.use_segmentation:
            annotated_frame = self.segmenter.annotate_image(image, detections)
        elif not self.use_segmentation:
            # Create a copy for drawing
            annotated_frame = image.copy()

            for detection in detections:
                # Get bounding box coordinates and class
                xmin, ymin, xmax, ymax = detection['bbox']
                # Get dominant color
                dominant_color = detection.get('dominant_color')
                
                # Create a label for the object with known properties
                label_parts = [detection['class_name']]
                
                if dominant_color:
                    color = COLOR_MAP.get(dominant_color, (255, 255, 255))
                    label_parts.append(dominant_color)
                #if detection.get('direction'):
                    #label_parts.append(detection['direction'])
                # Combine the label parts
                label = ' '.join(label_parts)

                # Draw the bounding box
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw transparent label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                # semi-transparent rectangle
                sub_image = annotated_frame[ymin - 20:ymin, xmin:xmin + label_size[0]]

                # Solid color with alpha channel
                rect_color = list(color)
                rect_color.append(0.5)
                
                # Blend the color with the existing background
                for c in range(0, 3):
                    sub_image[:,:,c] = sub_image[:,:,c] * 0.5 + rect_color[c] * 0.5
                
                # Put the modified sub-image back
                annotated_frame[int(ymin)-20:int(ymin), int(xmin):int(xmin)+label_size[0]] = sub_image

                # Draw label text, white for black objects and black for others
                if dominant_color == 'black' or dominant_color == 'blue':
                    cv2.putText(annotated_frame, label,
                        (int(xmin), int(ymin) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(annotated_frame, label,
                            (int(xmin), int(ymin) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            print("Error: No segmentation or bounding box method specified.")
            return

        # Save the annotated image
        output_path = os.path.join(self.found_dir, f"frame_{frame_num:06d}.jpg")
        cv2.imwrite(output_path, annotated_frame)
