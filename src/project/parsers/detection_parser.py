import os
import json
import cv2
import shutil
from .query_parser import QueryParser
from database.sqlite_database import Database
from constants.constants import COLOR_MAP
from .yolo_segmenter import YOLOSegmenter


class DetectionParser:
    def __init__(self, log_entries, query, output_dir, tracker: str, database_path: str, use_segmentation: bool = True,
                 frame_width: int = 640, frame_height: int = 327):
        self.log_entries = log_entries
        self.database_path = database_path
        self.db = Database(self.database_path)
        self.tracker = tracker
        self.use_segmentation = use_segmentation
        self.connection = None
        self.query_parser = QueryParser(query)
        self.segmenter = YOLOSegmenter()
        self.found_log_entries = {}
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.output_dir = output_dir
        
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
        print(f"Calculating interaction between {bbox1} and {bbox2}")
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate centers
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        # Calculate distance between centers
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        print(f"Distance between centers: {distance}")
        
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
                elif label == 'interaction':
                    self.filter_interactions.append(value)
                elif label == 'quadrant':
                    self.filter_quadrants.append(value)
                else:
                    self.unknown_conditions.append(value)

        print(f"Unknown conditions: {self.unknown_conditions}")
        # If multiple objects or colors are provided and interaction filter is set, add 'and' logic by default
        if (self.filter_interactions) \
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
        frames = self.db.get_frames_with_detections(
            'detections' if self.tracker == '-' else 'refined_detections',
            self.filter_objects, 
            self.filter_colors, 
            self.filter_directions
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
        
        # Loop through each frame and its detections
        for frame_number, frame_data in frames.items():
            detections = frame_data['detections']
            self.found_log_entries[frame_number] = detections
            self.save_frame(frame_number, detections)

        self.save_found_log()

    def apply_spatial_filters(self, frames):
        """Apply spatial filters (interactions and quadrants) to frames."""
        filtered_frames = {}

        for frame_number, frame_data in frames.items():
            detections = frame_data['detections']
            keep_frame = True

            # Filter by quadrants if specified
            if self.filter_quadrants:
                quadrant_match = False
                for detection in detections:
                    quadrant = self.get_frame_quadrant(detection['bbox'], (self.frame_width, self.frame_height))
                    if quadrant in self.filter_quadrants:
                        quadrant_match = True
                        break
                if not quadrant_match:
                    continue

            # Filter by interactions if specified
            if self.filter_interactions:
                interaction_match = False
                
                # Get combinations of objects that should interact based on query
                expected_pairs = []
                if self.filter_logic and self.filter_logic[0] == 'and':
                    # Find pairs of objects/colors that should interact
                    objects_with_props = []
                    for obj in self.filter_objects:
                        color = None
                        # Match colors with objects if specified
                        if self.filter_colors:
                            for element in self.query_parser.get_parsed_queries():
                                if 'object' in element and element['object'] == obj:
                                    if 'color' in element:
                                        color = element['color']
                        objects_with_props.append({'class_name': obj, 'color': color})
                    
                    # Create pairs of objects that should interact
                    for i in range(len(objects_with_props)):
                        for j in range(i + 1, len(objects_with_props)):
                            expected_pairs.append((objects_with_props[i], objects_with_props[j]))

                # Check each pair of detections
                for expected_pair in expected_pairs:
                    obj1_props, obj2_props = expected_pair
                    
                    # Find matching detections for each object in the pair
                    for i, det1 in enumerate(detections):
                        if self.matches_properties(det1, obj1_props):
                            for det2 in detections[i+1:]:
                                if self.matches_properties(det2, obj2_props):
                                    interaction = self.calculate_interaction(det1['bbox'], det2['bbox'])
                                    if interaction in self.filter_interactions:
                                        interaction_match = True
                                        break
                            if interaction_match:
                                break
                        # Check the reverse matching (in case objects are detected in different order)
                        elif self.matches_properties(det1, obj2_props):
                            for det2 in detections[i+1:]:
                                if self.matches_properties(det2, obj1_props):
                                    interaction = self.calculate_interaction(det1['bbox'], det2['bbox'])
                                    if interaction in self.filter_interactions:
                                        interaction_match = True
                                        break
                            if interaction_match:
                                break
                    if interaction_match:
                        break
                
                if not interaction_match:
                    continue

            # If frame passed all filters, add it to filtered frames
            filtered_frames[frame_number] = frame_data

        return filtered_frames

    def matches_properties(self, detection, expected_props):
        """
        Check if a detection matches the expected properties.
        
        Args:
            detection (dict): Detection with class_name, dominant_color, etc.
            expected_props (dict): Expected properties with class_name, color, etc.
        
        Returns:
            bool: True if the detection matches all specified properties
        """
        if expected_props['class_name'] != detection['class_name']:
            return False
        
        if expected_props['color'] is not None and expected_props['color'] != detection['dominant_color']:
            return False
            
        return True

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

        # Visualize detections with segmentation masks if the flag is set or with bounding boxes otherwise
        if self.use_segmentation:
            annotated_image = self.segmenter.annotate_image(image, detections)
        elif not self.use_segmentation:
            # Create a copy for drawing
            annotated_image = image.copy()

            for detection in detections:
                # Get bounding box coordinates and class
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection.get('class_name', 'unknown')
                dominant_color = detection.get('dominant_color', 'unknown')
                color = COLOR_MAP.get(dominant_color, (0, 255, 0))

                # Draw the bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # Add a label to the object
                label = f"{class_name}: {dominant_color}"
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print("Invalid segmentation flag value. Please use True or False.")
            return

        # Save the annotated image
        output_path = os.path.join(self.found_dir, f"frame_{frame_num:04d}_annotated.jpg")
        cv2.imwrite(output_path, annotated_image)
