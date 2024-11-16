import os
import json
import shutil
import cv2

class DetectionParser:
    def __init__(self, log_entries, query, output_dir):
        self.log_entries = log_entries
        self.queries = self.parse_complex_query(query)
        self.query_colors = []
        self.query_objects = []
        self.found_log_entries = {}
        self.output_dir = output_dir
        # Create a directory to save the found frames
        self.found_dir = os.path.join(output_dir, 'found_frames')
        os.makedirs(self.found_dir, exist_ok=True)

    def parse_complex_query(self, query):
        """
        Parse multiple object queries with optional color and support for AND/OR conditions.
        Example:
            Input: 'car and bus or person'
            Output: [{'logic': 'AND', 'conditions': [('car', None), ('bus', None)]}, {'logic': 'OR', 'conditions': [('person', None)]}]
        """
        # Split by 'or' first, then within each group split by 'and'
        or_groups = [group.strip() for group in query.lower().split('or')]
        parsed_queries = []
        
        for group in or_groups:
            and_conditions = [condition.strip() for condition in group.split('and')]
            conditions = []
            for condition in and_conditions:
                parts = condition.split(' ')
                if len(parts) == 2:
                    color, obj = parts
                    conditions.append((color, obj))
                elif len(parts) == 1:
                    # Only object is specified, no color requirement
                    obj = parts[0]
                    conditions.append((None, obj))
                else:
                    raise ValueError("Query must be in the format 'color object' or 'object', e.g., 'car' or 'red car'")
            parsed_queries.append({'logic': 'AND', 'conditions': conditions})

        return parsed_queries
    
    def parse_log_entries(self):
        """
        Parse log entries and filter based on the query.
        """
        for frame_num, detections in self.log_entries.items():
            # Check if the frame's detections match the query
            if self.check_query(detections):
                self.found_log_entries[frame_num] = detections

                # Get the frame file path
                frame_file_name = f'frame_{frame_num:04d}.jpg'
                frame_file_path = os.path.join(self.output_dir, 'extracted_frames', frame_file_name)

                # Copy the image to the found folder if it exists
                if os.path.exists(frame_file_path):
                    self.annotate_image(frame_file_path, detections, frame_num)
                else:
                    print(f"Warning: Frame file {frame_file_name} does not exist in {self.frames_output_dir}")

        # Save found log entries to a new JSON file
        self.save_found_log()

    
    def check_query(self, detections):
        """
        Check if the frame's detections satisfy any of the complex queries.
        A query can contain multiple conditions connected by 'AND' or 'OR'.
        """
        vehicle_query = ['car', 'truck', 'bus', 'motorcycle']

        # Go through each 'OR' group
        for query_group in self.queries:
            if query_group['logic'] == 'AND':
                # For 'AND', check if each condition is satisfied by at least one detection in the frame
                if all(
                    any(self.matches_condition(detection, condition, vehicle_query) for detection in detections)
                    for condition in query_group['conditions']
                ):
                    return True
            elif query_group['logic'] == 'OR':
                # For 'OR', check if at least one condition is satisfied by any detection in the frame
                if any(
                    any(self.matches_condition(detection, condition, vehicle_query) for detection in detections)
                    for condition in query_group['conditions']
                ):
                    return True

        return False

    

    def matches_condition(self, entry, condition, vehicle_query):
        """Check if a single condition (color-object pair) matches the log entry."""
        query_color, query_object = condition

        if query_color not in self.query_colors:
            self.query_colors.append(query_color)
        if query_object not in self.query_objects:
            self.query_objects.append(query_object)
            if query_object == 'vehicle':
                self.query_objects.extend(vehicle_query)

        # Object matching logic
        if query_object == 'vehicle':
            object_matches = entry['class_name'].lower() in vehicle_query
        else:
            object_matches = entry['class_name'].lower() == query_object
        
        # Color matching logic: skip if color is None
        color_matches = True if query_color is None else entry['dominant_color'].lower() == query_color
        
        confidence = entry['confidence']

        return object_matches and color_matches and confidence >= 0.3
    

    def save_found_log(self):
        """Save the filtered log entries to a new JSON file."""
        found_log_path = os.path.join(self.output_dir, 'found_log.json')
        with open(found_log_path, 'w') as found_log_file:
            json.dump(self.found_log_entries, found_log_file, indent=4)
        print(f"Found log saved to '{found_log_path}'.")


    def annotate_image(self, frame_file_path, detections, frame_num):
        """Annotate the image with bounding boxes and labels."""
        # Read the image
        image = cv2.imread(frame_file_path)

        if image is None:
            print(f"Failed to read image: {frame_file_path}")
            return

        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            dominant_color = detection['dominant_color']

            print(class_name, dominant_color)
            print(self.query_colors, self.query_objects)
            if dominant_color.lower() in self.query_colors and class_name in self.query_objects:
                # Draw the bounding box
                xmin, ymin, xmax, ymax = bbox
                color = (255, 255, 0)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw the label text
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Save the annotated image
        output_path = os.path.join(self.found_dir, f"frame_{frame_num:04d}_annotated.jpg")
        cv2.imwrite(output_path, image)