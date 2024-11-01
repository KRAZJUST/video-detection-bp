import os
import json
import shutil


class DetectionParser:
    def __init__(self, log_entries, query, output_dir):
        self.log_entries = log_entries
        self.queries = self.parse_complex_query(query)
        self.found_log_entries = {}
        self.output_dir = output_dir
        # Create a directory to save the found frames
        self.found_dir = os.path.join(output_dir, 'found_frames')
        os.makedirs(self.found_dir, exist_ok=True)

    def parse_complex_query(self, query):
        """
        Parse multiple color-object pairs with support for AND/OR conditions.
        Example input: 'red car and blue truck or yellow bus'
        Output: [{'logic': 'AND', 'conditions': [('red', 'car'), ('blue', 'truck')]}, {'logic': 'OR', 'conditions': [('yellow', 'bus')]}]
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
                else:
                    raise ValueError("Query must be in the format 'color object', for example 'red car'")
            parsed_queries.append({'logic': 'AND', 'conditions': conditions})

        print(parsed_queries)
        return parsed_queries
    
    def parse_log_entries(self):
        """
        Parse log entries and filter based on the query.
        """
        for frame_num, detections in self.log_entries.items():
            for entry in detections:
                # Check if the entry matches the query
                if self.check_query(entry):
                    if frame_num not in self.found_log_entries:
                        self.found_log_entries[frame_num] = []
                    self.found_log_entries[frame_num].append(entry)

                    # Get the frame file path
                    frame_file_name = f'frame_{frame_num:04d}.jpg'
                    frame_file_path = os.path.join(self.output_dir, 'extracted_frames', frame_file_name)

                    # Copy the image to the found folder if it exists
                    if os.path.exists(frame_file_path):
                        shutil.copy(frame_file_path, self.found_dir)
                    else:
                        print(f"Warning: Frame file {frame_file_name} does not exist in {self.frames_output_dir}")

        # Save found log entries to a new JSON file
        self.save_found_log()
    
    def check_query(self, entry):
        """
        Check if the log entry matches any of the complex queries.
        A query can contain multiple conditions connected by 'AND' or 'OR'.
        """
        vehicle_query = ['car', 'truck', 'bus', 'motorcycle']

        # Go through each 'OR' group
        for query_group in self.queries:
            if query_group['logic'] == 'AND':
                # All conditions in this group must be satisfied
                if all(self.matches_condition(entry, condition, vehicle_query) for condition in query_group['conditions']):
                    return True
            elif query_group['logic'] == 'OR':
                # At least one condition in this group must be satisfied
                if any(self.matches_condition(entry, condition, vehicle_query) for condition in query_group['conditions']):
                    return True
                
        return False
    

    def matches_condition(self, entry, condition, vehicle_query):
        """Check if a single condition (color-object pair) matches the log entry."""
        query_color, query_object = condition
        
        # Object matching logic
        if query_object == 'vehicle':
            object_matches = entry['class_name'].lower() in vehicle_query
        else:
            object_matches = entry['class_name'].lower() == query_object
        
        # Color matching logic
        color_matches = entry['dominant_color'].lower() == query_color
        
        confidence = entry['confidence']

        return object_matches and color_matches and confidence >= 0.3
    

    def save_found_log(self):
        """Save the filtered log entries to a new JSON file."""
        found_log_path = os.path.join(self.output_dir, 'found_log.json')
        with open(found_log_path, 'w') as found_log_file:
            json.dump(self.found_log_entries, found_log_file, indent=4)
        print(f"Found log saved to '{found_log_path}'.")