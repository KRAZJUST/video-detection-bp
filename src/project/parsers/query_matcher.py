""" Query matcher class to check if detections satisfy a parsed query. """

class QueryMatcher:
    def __init__(self, parsed_queries, vehicle_aliases=None):
        self.parsed_queries = parsed_queries
        self.vehicle_aliases = vehicle_aliases or ['car', 'truck', 'bus']
        self.parsed_colors = []
        self.parsed_objects = []

    def matches(self, detection, condition):
        """
        Check if a single detection matches a query condition.
        """
        query_color = condition.get('color')
        query_object = condition.get('object')
        query_direction = condition.get('direction')

        # Store parsed colors and objects for later use in the annotation
        if query_color and query_color not in self.parsed_colors:
            self.parsed_colors.append(query_color)
        if query_object and query_object not in self.parsed_objects:
            self.parsed_objects.append(query_object)
            if query_object == 'vehicle':
                self.parsed_objects.extend(self.vehicle_aliases)

        # Match object
        object_matches = (
            detection['class_name'].lower() == query_object or
            (query_object == 'vehicle' and detection['class_name'].lower() in self.vehicle_aliases)
        ) if query_object else True
        
        # Match color
        color_matches = (
            detection['dominant_color'].lower() == query_color
        ) if query_color else True
        
        # Match direction
        direction_matches = (
            detection.get('direction', '').lower() == query_direction
        ) if query_direction else True

        # Combine matches
        confidence = detection['confidence']
        return object_matches and color_matches and direction_matches and confidence >= 0.3

    def matches_query_group(self, detections, query_group):
        """
        Check if a group of conditions connected by AND/OR matches detections.
        """
        if query_group['logic'] == 'AND':
            return all(
                any(self.matches(detection, condition) for detection in detections)
                for condition in query_group['conditions']
            )
        elif query_group['logic'] == 'OR':
            return any(
                any(self.matches(detection, condition) for detection in detections)
                for condition in query_group['conditions']
            )
        return False

    def check_query(self, detections):
        """
        Check if detections satisfy any of the parsed query groups.
        """
        return any(self.matches_query_group(detections, group) for group in self.parsed_queries)
    
    def get_parsed_colors(self):
        """Return the parsed colors."""
        return self.parsed_colors
    
    def get_parsed_objects(self):
        """Return the parsed objects."""
        return self.parsed_objects
