""" This module contains the QueryParser class that parses the user query into structured format. """

class QueryParser:
    DIRECTIONS = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']
    
    def __init__(self, query):
        self.raw_query = query
        self.parsed_queries = self.parse_complex_query(query)

    def parse_complex_query(self, query):
        """
        Parse queries with color, object, and direction into structured formats.
        """
        or_groups = [group.strip() for group in query.lower().split('or')]
        parsed_queries = []
        print(f"QUERY: {query}")
        
        for group in or_groups:
            and_conditions = [condition.strip() for condition in group.split('and')]
            conditions = []
            for condition in and_conditions:
                parts = condition.split(' ')
                color, obj, direction = None, None, None
                if len(parts) == 3:
                    color, obj, direction = parts
                elif len(parts) == 2:
                    if parts[1] in self.DIRECTIONS:
                        obj, direction = parts
                    else:
                        color, obj = parts
                elif len(parts) == 1:
                    obj = parts[0]
                else:
                    raise ValueError("Invalid query format.")
                conditions.append({'color': color, 'object': obj, 'direction': direction})
            parsed_queries.append({'logic': 'AND', 'conditions': conditions})
        
        return parsed_queries

    def get_parsed_queries(self):
        """Return the parsed query structure."""
        return self.parsed_queries
