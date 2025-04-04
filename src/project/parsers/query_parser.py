""" This module contains the QueryParser class that parses the user query into structured format. """
from transformers import pipeline
from constants.constants import COLORS, OBJECTS, DIRECTIONS, QUADRANTS, INTERACTIONS

class QueryParser:
    def __init__(self, query):
        self.raw_query = query
        self.siglip_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.used_words = set()
        self.query_words = set(self.raw_query.lower().split())
        self.parsed_queries = self.parse_with_siglip(query)

    def parse_with_siglip(self, query):
        """
        Hugging Face's SigLIP model to parse the query into structured components.

        url: https://huggingface.co/docs/transformers/en/model_doc/siglip
        """
        possible_labels = OBJECTS + COLORS + list(DIRECTIONS.values()) + ['and', 'or'] + QUADRANTS + INTERACTIONS


        # Get predictions from SigLIP
        result = self.siglip_model(query, possible_labels, return_tensors="pt")
        print(result)

        # Extract the result into the structured format
        parsed_conditions = self.extract_conditions(result)
        print(parsed_conditions)
        return parsed_conditions

    def extract_conditions(self, result):
        """
        Extract what to search for from SigLIP results.
        """
        conditions = []
        threshold = 0.020
        
        # Store by category
        logic_operators = []
        interactions = []
        directions = []
        colors = []
        objects = []
        quadrants = []
        unknowns = []
        
        # Process and categorize all labels
        for label, score in zip(result['labels'], result['scores']):
            if score >= threshold:
                label_lower = label.lower()
                if label_lower in self.query_words:
                    self.used_words.add(label_lower)
                
                if '-' in label_lower:
                    parts = label_lower.split('-')
                    self.used_words.update(parts)
                    
                if label in DIRECTIONS.values():
                    if score > 0.20:
                        directions.append((label, score))
                elif label in COLORS:
                    colors.append(({'color': label}, score))
                elif label in OBJECTS:
                    objects.append(({'object': label}, score))
                elif label in QUADRANTS:
                    if score > 0.25:
                        quadrants.append(({'quadrant': label}, score))
                elif label in INTERACTIONS:
                    if score > 0.20:
                        interactions.append((label, score))
                elif label in ['and', 'or']:
                    if score > 0.13:
                        logic_operators.append((label, score))
                else:
                    unknowns.append(({'Unknown': label}, score))
        
        # Sort logic operators and filter to same type
        logic_operators.sort(key=lambda x: x[1], reverse=True)
        if logic_operators:
            highest_operator_type = logic_operators[0][0]
            logic_operators = [(op, score) for op, score in logic_operators 
                            if op == highest_operator_type]
        
        # If there is interaction, but no logic operators, add 'and'
        # so it can filter correctly
        has_interaction = len(interactions) > 0
        if has_interaction and not logic_operators:
            # 'and' with score 1.0 so it is always selected
            logic_operators.append(('and', 1.0))

        # Determine how many items to select from each category
        items_per_category = 1
        if logic_operators:
            items_per_category = len(logic_operators) + 1
        
        # Sort each category by score
        colors.sort(key=lambda x: x[1], reverse=True)
        objects.sort(key=lambda x: x[1], reverse=True)
        quadrants.sort(key=lambda x: x[1], reverse=True)
        unknowns.sort(key=lambda x: x[1], reverse=True)
        
        # Add top items from each category
        for category in [colors, objects, quadrants, unknowns]:
            for condition, _ in category[:items_per_category]:
                conditions.append(condition)
        
        # Add logic operators
        for logic_op, _ in logic_operators:
            conditions.append({'logic': logic_op})
            self.used_words.add(logic_op.lower())
        
        # Add top interaction if exists
        interactions.sort(key=lambda x: x[1], reverse=True)
        if interactions:
            conditions.append({'interaction': interactions[0][0]})
            self.used_words.add(interactions[0][0].lower())
        
        # Add top direction if exists
        directions.sort(key=lambda x: x[1], reverse=True)
        if directions:
            conditions.append({'direction': directions[0][0]})
            self.used_words.add(directions[0][0].lower())
        
        return conditions

    def get_parsed_queries(self):
        """Return the parsed query structure."""
        unused_words = self.query_words - self.used_words
        # Convert back to list and maintain original order
        unused = [word for word in self.raw_query.split() 
                 if word.lower() in unused_words]
        print(unused)

        return self.parsed_queries
