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
        
        # Filter out labels with low confidence
        threshold = 0.02
        # Placeholder for the logic operator
        logic_operator = None
        logic_score = 0.0
        # Placeholder for the interaction
        interaction = None
        interaction_score = 0.0
        
        for label, score in zip(result['labels'], result['scores']):
            if score >= threshold:
                label_lower = label.lower()
                # Check if this exact label appears in the query
                if label_lower in self.query_words:
                    self.used_words.add(label_lower)
                # Check for compound words (e.g., "north-east")
                if '-' in label_lower:
                    parts = label_lower.split('-')
                    self.used_words.update(parts)

                if label in DIRECTIONS.values():
                    conditions.append({'direction': label})
                elif label in COLORS:
                    conditions.append({'color': label})
                elif label in OBJECTS:
                    conditions.append({'object': label})
                elif label in QUADRANTS:
                    conditions.append({'quadrant': label})
                elif label in INTERACTIONS:
                    if score > interaction_score and score > 0.25:
                        interaction = label
                        interaction_score = score
                elif label in ['and', 'or']:
                    if score > logic_score and score > 0.13:
                        logic_operator = label
                        logic_score = score
                else:
                    conditions.append({'Unknown': label})

        # Append only the logic operator with highest probability score and if it exists
        if logic_operator:
            conditions.append({'logic': logic_operator})
            self.used_words.add(logic_operator.lower())
        # Append only the interaction with highest probability score and if it exists
        if interaction:
            conditions.append({'interaction': interaction})
            self.used_words.add(interaction.lower())
        
        return conditions

    def get_parsed_queries(self):
        """Return the parsed query structure."""
        unused_words = self.query_words - self.used_words
        # Convert back to list and maintain original order
        unused = [word for word in self.raw_query.split() 
                 if word.lower() in unused_words]
        print(unused)

        return self.parsed_queries
