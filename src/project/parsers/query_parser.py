""" This module contains the QueryParser class that parses the user query into structured format. """
from transformers import pipeline
from constants.constants import COLORS, OBJECTS, DIRECTIONS

class QueryParser:
    
    def __init__(self, query):
        self.raw_query = query
        self.siglip_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.parsed_queries = self.parse_with_siglip(query)

    def parse_with_siglip(self, query):
        """
        Hugging Face's SigLIP model to parse the query into structured components.

        url: https://huggingface.co/docs/transformers/en/model_doc/siglip
        """
        possible_labels = OBJECTS + COLORS + list(DIRECTIONS.values()) + ['and', 'or']


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
        threshold = 0.2
        
        for label, score in zip(result['labels'], result['scores']):
            if score >= threshold: 
                if label in DIRECTIONS.values():
                    conditions.append({'direction': label})
                elif label in COLORS:
                    conditions.append({'color': label})
                elif label in OBJECTS:
                    conditions.append({'object': label})
        
        return conditions

    def get_parsed_queries(self):
        """Return the parsed query structure."""
        return self.parsed_queries
