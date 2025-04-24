# This code uses Facebook's BART model for zero-shot text classification
# from Huggingface transformers
# Citation: Lewis, M., et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. ACL 2020. https://arxiv.org/abs/1910.13461
# Model: facebook/bart-large-mnli (https://huggingface.co/facebook/bart-large-mnli)

from transformers import pipeline
from constants.constants import COLORS, OBJECTS, DIRECTIONS, QUADRANTS, INTERACTIONS

class QueryParser:
    """
    Module using Facebook's BART model for zero-shot text classification
    accessed through Huggingface Transformers.
    This class is designed to parse user queries into structured categories
    such as colors, objects, directions, quadrants, and interactions.  

    Model: facebook/bart-large-mnli
    Model Type: BART
    Paper: Lewis, M., et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training 
    for Natural Language Generation, Translation, and Comprehension. ACL 2020.
    https://arxiv.org/abs/1910.13461
    
    Repository: https://github.com/facebookresearch/fairseq/tree/main/examples/bart
    Huggingface model: https://huggingface.co/facebook/bart-large-mnli
    """

    def __init__(self, query):
        self.raw_query = query
        self.bart_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.used_words = set()
        self.query_words = set(query.lower().split())
        self.parsed_queries = self.parse_with_bart(query)

    def parse_with_bart(self, query):
        """
        Parse the query using BART model for zero-shot classification.
        This function takes a query string and classifies it into various categories
        such as colors, objects, directions, quadrants, and interactions.
        It returns a structured format of the parsed query.

        Using the huggingface pipeline for zero-shot classification with BART model.
        """
        possible_labels = OBJECTS + COLORS + list(DIRECTIONS.values()) + ['and', 'or'] + QUADRANTS + INTERACTIONS

        # Get predictions
        result = self.bart_model(query, possible_labels, return_tensors="pt")
        print(result)

        # Extract the result into the structured format
        parsed_conditions = self.extract_conditions(result, query)
        print(parsed_conditions)
        return parsed_conditions

    def extract_conditions(self, result, query):
        """
        Extract conditions from the BART model result.

        This function processes the result of the BART model and categorizes
        the labels into different categories such as colors, objects, directions,
        quadrants, interactions, and logic operators. It also handles the
        categorization of unknown labels. The function returns a list of
        structured conditions based on the parsed query.
        The function uses a threshold to filter out low-confidence predictions
        and categorizes the labels based on their type and score.

        Args:
            result (dict): The result from the BART model containing labels and scores.
            query (str): The original query string.
        Returns:
            list: A list of structured conditions extracted from the result.
        """
        conditions = []
        threshold = 0.020
        
        # Initialize categorized collections
        categorized_results = {
            'logic_operators': [],
            'interactions': [],
            'directions': [],
            'colors': [],
            'objects': [],
            'quadrants': [],
            'unknowns': []
        }
        
        # high interactions threshold as it likes to appear when it is not
        # actually present in the query
        interaction_threshold = 0.37
        # Check if interaction word actually appears in the query
        interaction_words_in_query = [word for word in INTERACTIONS if word.lower() in query.lower().split()]
        
        # Process and categorize all labels
        for label, score in zip(result['labels'], result['scores']):
            if score < threshold:
                continue
                
            label_lower = label.lower()
            if label_lower in self.query_words:
                self.used_words.add(label_lower)
            
            if '-' in label_lower:
                self.used_words.update(label_lower.split('-'))
                
            # Special handling for interactions
            if label in INTERACTIONS:
                # Only add interaction if:
                # 1. It has very high confidence (> interaction_threshold) OR
                # 2. It actually appears in the query AND has decent confidence (> 0.20)
                if (score > interaction_threshold) or (label.lower() in interaction_words_in_query and score > 0.20):
                    categorized_results['interactions'].append((label, score))
            elif label in DIRECTIONS.values() and score > 0.20:
                categorized_results['directions'].append((label, score))
            elif label in COLORS:
                categorized_results['colors'].append(({'color': label}, score))
            elif label in OBJECTS:
                categorized_results['objects'].append(({'object': label}, score))
            elif label in QUADRANTS and score > 0.25:
                categorized_results['quadrants'].append(({'quadrant': label}, score))
            elif label in ['and', 'or'] and score > 0.1:
                categorized_results['logic_operators'].append((label, score))
            else:
                categorized_results['unknowns'].append(({'Unknown': label}, score))
                
        # Handle logic operators 
        logic_operators = categorized_results['logic_operators']
        if logic_operators:
            # Get the highest confidence logic operator type
            logic_operators.sort(key=lambda x: x[1], reverse=True)
            highest_operator_type = logic_operators[0][0]
            highest_score = logic_operators[0][1]
            
            # Count occurrences with regex pattern to get the precise number of
            # logic operators in the query
            import re
            pattern = r'\b{}\b'.format(re.escape(highest_operator_type))
            count = len(re.findall(pattern, query.lower()))
            count = max(count, 1)  # Ensure at least one if detected
            
            # create a list of logic operators with the highest score
            logic_operators = [(highest_operator_type, highest_score)] * count
        
        # Add default 'and' if the interactions are present and no logic operators
        # are found
        if categorized_results['interactions'] and not logic_operators:
            logic_operators = [('and', 1.0)]

        # Calculate items per category once
        items_per_category = len(logic_operators) + 1 if logic_operators else 1
        
        # Sort and add top items from each category
        for category_name in ['colors', 'objects', 'quadrants', 'unknowns']:
            category = categorized_results[category_name]
            category.sort(key=lambda x: x[1], reverse=True)
            conditions.extend(cond for cond, _ in category[:items_per_category])
        
        # Add logic operators
        for logic_op, _ in logic_operators:
            conditions.append({'logic': logic_op})
            self.used_words.add(logic_op.lower())
        
        # Add top interaction if exists
        interactions = categorized_results['interactions']
        if interactions:
            interactions.sort(key=lambda x: x[1], reverse=True)
            conditions.append({'interaction': interactions[0][0]})
            self.used_words.add(interactions[0][0].lower())
        
        # Add top direction if exists
        directions = categorized_results['directions']
        if directions:
            directions.sort(key=lambda x: x[1], reverse=True)
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
