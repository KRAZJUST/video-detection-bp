import torch
from transformers import XCLIPProcessor, XCLIPModel
from torch.nn import functional as F
from database.vector_database import VectorDatabaseManager
import os
import numpy as np

class XClipParser:
    def __init__(self, video_path, query: str):
        self.processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
        self.model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.query = query
        self.vector_db = VectorDatabaseManager(
            database_path="vector_database",
            collection_name=f"embeddings_{os.path.basename(video_path)}"
        )

    def get_query_embeddings(self):
        try:
            print(f'Query: {self.query}')
            # Check if the query is a list of strings
            query_list = [self.query] if isinstance(self.query, str) else self.query

            text_inputs = self.processor(text=query_list, return_tensors="pt", padding=True).to(self.device)
            print(f'Text inputs: {text_inputs}')

            # Extract text embeddings
            with torch.no_grad():
                text_outputs = self.model.get_text_features(**text_inputs)
            print(f'Text outputs: {text_outputs}')  
            print(f'Text outputs shape: {text_outputs.shape}')
            return text_outputs
        except Exception as e:
            print(f'Error in get_query_embeddings: {e}')
            raise

    def search_embeddings(self, top_k=5):
        """
        Search embeddings in ChromaDB vector database
        
        Args:
            top_k (int): Number of top results to return
        
        Returns:
            Tuple of (similarities, indices, metadata)
        """
        try:
            print('Searching embeddings...')
            
            # Get query embeddings
            query_embedding = self.get_query_embeddings()

            # Perform ChromaDB query for similar embeddings
            query_result = self.vector_db.query_batch_embeddings(
                query_embedding=query_embedding, 
                n_results=top_k
            )
            
            # Get the distances from the query result search in ChromaDB
            distances = query_result['distances'][0]
            
            # Convert distances to similarities (lower distance means higher similarity)
            similarities = 1 / (1 + np.array(distances))
            
            # Get metadata
            metadata = query_result['metadatas'][0]

            return similarities, metadata
        
        except Exception as e:
            print(f'Error in search_embeddings: {e}')
            import traceback
            traceback.print_exc()
            raise