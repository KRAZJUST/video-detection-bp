from PIL import Image
import torch
from transformers import SiglipProcessor, SiglipModel
import numpy as np
import os
from database.vector_database import VectorDatabaseManager

class SigLIPParser:
    def __init__(self,
                 video_path,
                 model_name: str = None,
                 query: str = None,
                 output_dir: str = None):
        """
        Initialize SigLIP parser for querying video frames.
        
        Args:
            video_path: Path to the video file
            model_name: SigLIP model name or path
            query: Optional initial query text
            output_dir: Directory for outputs
        """
        if model_name is None or model_name == 'siglip':
            self.model_name = "google/siglip-base-patch16-224"

        self.query = query
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.siglip_model = SiglipModel.from_pretrained(self.model_name).to(self.device)
        self.siglip_processor = SiglipProcessor.from_pretrained(self.model_name)
        self.collection_name = f"siglip_embeddings_{os.path.basename(video_path)}"

        self.top_frames = []
        
        # Initialize vector database
        self.vector_db = VectorDatabaseManager(
            database_path="vector_database",
            collection_name=self.collection_name
        )
    
    def get_query_embedding(self, text):
        """Generate embedding for a text query"""
        # Check if the query is a list of strings
        query_list = [text] if isinstance(text, str) else text
        text_inputs = self.siglip_processor(text=query_list, return_tensors="pt", padding=True).to(self.device)
        # Extract text embeddings
        with torch.no_grad():
            text_outputs = self.siglip_model.get_text_features(**text_inputs)

        return text_outputs

    def search_embeddings(self, n_results=5):
        """
        Query frames using text description.
        
        Args:
            query: Text description to search for (overrides self.query if provided)
            n_results: Number of results to return
            
        Returns:
            List of dictionaries with frame information and similarity scores
        """        
        # Generate text embedding
        text_embedding = self.get_query_embedding(self.query)
        
        # Query vector database
        results = self.vector_db.query_batch_embeddings(
            query_embedding=text_embedding,
            n_results=n_results
        )
        
        # Get the distances from the results
        distances = results['distances'][0]
        # Convert distances to similarities (lower distance means higher similarity)
        similarities = 1 / (1 + np.array(distances))
        # Get metadata
        metadata = results['metadatas'][0]

        self.top_frames = self.copy_top_k_frames(metadata)

        return similarities, metadata
    
    def copy_top_k_frames(self, metadata):
        """
        Copy top K frames to the output directory.
        
        Args:
            metadata: Metadata containing frame paths and other information
            
        Returns:
            List of copied frame paths
        """
        # Ensure found_frames directory exists
        found_frames_dir = os.path.join(self.output_dir, 'found_frames')
        if not os.path.exists(found_frames_dir):
            os.makedirs(found_frames_dir, exist_ok=True)
        else:
            # Clear existing frames
            import shutil
            shutil.rmtree(found_frames_dir)
            os.makedirs(found_frames_dir, exist_ok=True)
        
        # Track copied files to avoid duplicates
        copied_frames = []
        # Directory to store results (path → empty list)
        frame_results = {}
        
        for meta in metadata:
            # Extract the frame path of the single frame
            frame_path = meta['frame_paths_str']
            # Extract just the filename
            frame_filename = os.path.basename(frame_path)

            # Copy the frame to the found_frames directory
            if frame_filename not in copied_frames:
                # construct the destination path
                dest_path = os.path.join(found_frames_dir, frame_filename)
                # Copy the file
                try:
                    import shutil
                    shutil.copy(frame_path, dest_path)
                    copied_frames.append(dest_path)
                    # add frame path to the results with empty list to match other parsers
                    frame_results[frame_path] = []
                except Exception as e:
                    print(f"Error copying frame {frame_filename}: {e}")
                    continue
        
        return frame_results