# =============================================================================
# File: xclip_parser.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# IMPORTANT: This module uses the SigLIP model from HuggingFace transformers library.
#   Model: microsoft/xclip-base-patch32
#   Model paper: https://arxiv.org/abs/2208.02816
#   Official model repository: https://github.com/microsoft/VideoX/tree/master/X-CLIP
#   Huggingface model: https://huggingface.co/microsoft/xclip-base-patch32
#
# Description:
# This module provides a class for embedding a query using the text encoder of the
# X-CLIP model. It also provides a method for searching the vector database for
# similar frames based on the query. The results are returned as a list of
# dictionaries containing frame information and similarity scores.
# The module also includes a method for copying the top K frames to a specified
# output directory.
#
# =============================================================================

import torch
from transformers import XCLIPProcessor, XCLIPModel
from torch.nn import functional as F
from database.vector_database import VectorDatabaseManager
import os
import numpy as np

class XClipParser:
    """
    Module is using Microsoft's X-CLIP model for video-text representation learning.
    
    Model: microsoft/xclip-base-patch32
    Model Type: X-CLIP
    Paper: Ni, Bolin, et al. "Expanding Language-Image Pretrained Models for General Video Recognition." availible at: https://arxiv.org/abs/2208.02816
    
    Repository: https://huggingface.co/microsoft/xclip-base-patch32
    Huggingface model: https://huggingface.co/microsoft/xclip-base-patch32
    """

    def __init__(self, 
                 video_path,
                 query: str, 
                 output_dir: str, 
                 model_name: str = None):
        """
        Initialize the XClipParser with video path, query, and output directory.

        Args:
            video_path (str): Path to the video file
            query (str): Query string for searching in the video
            output_dir (str): Directory to save the output frames
        """
        # Use the default model name if none is provided
        if model_name is None:
            self.model_name = "microsoft/xclip-base-patch32"
        elif model_name == 'xclip-32':
            self.model_name = "microsoft/xclip-base-patch32"

        self.processor = XCLIPProcessor.from_pretrained(self.model_name)
        self.model = XCLIPModel.from_pretrained(self.model_name)
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.query = query
        self.vector_db = VectorDatabaseManager(
            database_path="vector_database",
            collection_name=f"{model_name}_embeddings_{os.path.basename(video_path)}"
        )
        self.output_dir = output_dir
        self.top_frames = []

    def get_query_embeddings(self):
        """
        Generate embeddings for the query text using the X-CLIP model.

        Returns:
            torch.Tensor: The generated text embeddings.
        """
        try:
            print(f'Query: {self.query}')
            # Check if the query is a list of strings
            query_list = [self.query] if isinstance(self.query, str) else self.query

            text_inputs = self.processor(text=query_list, return_tensors="pt", padding=True).to(self.device)
            print(f'Text inputs: {text_inputs}')

            # Extract text embeddings
            with torch.no_grad():
                text_outputs = self.model.get_text_features(**text_inputs)
                
            # Hugging Face API returns an object with pooler_output in newer versions
            if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
                text_outputs = text_outputs.pooler_output
            elif hasattr(text_outputs, "last_hidden_state") and text_outputs.last_hidden_state is not None:
                text_outputs = text_outputs.last_hidden_state
                
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
            print(f'Query result: {query_result}')
            
            # Get the distances from the query result search in ChromaDB
            distances = query_result['distances'][0]
            # Convert distances to similarities (lower distance means higher similarity)
            similarities = 1 / (1 + np.array(distances))       
            # Get metadata
            metadata = query_result['metadatas'][0]
            print(f'Metadata: {metadata}')

            self.top_frames = self.copy_top_k_frames(metadata)

            return similarities, metadata
        
        except Exception as e:
            print(f'Error in search_embeddings: {e}')
            import traceback
            traceback.print_exc()
            raise

    def copy_top_k_frames(self, metadata):
        """
        Copy top-k found frames to a found_frames directory

        Args:
            metadata (list): List of metadata dictionaries containing frame paths
        Returns:
            dict: Dictionary with frame paths as keys and empty lists as values in 
                    a format to match other parsers
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
        copied_files = set()
        # Dictionary to store results (path → empty list)
        frame_results = {}

        # Iterate through metadata to copy frames
        for meta in metadata:
            # Split frame paths string into individual frame paths
            frame_paths = meta['frame_paths_str'].split(',')
            
            # Copy each frame in the batch
            for frame_path in frame_paths:
                # Extract just the filename
                frame_filename = os.path.basename(frame_path)
                
                # Avoid copying duplicate frames
                if frame_filename not in copied_files:
                    # Construct destination path
                    dest_path = os.path.join(found_frames_dir, frame_filename)
                    
                    # Copy the file
                    try:
                        import shutil
                        shutil.copy2(frame_path, dest_path)
                        copied_files.add(frame_filename)
                        # Add frame path to results with empty list to match other parsers
                        frame_results[frame_path] = []
                        print(f"Copied {frame_filename} to {found_frames_dir}")
                    except Exception as e:
                        print(f"Error copying {frame_path}: {e}")

        return frame_results