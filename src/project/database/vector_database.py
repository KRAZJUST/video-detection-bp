# =============================================================================
# File: vector_database.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# IMPORTANT: This module uses open-source vector database ChromaDB.
#   Available at: https://www.trychroma.com/
#
# Description:
# This module provides a class for managing a vector database using ChromaDB.
# It includes methods for adding, querying, and managing embeddings and their 
# metadata.
#
# =============================================================================

import os
import chromadb
import torch
from typing import List, Dict, Any, Optional
import uuid

class VectorDatabaseManager:
    def __init__(self, 
                 database_path: str, 
                 collection_name: str = "video_embeddings",
                 persist_directory: Optional[str] = None,
                 reset_database: bool = False):
        """
        Initialize ChromaDB vector database for storing video embeddings.
        
        Args:
            database_path (str): Path to the database directory
            collection_name (str, optional): Name of the collection to use. Defaults to "video_embeddings"
            persist_directory (str, optional): Directory to persist the database. If None, uses database_path
            reset_database (bool, optional): Whether to reset the database. Defaults to False
        """
        # Ensure database directory exists
        os.makedirs(database_path, exist_ok=True)
        
        # Set persist directory 
        self.persist_directory = persist_directory or database_path
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        print(f"ChromaDB client initialized with path: {self.persist_directory}")

        if reset_database:
            try:
                self.chroma_client.delete_collection(name=collection_name)
                print(f"Collection {collection_name} deleted.")
            except Exception as e:
                print(f"Collection {collection_name} not found.", e)

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for embedding matching
        )
        print(f"Collection {collection_name} created or retrieved.")
        
        self.collection_name = collection_name
        
    def add_batch_embeddings(self, 
                              batch_embeddings, 
                              batch_metadata: List[Dict[str, Any]], 
                              embedding_strategy: str = 'mean'):
        """
        Add embeddings from a batch of frames with associated metadata.
        
        Args:
            batch_embeddings (torch.Tensor): Tensor of embeddings, typically 1 embedding for the batch
            batch_metadata (List[Dict]): Metadata for all frames in the batch
            embedding_strategy (str): Strategy to handle batch embedding 
                                     Options: 'mean', 'first', 'last'
        Returns:
            str: Unique ID for the added batch
        """
        # Validate input
        if batch_embeddings.dim() > 2:
            batch_embeddings = batch_embeddings.squeeze()
        
        # Handle different embedding strategies
        if embedding_strategy == 'mean':
            # If batch_embeddings is 1D, expand to 2D
            if batch_embeddings.dim() == 1:
                batch_embeddings = batch_embeddings.unsqueeze(0)
        elif embedding_strategy == 'first':
            batch_embeddings = batch_embeddings[:1]
        elif embedding_strategy == 'last':
            batch_embeddings = batch_embeddings[-1:]
        else:
            raise ValueError(f"Unknown embedding strategy: {embedding_strategy}")
        
        # Convert torch tensor to list of lists for ChromaDB
        embeddings_list = batch_embeddings.squeeze(0).tolist()
        # Remove the extra batch dimension
        if len(embeddings_list) == 1 and isinstance(embeddings_list[0], list):
            embeddings_list = embeddings_list[0]  # Flatten if embeddings are nested
        
        # Flatten metadata to simple types
        aggregated_metadata = {
            "frame_paths_str": ','.join(meta.get('frame_path', '') for meta in batch_metadata),
            "frame_numbers_str": ','.join(str(meta.get('frame_number', -1)) for meta in batch_metadata),
            "batch_number": batch_metadata[0].get('batch_number', -1),
            "num_frames": len(batch_metadata),
            "timestamps": ','.join(str(meta.get('timestamp', -1)) for meta in batch_metadata),
        }
        
        # Generate a unique ID for this batch
        batch_id = str(uuid.uuid4())

        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings_list,
            metadatas=[aggregated_metadata],
            ids=[batch_id]
        )
        
        return batch_id
    
    def add_single_frame_embedding(self, 
                                   embedding, 
                                    frame_path: str,
                                    frame_number: int,
                                    timestamp: float):
        """
        Add embedding for a single frame with metadata.
        
        Args:
            embedding (torch.Tensor): Tensor of embeddings for a single frame
            frame_path (str): Path to the frame file
            frame_number (int): Frame number in the video
            timestamp (float): Timestamp of the frame in the video
        Returns:
            str: Unique ID for the added frame
        """
        # Ensure embedding is in the right format (1D)
        if embedding.dim() > 1:
            embedding = embedding.squeeze(0)
        
        # Convert torch tensor to list for ChromaDB
        embedding_list = embedding.tolist()
        
        # Create metadata for this frame
        metadata = {
            "frame_paths_str": frame_path,
            "frame_numbers_str": str(frame_number),
            "timestamps": str(timestamp),
            "batch_number": -1,  # Not applicable for single frames
            "num_frames": 1,
        }
        
        # Generate a unique ID for this frame
        frame_id = str(uuid.uuid4())

        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embedding_list,
            metadatas=[metadata],
            ids=[frame_id]
        )
        
        return frame_id

    def query_batch_embeddings(self, 
                                query_embedding: torch.Tensor, 
                                n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector database for similar batch embeddings.
        
        Args:
            query_embedding (torch.Tensor): Embedding to query against
            n_results (int, optional): Number of results to return
        
        Returns:
            Dict containing query results with distances, metadata, etc.
        """
        # Ensure query embedding is in the right format
        if query_embedding.dim() > 2:
            print(f'Squeezing Query embedding shape: {query_embedding.shape}')
            query_embedding = query_embedding.squeeze(0)
        
        if query_embedding.dim() == 1:
            print(f'Unsqueezing Query embedding shape: {query_embedding.shape}')
            query_embedding = query_embedding.unsqueeze(0)
        
        # Convert query embedding to list
        query_list = query_embedding.squeeze(0).tolist()
        
        # Perform query
        query_result = self.collection.query(
            query_embeddings=[query_list],
            n_results=n_results
        )
        
        return query_result
    
    def clear_collection(self):
        """
        Clear all embeddings from the current collection.
        """
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )