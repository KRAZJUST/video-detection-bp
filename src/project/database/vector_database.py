import os
import chromadb
import torch
from typing import List, Dict, Any, Optional
import uuid

class VectorDatabaseManager:
    def __init__(self, 
                 database_path: str, 
                 collection_name: str = "video_embeddings",
                 persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB vector database for storing video embeddings.
        
        Args:
            database_path (str): Path to the database directory
            collection_name (str, optional): Name of the collection to use. Defaults to "video_embeddings"
            persist_directory (str, optional): Directory to persist the database. If None, uses database_path
        """
        # Ensure database directory exists
        os.makedirs(database_path, exist_ok=True)
        
        # Set persist directory 
        self.persist_directory = persist_directory or database_path
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for embedding matching
        )
        
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
        print(f'Embeddings list: {embeddings_list}')
        
        # Flatten metadata to simple types
        aggregated_metadata = {
            "frame_paths_str": ','.join(meta.get('frame_path', '') for meta in batch_metadata),
            "frame_numbers_str": ','.join(str(meta.get('frame_number', -1)) for meta in batch_metadata),
            "batch_number": batch_metadata[0].get('batch_number', -1),
            "num_frames": len(batch_metadata)
        }
        
        # Generate a unique ID for this batch
        batch_id = str(uuid.uuid4())
        
        print(f"Adding batch embeddings with ID: {batch_id}")
        print(f"Batch metadata: {aggregated_metadata}")
        print(f"Embeddings: {batch_embeddings}")

        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings_list,
            metadatas=[aggregated_metadata],
            ids=[batch_id]
        )
        
        return batch_id
    
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
            query_embedding = query_embedding.squeeze(0)
        
        if query_embedding.dim() == 1:
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