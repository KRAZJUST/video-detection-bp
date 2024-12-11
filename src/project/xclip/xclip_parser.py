import torch
from transformers import XCLIPProcessor, XCLIPModel
from torch.nn import functional as F

class XClipParser:
    def __init__(self, temp_embeddings, query: str):
        self.processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
        self.model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.query = query
        
        # Ensure temp_embeddings is a tensor
        if isinstance(temp_embeddings, list):
            # If it's a list of tensors, stack them
            temp_embeddings = torch.stack(temp_embeddings)
        elif not isinstance(temp_embeddings, torch.Tensor):
            # Convert to tensor if not already
            temp_embeddings = torch.tensor(temp_embeddings, dtype=torch.float32)

        if temp_embeddings.dim() == 1:
            temp_embeddings = temp_embeddings.unsqueeze(0)

        self.temp_embeddings = temp_embeddings.to(self.device)

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

            return text_outputs
        except Exception as e:
            print(f'Error in get_query_embeddings: {e}')
            raise

    def search_embeddings(self, top_k=5):
        try:
            print('Searching embeddings...')
            query_embedding = self.get_query_embeddings()

            # Ensure query_embedding is a tensor in 2D
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)

            # Ensure temp_embeddings is a tensor in 2D max
            if self.temp_embeddings.dim() > 2:
                self.temp_embeddings = self.temp_embeddings.view(-1, self.temp_embeddings.shape[-1])
            
            # Normalize embeddings
            query_embedding = F.normalize(query_embedding, p=2, dim=-1)
            temp_embeddings = F.normalize(self.temp_embeddings, p=2, dim=-1)

            # Calculate cosine similarity between query and temp embeddings
            # NOTE: Using permute to match dimensions for matrix multiplication
            similarity = torch.matmul(query_embedding, temp_embeddings.permute(1, 0))

            # Flatten similarity and get top-k
            flattened_similarity = similarity.flatten()
            top_similarities, top_indices = torch.topk(flattened_similarity, k=min(top_k, len(flattened_similarity)))
            
            return top_similarities.cpu().numpy(), top_indices.cpu().numpy()
        except Exception as e:
            print(f'Error in search_embeddings: {e}')
            import traceback
            traceback.print_exc()
            raise