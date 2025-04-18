import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import numpy as np
from profiling_utils.profiling_utils import profile_time_usage

class SigLIPModel:
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        """
        Initialize SigLIP model for generating image embeddings.
        
        Args:
            model_name (str): Name or path of the SigLIP model to load
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Load model and processor
        self.processor = SiglipProcessor.from_pretrained(self.model_name)
        self.model = SiglipModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        print(f"SigLIP model {self.model_name} loaded successfully")
        
    @torch.no_grad()
    def extract_embedding(self, image):
        """Extract embedding from a single image"""
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        # Process the image
        # Padding has to be max_length as the model was trained with max_length
        inputs = self.processor(images=image, return_tensors="pt", padding="max_length").to(self.device)
        
        # Extract features
        image_features = self.model.get_image_features(**inputs)
                
        return image_features
    
    @torch.no_grad()
    def extract_batch_embeddings(self, images):
        """
        Extract embeddings from a batch of images individually
        
        Args:
            images: List of PIL.Image or numpy arrays
            
        Returns:
            torch.Tensor: Batch of image embeddings (batch_size x embedding_dim)
        """
        # Convert to PIL Images if needed
        pil_images = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            pil_images.append(img)
            
        # Process the batch of images
        inputs = self.processor(images=pil_images, return_tensors="pt", padding="max_length").to(self.device)
        
        # Extract features
        image_features = self.model.get_image_features(**inputs)
                
        return image_features
    
    @torch.no_grad()
    def extract_clip_embedding(self, images, pooling_strategy='mean'):
        """
        Extract a single embedding from a batch of images by pooling
        
        Args:
            images: List of PIL.Image or numpy arrays
            pooling_strategy: Strategy for pooling embeddings ('mean' or 'max')
            
        Returns:
            torch.Tensor: Single pooled embedding representing the clip (1 x embedding_dim)
        """
        # Get individual embeddings
        batch_embeddings = self.extract_batch_embeddings(images)
        
        # Apply pooling strategy
        if pooling_strategy == 'mean':
            # Average pooling
            clip_embedding = torch.mean(batch_embeddings, dim=0, keepdim=True)
        elif pooling_strategy == 'max':
            # Max pooling
            clip_embedding = torch.max(batch_embeddings, dim=0, keepdim=True)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return clip_embedding
