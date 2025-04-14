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
        
        # Load model and processor
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print(f"SigLIP model {model_name} loaded successfully")
        
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