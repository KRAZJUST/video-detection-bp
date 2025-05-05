# =============================================================================
# File: siglip_model.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# IMPORTANT: This module uses the SigLIP model from HuggingFace transformers library.
#   Model: google/siglip-base-patch16-224
#   Model paper: https://arxiv.org/abs/2303.15343
#   Huggingface model: https://huggingface.co/google/siglip-base-patch16-224
#
# Description:
# This module provides a wrapper for the SigLIP model from Hugging Face.
# It includes methods for loading the model, processing images, and extracting
# embeddings. The model is used for generating image embeddings for further
# processing and analysis.
#
# =============================================================================

import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import numpy as np
from profiling_utils.profiling_utils import profile_time_usage

class SigLIPModel:
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        """
        Module is using the SigLIP model from HuggingFace transformers library.
        Model: google/siglip-base-patch16-224
        Model paper: https://arxiv.org/abs/2303.15343
        Huggingface model: https://huggingface.co/google/siglip-base-patch16-224
        
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
        """
        Extract embedding from a single image
        
        Args:
            image (np.ndarray or PIL.Image): Input image to process. Can be a numpy array or PIL Image.
        Returns:
            torch.Tensor: Extracted image features
        """
            
        # Process the image
        # Padding has to be max_length as the model was trained with max_length
        inputs = self.processor(images=image, return_tensors="pt", padding="max_length").to(self.device)
        
        # Extract features
        image_features = self.model.get_image_features(**inputs)
                
        return image_features