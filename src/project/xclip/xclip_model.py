# =============================================================================
# File: xclip_model.py
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
# This module provides a wrapper for the X-CLIP model from HuggingFace transformers library.
# It includes methods for loading the model, processing images, and extracting
# embeddings. The model is used for generating image embeddings for further
# processing and analysis. It processes the frames in batches of 8.
#
# =============================================================================
from transformers import XCLIPProcessor, XCLIPModel
import torch

class XClipModel:
    """
    Module is using Microsoft's X-CLIP model for video-text representation learning.
    
    Model: microsoft/xclip-base-patch32
    Model Type: X-CLIP
    Paper: Ni, Bolin, et al. "Expanding Language-Image Pretrained Models for General Video Recognition." availible at: https://arxiv.org/abs/2208.02816
    
    Repository: https://github.com/microsoft/VideoX/tree/master/X-CLIP
    Huggingface model: https://huggingface.co/microsoft/xclip-base-patch32
    """
    def __init__(self, model_name=None):
        """
        Initialize the XClipModel with a specified model name.

        Args:
            model_name(str, optional): The name of the model to use. If None, defaults to "microsoft/xclip-base-patch32".
        """

        # Use the default model name if none is provided
        if model_name is None:
            self.model_name = "microsoft/xclip-base-patch32"
        elif model_name == 'xclip-32':
            self.model_name = "microsoft/xclip-base-patch32"

        # Load the XCLIP processor and model
        self.processor = XCLIPProcessor.from_pretrained(self.model_name)
        self.model = XCLIPModel.from_pretrained(self.model_name)
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    @torch.no_grad()
    def extract_embeddings(self, frames):
        """
        Extract embeddings from a batch of frames.

        Args:
            frames (list of np.ndarray): List of frames to process. Each frame should be a numpy array.
        Returns:
            torch.Tensor: The extracted video embeddings.
        """
        # Preprocess input frames into tensors and move them to the device
        inputs = self.processor(videos=[frames], return_tensors="pt").to(self.device)

        # Extract video embeddings
        outputs = self.model.get_video_features(**inputs)

        return outputs