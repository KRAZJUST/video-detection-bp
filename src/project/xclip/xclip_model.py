# This code uses Microsoft's X-CLIP model from Hugging Face
# Citation: Ni, Bolin, et al. "Expanding Language-Image Pretrained Models for General Video Recognition." availible at: https://arxiv.org/abs/2208.02816
# Model: microsoft/xclip-base-patch32 (https://huggingface.co/microsoft/xclip-base-patch32)

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
        elif model_name == 'xclip-16':
            self.model_name = "microsoft/xclip-base-patch16"

        # Load the XCLIP processor and model
        self.processor = XCLIPProcessor.from_pretrained(self.model_name)
        self.model = XCLIPModel.from_pretrained(self.model_name)
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    @torch.no_grad()
    def extract_embeddings(self, frames):
        # Preprocess input frames into tensors and move them to the device
        inputs = self.processor(videos=[frames], return_tensors="pt").to(self.device)

        # Extract video embeddings
        outputs = self.model.get_video_features(**inputs)

        return outputs