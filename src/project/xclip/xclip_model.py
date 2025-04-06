from transformers import XCLIPProcessor, XCLIPModel
import torch

class XClipModel:
    def __init__(self, model_name=None):
        """
        Initialize the XCLIP model and processor.
        :param model_name: Optional; specify a different model name if needed.
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

    def extract_embeddings(self, frames):
        # Preprocess input frames into tensors and move them to the device
        inputs = self.processor(videos=[frames], return_tensors="pt").to(self.device)

        # Extract video embeddings
        with torch.no_grad():
            outputs = self.model.get_video_features(**inputs)

        return outputs