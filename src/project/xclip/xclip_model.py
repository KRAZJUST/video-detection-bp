from transformers import XCLIPProcessor, XCLIPModel
import torch

class XClipModel:
    def __init__(self):
        self.model_name = "microsoft/xclip-base-patch32"
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

        print(f'Extracted video embeddings: {outputs.shape}')
        return outputs