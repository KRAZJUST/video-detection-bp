from transformers import XCLIPProcessor, XCLIPModel
import torch

class XClipModel:
    def __init__(self):
        self.model_name = "microsoft/xclip-base-patch16-zero-shot"
        self.processor = XCLIPProcessor.from_pretrained(self.model_name)
        self.model = XCLIPModel.from_pretrained(self.model_name)

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def extract_embeddings(self, image_paths):
        # Preprocess input
        inputs = self.processor(images=image_paths, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract image embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0]

        return embedding