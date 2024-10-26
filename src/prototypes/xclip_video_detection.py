import os
import argparse
import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import XCLIPProcessor, XCLIPModel
from PIL import Image

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))

    def sample_frames(self, clip_len=8, frame_sample_rate=1):
        """Sample frames from the video."""
        self.vr.seek(0)
        seg_len = len(self.vr)
        indices = self._sample_frame_indices(clip_len, frame_sample_rate, seg_len)
        video = self.vr.get_batch(indices).asnumpy()
        return video

    @staticmethod
    def _sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        """Generate random frame indices to sample."""
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

class XCLIPModelWrapper:
    def __init__(self, model_name="microsoft/xclip-base-patch32"):
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.model = XCLIPModel.from_pretrained(model_name)

    def predict(self, video, text_prompts):
        """Run inference on the video using the specified text prompts."""
        inputs = self.processor(
            text=text_prompts,
            videos=list(video),
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_video.softmax(dim=1)
        return probs

class VideoRecognitionApp:
    def __init__(self, video_path, query, output_dir):
        self.video_path = video_path
        self.query = query
        self.output_dir = output_dir
        self.video_processor = VideoProcessor(video_path)
        self.model_wrapper = XCLIPModelWrapper()

    def run(self):
        # Check if the video file exists
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"The video file at {self.video_path} does not exist.")

        # Sample frames from the video
        video_length = len(VideoReader(self.video_path))
        found_images = []
        # Step size for frame sampling
        for idx in range(0, video_length, 8):
            video = self.video_processor.sample_frames()
            print(f"Processing frame indices: {idx} to {idx + 8}")

            # Run inference and get probabilities
            probs = self.model_wrapper.predict(video, [self.query])

            # Check the probabilities and addapt the tresholds
            # Single class case
            if probs.shape[1] == 1:
                if probs[0, 0] > 0.5:
                    found_images.append(video)
            # Multiple classes
            else:
                if probs[0, 1] > 0.5:
                    found_images.append(video)


        self.save_found_images(found_images)

    def save_found_images(self, found_images):
        """Save images where the query was found."""
        os.makedirs(self.output_dir, exist_ok=True)
        for i, image_array in enumerate(found_images):
            # Convert numpy array to image and save
            image = Image.fromarray(image_array[0].astype('uint8'))
            image.save(os.path.join(self.output_dir, f"found_frame_{i}.png"))
            print(f"Saved found frame as 'found_frame_{i}.png' in '{self.output_dir}'")

def main(video_path, query, output_dir):
    app = VideoRecognitionApp(video_path, query, output_dir)
    app.run()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run X-CLIP model on a video and query for specific actions.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("query", type=str, help="Text query to search for in the video.")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory to save found frames.")
    args = parser.parse_args()

    # Run the main function with the provided video path and query
    main(args.video_path, args.query, args.output_dir)
