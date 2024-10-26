""" 
Search for frames in a video by query using CLIP. 

This script extracts frames from a video at a specified interval and searches for relevant images based on a query using the CLIP model.
For the script to work correctly, you need to have all the required libraries installed. You can install them using the following command: 
    pip install torch torchvision transformers opencv-python pillow

Usage:
    python clip_detection.py --video_path <path_to_video> --query <search_query> --output_dir <output_directory> [--interval <interval>]

    NOTE: for the correct functionality of the script, the treshold value should be adjusted in the code and the query should be VERY specific,
          for example instead of searching for 'ambulance' the query should be 'yellow emergency vehicle ambulance'.
"""

import os
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Tuple
import argparse
import torch.nn.functional as F

class CLIPImageSearcher:
    def __init__(self, model_name: str):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def search_images(self, images: List[Image.Image], query: str, threshold: float = 0.5):
        """Search images based on the query using CLIP."""
        inputs = self.processor(text=[query], images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extracts embeddings for images and text
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalizes embeddings
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # Calculates cosine similarity scores
        similarity_scores = torch.matmul(image_embeds, text_embeds.T).squeeze()

        # TODO: remove debug prints
        for i, score in enumerate(similarity_scores):
            print(f"Frame {i}: Similarity score = {score.item()}")

        # Filters frames based on cosine similarity score that exceed the threshold
        top_indices = [(i, similarity_scores[i].item()) for i in range(len(similarity_scores)) if similarity_scores[i].item() > threshold]

        # Return the indices and their similarity scores that exceed the threshold
        return top_indices

class VideoFrameExtractor:
    @staticmethod
    def extract_frames(video_path: str, interval: int, output_dir: str) -> List[Tuple[Image.Image, float]]:
        """Extract frames from the video at a specified interval in seconds and save them."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"ERROR: Unable to open video file '{video_path}'")

        # Get basic video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        print(f"Video FPS: {fps}")
        print(f"Total frames: {frame_count}")
        print(f"Video duration (s): {duration:.2f}")

        last_extracted_time = -1

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        while cap.isOpened():
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
            ret, frame = cap.read()

            if not ret:
                break

            if current_time >= last_extracted_time + interval:
                # Convert frame from BGR (OpenCV format) to RGB (PIL format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                frames.append((pil_image, current_time))
                last_extracted_time = current_time

        cap.release()
        return frames

def save_results(found_indices: List[int], images: List[Image.Image], timestamps: List[float], output_dir: str):
    """Save the found images and their metadata."""
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save images and their corresponding timestamps
    with open(os.path.join(output_dir, "found_frames.txt"), "w") as f:
        for index in found_indices:
            image_path = os.path.join(output_dir, f"frame_{int(timestamps[index])}s.png")
            images[index].save(image_path)
            f.write(f"Frame {index}, Timestamp: {timestamps[index]:.2f}s, Image saved at: {image_path}\n")
            print(f"Saved: {image_path} at {timestamps[index]:.2f}s")

def main(video_path: str, query: str, output_dir: str, interval: int = 2):
    print(f"Query: {query}")
    clip_searcher = CLIPImageSearcher(model_name="openai/clip-vit-base-patch32")

    # Extract frames from the video
    frames_with_timestamps = VideoFrameExtractor.extract_frames(video_path, interval, output_dir)
    frames = [frame[0] for frame in frames_with_timestamps]
    timestamps = [frame[1] for frame in frames_with_timestamps]

    # TODO: add a threshold as CLI argument
    threshold = 0.20
    # Search for relevant images based on the query
    results = clip_searcher.search_images(frames, query, threshold)

    # Save results (images and metadata) where query was found
    found_indices = [index for index, _ in results]
    save_results(found_indices, frames, timestamps, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search frames in a video by query using CLIP.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the found images.")
    parser.add_argument("--query", type=str, help="Search query (e.g., 'a girl in a red dress').")
    parser.add_argument("--interval", type=int, default=2, help="Time interval in seconds to extract frames.")

    args = parser.parse_args()
    main(args.video_path, args.query, args.output_dir, args.interval)
