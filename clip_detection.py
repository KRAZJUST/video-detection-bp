import os
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Tuple
import argparse

class CLIPImageSearcher:
    def __init__(self, model_name: str):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def search_images(self, images: List[Image.Image], query: str):
        """Search images based on the query using CLIP."""
        inputs = self.processor(text=[query], images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Calculate cosine similarities
        logits_per_image = outputs.logits_per_image        # Stores image-text similarity scores
        probs = logits_per_image.softmax(dim=1)            # Uses softmax to get the label probabilities

        # Get the indices of the top matches
        top_indices = probs.argsort(descending=True)

        return [(i, probs[0, i].item()) for i in top_indices[0]]

class VideoFrameExtractor:
    @staticmethod
    def extract_frames(video_path: str, interval: int, output_dir: str) -> List[Tuple[Image.Image, float]]:
        """Extract frames from the video at a specified interval in seconds and save them."""
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return frames

        last_extracted_time = -1
        frame_count = 0

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        while cap.isOpened():
            current_frame = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
            ret, frame = cap.read()

            if not ret:
                break

            if current_frame >= last_extracted_time + interval:
                # Convert frame from BGR (OpenCV format) to RGB (PIL format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Save the frame as an image
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
                pil_image.save(frame_filename)
                print(f"Saved frame {frame_count} at {current_frame:.2f}s: {frame_filename}")

                frames.append((pil_image, current_frame))
                last_extracted_time = current_frame
                frame_count += 1

        cap.release()
        return frames


def save_images(images: List[Image.Image], indices: List[int], timestamps: List[float], output_dir: str):
    """Save images to the specified output directory and create a text file with timestamps."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save images
    for index in indices:
        image_path = os.path.join(output_dir, f"frame_{index}.png")
        images[index].save(image_path)
        print(f"Saved: {image_path}")

    # Save timestamps to a text file
    with open(os.path.join(output_dir, "found_frames.txt"), "w") as f:
        for index in indices:
            f.write(f"Frame: {index}, Timestamp: {timestamps[index]:.2f}s\n")

def main(video_path: str, query: str, output_dir: str, interval: int = 2):
    print(query)
    clip_searcher = CLIPImageSearcher(model_name="openai/clip-vit-base-patch32")

    # Extract frames from the video
    frames_with_timestamps = VideoFrameExtractor.extract_frames(video_path, interval, output_dir)
    frames = [frame[0] for frame in frames_with_timestamps]
    timestamps = [frame[1] for frame in frames_with_timestamps]

    # Search for relevant images based on the query
    results = clip_searcher.search_images(frames, query)

    # Save results to the output directory
    save_images(frames, [index for index, _ in results], timestamps, output_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Search images in a video by query using CLIP.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("query", type=str, help="Search query (e.g., 'a girl in a red dress').")
    parser.add_argument("output_dir", type=str, help="Directory to save the found images.")
    parser.add_argument("--interval", type=int, default=2, help="Time interval in seconds to extract frames.")

    args = parser.parse_args()
    main(args.video_path, args.query, args.output_dir, args.interval)
