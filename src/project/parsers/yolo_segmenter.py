import numpy as np
import cv2
import torch
from ultralytics import YOLO
from constants.constants import COLOR_MAP

class YOLOSegmenter:
    def __init__(self, model_path='yolo11n-seg.pt'):
        """
        Initialize YOLO Segmenter for segmentation and annotation.
        Args:
            model_path: Path to the YOLO model with segmentation capabilities.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)  # Load YOLO model with segmentation head

    def annotate_image(self, image, detections):
        """
        Annotate image with YOLO segmentation masks and labels.
        Args:
            image: BGR image array
            detections: List of detection dictionaries containing 'bbox', 'class_name', etc.
        Returns:
            Annotated image with masks and labels
        """
        # Ensure input image is in RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO expects an entire image for processing, but we only need the masks for detections in the frame
        # so re-use the detections
        results = self.model(rgb_image)

        # Create a copy for drawing
        annotated_image = image.copy()

        for detection in detections:
            print(detection)
            # Get bounding box coordinates and class
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection.get('class_name', 'unknown')
            dominant_color = detection.get('dominant_color', 'unknown')

            # Filter YOLO results to find the corresponding detection
            for result in results[0].masks.xy:
                # Convert YOLO masks and check if they overlap the detection bbox
                mask = result
                mask_binary = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask_binary, [np.array(mask, dtype=np.int32)], 255)

                # Check intersection with the bbox
                roi_mask = mask_binary[int(y1):int(y2), int(x1):int(x2)]
                if np.sum(roi_mask) > 0:  # If the mask overlaps the bbox
                    # Create a colored overlay for the mask
                    color = COLOR_MAP.get(dominant_color, (0, 255, 0))
                    colored_mask = np.zeros_like(image)
                    for c in range(3):
                        colored_mask[:, :, c] = (mask_binary / 255) * color[c]

                    # Blend the mask with the original image
                    alpha = 0.5
                    annotated_image = cv2.addWeighted(
                        annotated_image, 1,
                        colored_mask.astype(np.uint8), alpha,
                        0
                    )

                    # Draw contours around the mask
                    contours, _ = cv2.findContours(
                        mask_binary,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(annotated_image, contours, -1, color, 2)

                    # Add a label to the object
                    label = f"{class_name}: {dominant_color}"
                    cv2.putText(
                        annotated_image, label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

                    break

        return annotated_image
