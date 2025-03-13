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
        # Load YOLO model with segmentation head
        self.model = YOLO(model_path).to(self.device)

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
            # Get bounding box coordinates and class
            xmin, ymin, xmax, ymax = detection['bbox']
            class_name = detection.get('class_name', 'unknown')
            dominant_color = detection.get('dominant_color', 'unknown')

            # Get all masks from YOLO results
            if results[0].masks is None:
                continue
                
            best_mask = self.get_mask_with_highest_iou(results[0].masks.xy, detection['bbox'],image=image)
            
            if best_mask is not None:
                # Create colored overlay
                color = COLOR_MAP.get(dominant_color, (0, 255, 0))
                colored_mask = np.zeros_like(image)
                for c in range(3):
                    colored_mask[:, :, c] = (best_mask / 255) * color[c]

                # Blend the mask with the original image
                alpha = 0.3
                annotated_image = cv2.addWeighted(
                    annotated_image, 1,
                    colored_mask.astype(np.uint8), alpha,
                    0
                )

                # Draw contours around the mask
                contours, _ = cv2.findContours(
                    best_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated_image, contours, -1, color, 2)

                # Add a label to the object
                label = f"{class_name}: {dominant_color}"
                # get the size of the text to determine rectangle size
                (text_width, text_height), _ = cv2.getTextSize(
                    label, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    2
                )

                # Calculate rectangle coordinates
                rect_x = int(xmin)
                rect_y = int(ymin) - text_height - 15
                # + padding
                rect_w = text_width + 10
                rect_h = text_height + 10

                # Create a separate overlay for the semi-transparent rectangle
                overlay = annotated_image.copy()
                cv2.rectangle(
                    overlay,
                    (rect_x, rect_y),
                    (rect_x + rect_w, rect_y + rect_h),
                    # Dark gray
                    (64, 64, 64),
                    -1
                )

                # Add the overlay with transparency
                alpha = 0.8 
                cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)

                # Add the text on top
                cv2.putText(
                    annotated_image,
                    label,
                    # put the label in the middle of the rectangle
                    (rect_x + 5, rect_y + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        return annotated_image

    def get_mask_with_highest_iou(self, masks, bbox, image):
        """
        Select mask with highest IoU (Intersection over Union) with the bounding box
        """
        x1, y1, x2, y2 = map(int, bbox)
        bbox_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 255
        
        best_mask = None
        best_iou = 0
        
        for mask in masks:
            # Create binary mask
            current_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.fillPoly(current_mask, [np.array(mask, dtype=np.int32)], 255)
            
            # Calculate IoU
            intersection = np.logical_and(bbox_mask, current_mask)
            union = np.logical_or(bbox_mask, current_mask)
            iou = np.sum(intersection) / np.sum(union)
            
            if iou > best_iou:
                best_iou = iou
                best_mask = current_mask
        
        # Return the best mask if IoU is above a threshold
        return best_mask if best_iou > 0.2 else None 