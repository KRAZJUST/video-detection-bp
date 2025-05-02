# =============================================================================
# File: yolo_segmenter.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# IMPORTANT: This module uses YOLOv11-seg from Ultralytics for object instance segmentation.
#   Model from ultralytics: https://docs.ultralytics.com/models/yolo11/
#
# Description:
# This module implements the YOLOSegmenter class, which uses YOLOv11-seg from
# Ultralytics for object instance segmentation of the object. The model's 
# weights are loaded from a specified file path. It also provides method for
# annotating the image with segmentation masks and labels based on existing
# detections. 
#
# =============================================================================

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from constants.constants import COLOR_MAP

class YOLOSegmenter:
    """
    Module using the YOLOv11-seg model for object instance segmentation.
    The model's weights are loaded from a specified file path.

    The segmentation architecture builds on YOLOv11's detection capabilities with an additional
    segmentation head that produces high-quality instance masks.

    Original YOLO architecture by Joseph Redmon et al.
    Model: yolov11n-seg.pt (YOLOv11 with segmentation head)
    Citation: Jocher, G., et al. (2023). Ultralytics YOLOv11-seg. https://docs.ultralytics.com/models/yolo11/
    Repository: https://github.com/ultralytics/ultralytics
    """

    def __init__(self, model_path='yolo11n-seg.pt'):
        """
        Initialize the YOLO Segmenter.
        
        Args:
            model_path: Path to the YOLOv11-seg model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model with segmentation head
        self.model = YOLO(model_path).to(self.device)
        print(f"YOLO model loaded on {self.device}")
    
    def segment_region(self, image, bbox, padding=10):
        """
        Apply YOLOv11-seg only to a specific region defined by the bounding box.
        
        Args:
            image: RGB image array
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            padding: Extra padding around the bbox to ensure complete object capture
            
        Returns:
            Binary mask for the object in the bounding box
        """
        # Extract bbox coordinates and ensure they're integers
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding but stay within image boundaries
        h, w = image.shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)
        
        # Crop the region
        region = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Skip empty regions
        if region.size == 0:
            return None
        
        # Create full-size mask
        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Run segmentation on just the region
        results = self.model(region)
        
        if results[0].masks is not None and len(results[0].masks.xy) > 0:
            # Get the first mask (should be the best one for a cropped region)
            mask_points = results[0].masks.xy[0]
            
            # Adjust coordinates to full image space
            mask_points[:, 0] += x1_pad
            mask_points[:, 1] += y1_pad
            
            # Draw the polygon on the full mask
            cv2.fillPoly(full_mask, [np.array(mask_points, dtype=np.int32)], 255)
        
        return full_mask

    def annotate_image(self, image, detections):
        """
        Annotate image with segmentation masks and labels based on existing detections.
        
        Args:
            image: BGR image array
            detections: List of detection dictionaries containing 'bbox', 'class_name', etc.
        
        Returns:
            Annotated image with masks and labels
        """
        # Convert BGR to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a copy for drawing
        annotated_image = image.copy()
        
        for detection in detections:
            # Get bounding box coordinates and class
            bbox = detection['bbox']
            class_name = detection.get('class_name', 'unknown')
            dominant_color = detection.get('dominant_color', 'unknown')
            
            # Generate mask for this region only
            mask = self.segment_region(rgb_image, bbox)
            
            if mask is not None:
                # Get color for this class
                color = COLOR_MAP.get(dominant_color, (0, 255, 0))
                
                # Create colored overlay
                colored_mask = np.zeros_like(image)
                for c in range(3):
                    colored_mask[:, :, c] = (mask / 255) * color[c]
                
                # Blend the mask with the original image
                alpha = 0.3
                annotated_image = cv2.addWeighted(
                    annotated_image, 1,
                    colored_mask.astype(np.uint8), alpha,
                    0
                )
                
                # Draw contours around the mask
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated_image, contours, -1, color, 2)
                
                # Add a label to the object
                xmin, ymin, xmax, ymax = map(int, bbox)
                label = f"{class_name}: {dominant_color}"
                
                # Get the size of the text to determine rectangle size
                (text_width, text_height), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    2
                )
                
                # Calculate rectangle coordinates
                rect_x = xmin
                rect_y = ymin - text_height - 15
                rect_w = text_width + 10
                rect_h = text_height + 10
                
                # Create overlay for text background
                overlay = annotated_image.copy()
                cv2.rectangle(
                    overlay,
                    (rect_x, rect_y),
                    (rect_x + rect_w, rect_y + rect_h),
                    (64, 64, 64),  # Dark gray
                    -1
                )
                
                # Blend overlay
                alpha = 0.8
                cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
                
                # Add text
                cv2.putText(
                    annotated_image,
                    label,
                    (rect_x + 5, rect_y + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                
        return annotated_image