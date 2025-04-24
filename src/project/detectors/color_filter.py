"""
This module contains the ColorFilter class, which is used to detect the dominant color within a bounding box of an image.
"""

import numpy as np
import cv2
from typing import Tuple

class ColorFilter:
    def __init__(self):
        """
        Initialize the color filter with predefined HSV ranges for colors.
        """
        # Defines HSV ranges for colors - the values in format (H, S, V) are overlapping to prevent unclassified pixels
        self.color_ranges = {
            'red': [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
            'blue': [((100, 150, 0), (140, 255, 255))],
            'green': [((40, 70, 70), (80, 255, 255))],
            'yellow': [((20, 50, 100), (40, 255, 255))], 
            'white': [((0, 0, 168), (180, 25, 255))],
            'orange': [((5, 100, 100), (25, 255, 255))],
            'purple': [((140, 50, 50), (160, 255, 255))],
            'brown': [((5, 50, 50), (15, 200, 130))],
            'black': [((0, 0, 0), (180, 155, 30))],
            'pink': [((140, 50, 200), (170, 255, 255))],
            'beige': [((15, 30, 150), (25, 100, 255))],
            # Catchall for grays and uncertain colors limited to low saturation areas
            'gray': [((0, 0, 40), (180, 18, 230))]
        }

    def analyze_object_color(self, image, bbox, segmenter=None):
        """
        Analyze the color of an object using segmentation (if available) for more precise results.
        
        Args:
            image (np.ndarray): The input image in BGR format
            bbox (tuple): The bounding box coordinates (xmin, ymin, xmax, ymax)
            segmenter (YOLOSegmenter, optional): The segmenter instance for mask generation
            
        Returns:
            str: The dominant color of the object
            dict: The color presence percentages
        """
        segmentation_mask = None
        if segmenter is not None:
            segmentation_mask = segmenter.segment_region(image, bbox)
            # Ensure the mask is atleast a few pixels in size
            if segmentation_mask is not None:
                mask_height, mask_width = segmentation_mask.shape
                non_zero_pixels = np.count_nonzero(segmentation_mask)
                if mask_height < 5 or mask_width < 5 or non_zero_pixels < 25:
                    # If the mask is too small or has too few non-zero pixels, ignore it
                    # and fallback to the bounding box
                    segmentation_mask = None
        
        # Use the segmentation mask if available
        dominant_color = self.detect_dominant_color(image, bbox, segmentation_mask)
        
        return dominant_color

    def detect_dominant_color(self, image: np.ndarray, bbox: Tuple, segmentation_mask=None) -> str:
        """
        Detect the most prominent color within the bounding box of the image,
        optionally using a segmentation mask for more precise detection.

        Args:
            image (np.ndarray): The original frame in BGR format.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (xmin, ymin, xmax, ymax).
            segmentation_mask (np.ndarray, optional): Binary mask of the object. If None, uses shape-based masking.

        Returns:
            str: The name of the most prominent color, or 'none' if no color is prominent.
        """
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        roi = image[ymin:ymax, xmin:xmax]

        if roi.size == 0:
            return 'none'

        # Create the mask - either use segmentation mask or shape-based mask
        if segmentation_mask is not None:
            # Extract the portion of the segmentation mask that corresponds to the ROI
            mask = segmentation_mask[ymin:ymax, xmin:xmax]
            # Ensure mask is binary (0 or 255)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            # Fall back to the original shape-based masking
            width = xmax - xmin
            height = ymax - ymin
            aspect_ratio = width / height

            if aspect_ratio <= 0.8 or aspect_ratio >= 1.2:
                mask = self.create_elliptical_mask(width, height)
            else:
                mask = self.create_circular_mask(width, height)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        color_presence = self.calculate_color_presence(masked_hsv, mask)

        # Apply threshold and filter out colors with low presence
        threshold = 0.05
        filtered_colors = {color: presence for color, presence in color_presence.items() if presence >= threshold}

        if not filtered_colors:
            # If no colors detected, return the color with the highest presence
            if color_presence:
                return max(color_presence.items(), key=lambda x: x[1])[0]
            return 'none'

        # Return the color with the highest presence
        dominant_color = max(filtered_colors, key=filtered_colors.get)
        return dominant_color

    def create_circular_mask(self, width: int, height: int) -> np.ndarray:
        """
        Create a circular mask for the given width and height.

        Args:
            width (int): The width of the mask.
            height (int): The height of the mask.

        Returns:
            np.ndarray: The circular mask.
        """
        mask = np.zeros((height, width), np.uint8)
        center = (width // 2, height // 2)
        radius = min(width, height) // 2
        cv2.circle(mask, center, radius, 255, -1)
        return mask
    
    def create_elliptical_mask(self, width: int, height: int) -> np.ndarray:
        """
        Create an elliptical mask for the given width and height.

        Args:
            width (int): The width of the mask.
            height (int): The height of the mask.

        Returns:
            np.ndarray: The elliptical mask.
        """
        mask = np.zeros((height, width), np.uint8)
        center = (width // 2, height // 2)
        axes = (width // 2, height // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return mask
    
    def calculate_color_presence(self, hsv: np.ndarray, mask: np.ndarray) -> dict:
        """
        Calculate the presence of different colors in the masked HSV image.
        Works in two passes: first detects all colors except gray, then checks for gray presence
        in unclassified pixels if no other colors are dominant.
        
        Args:
            hsv (np.ndarray): The HSV image.
            mask (np.ndarray): The mask of the region of interest.
        
        Returns:
            dict: A dictionary with color names as keys and their presence as values.
        """
        color_presence = {}
        total_mask_pixels = np.sum(mask) / 255
        
        if total_mask_pixels == 0:
            return {'unknown': 1.0}
            
        unclassified_pixels = np.ones_like(mask) * 255
        
        # First pass: detect all colors except gray
        non_gray_colors = {k: v for k, v in self.color_ranges.items() if k != 'gray'}
        for color, ranges in non_gray_colors.items():
            combined_mask = None
            for lower, upper in ranges:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                current_mask = cv2.inRange(hsv, lower_np, upper_np)
                
                if combined_mask is None:
                    combined_mask = current_mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, current_mask)
            
            if combined_mask is not None:
                matching_pixels = cv2.bitwise_and(combined_mask, mask)
                pixel_count = np.sum(matching_pixels) / 255
                percentage = pixel_count / total_mask_pixels
                if percentage > 0.05:  # Only include if more than 5%
                    color_presence[color] = percentage
                
                # Update unclassified pixels
                unclassified_pixels = cv2.bitwise_and(
                    unclassified_pixels, 
                    cv2.bitwise_not(matching_pixels)
                )
        
        # Second pass: check if unclassified pixels match gray criteria
        unclassified_percentage = np.sum(unclassified_pixels) / (255 * total_mask_pixels)
        if unclassified_percentage > 0.5:  # Only classify as gray if majority unclassified
            gray_ranges = self.color_ranges['gray']
            gray_mask = None
            for lower, upper in gray_ranges:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                current_mask = cv2.inRange(hsv, lower_np, upper_np)
                
                if gray_mask is None:
                    gray_mask = current_mask
                else:
                    gray_mask = cv2.bitwise_or(gray_mask, current_mask)
            
            if gray_mask is not None:
                gray_pixels = cv2.bitwise_and(gray_mask, cv2.bitwise_and(mask, unclassified_pixels))
                gray_percentage = np.sum(gray_pixels) / (255 * total_mask_pixels)
                if gray_percentage > 0.35:  # Only include if significant gray presence
                    color_presence['gray'] = gray_percentage
        
        # If no colors detected, assign the strongest match even if below threshold
        if not color_presence:
            max_presence = 0
            max_color = 'unknown'
            for color, ranges in non_gray_colors.items():
                for lower, upper in ranges:
                    lower_np = np.array(lower, dtype=np.uint8)
                    upper_np = np.array(upper, dtype=np.uint8)
                    current_mask = cv2.inRange(hsv, lower_np, upper_np)
                    matching_pixels = cv2.bitwise_and(current_mask, mask)
                    presence = np.sum(matching_pixels) / (255 * total_mask_pixels)
                    if presence > max_presence:
                        max_presence = presence
                        max_color = color
            color_presence[max_color] = max_presence
        
        return color_presence
