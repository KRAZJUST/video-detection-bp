"""
This module contains the ColorFilter class, which is used to detect the most prominent color within a bounding box of an image.
"""

import numpy as np
import cv2
from typing import Tuple

class ColorFilter:
    def __init__(self):
        """
        Initialize the color filter with predefined HSV ranges for colors.
        """
        # Defines HSV ranges for colors
        self.color_ranges = {
            'red': [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
            'blue': [((100, 150, 0), (140, 255, 255))],
            'green': [((40, 70, 70), (80, 255, 255))],
            'yellow': [((20, 100, 100), (30, 255, 255))],
            'white': [((0, 0, 200), (180, 25, 255))],
            'orange': [((10, 100, 100), (20, 255, 255))],
            'purple': [((140, 50, 50), (160, 255, 255))],
            'brown': [((10, 50, 50), (20, 200, 200))],
            'black': [((0, 0, 0), (180, 255, 30))],
            'gray': [((0, 0, 40), (180, 30, 200))],
        }

    def detect_dominant_color(self, image: np.ndarray, bbox: Tuple) -> str:
        """
        Detect the most prominent color within the bounding box of the image.

        Args:
            image (np.ndarray): The original frame in BGR format.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (xmin, ymin, xmax, ymax).

        Returns:
            str: The name of the most prominent color, or 'none' if no color is prominent.
        """
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        roi = image[ymin:ymax, xmin:xmax]

        if roi.size == 0:
            return 'none'

        width = xmax - xmin
        height = ymax - ymin
        aspect_ratio = width / height

        if aspect_ratio <= 0.8 and aspect_ratio >= 1.2:
            mask = self._create_circular_mask(width, height)
        else:
            mask = self._create_elliptical_mask(width, height)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        color_presence = self._calculate_color_presence(masked_hsv, mask)

         # Apply threshold and filter out colors with low presence
        threshold = 0.001
        filtered_colors = {color: presence for color, presence in color_presence.items() if presence >= threshold}

        if not filtered_colors:
            return 'none'

        # Return the color with the highest presence
        dominant_color = max(filtered_colors, key=filtered_colors.get)
        print(f'dominant_color: {dominant_color}')
        return dominant_color

    def _create_circular_mask(self, width: int, height: int) -> np.ndarray:
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
    
    def _create_elliptical_mask(self, width: int, height: int) -> np.ndarray:
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
    
    def _calculate_color_presence(self, hsv: np.ndarray, mask: np.ndarray) -> dict:
        """
        Calculate the presence percentage of each predefined color in the masked HSV region.
        """
        color_presence = {}
        for color, ranges in self.color_ranges.items():
            combined_mask = None
            for lower, upper in ranges:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                current_mask = cv2.inRange(hsv, lower_np, upper_np)
                if combined_mask is None:
                    combined_mask = current_mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, current_mask)

            # Calculate the percentage of masked pixels matching the color
            # sum the matching pixels, divide by 255 to normalize, and then by the total number of pixels in the mask
            intersection_pixels = np.sum(cv2.bitwise_and(combined_mask, mask))
            total_mask_pixels = np.sum(mask)
            if total_mask_pixels > 0:  # Avoid division by zero
                percentage = intersection_pixels / total_mask_pixels
            else:
                percentage = 0

            color_presence[color] = percentage

        return color_presence
