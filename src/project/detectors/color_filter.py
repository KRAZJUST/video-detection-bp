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
            # TODO: add more options
        }

    def detect_dominant_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Detect the most prominent color within the bounding box of the image.

        Args:
            image (np.ndarray): The original frame in BGR format.
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (xmin, ymin, xmax, ymax).

        Returns:
            str: The name of the most prominent color, or 'none' if no color is prominent.
        """
        xmin, ymin, xmax, ymax = bbox
        roi = image[ymin:ymax, xmin:xmax]

        if roi.size == 0:
            return 'none'

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        color_presence = {}
        # Loops through each color and calculate the percentage of that color in the ROI
        for color, ranges in self.color_ranges.items():
            mask = None
            for lower, upper in ranges:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                current_mask = cv2.inRange(hsv, lower_np, upper_np)
                if mask is None:
                    mask = current_mask
                else:
                    mask = cv2.bitwise_or(mask, current_mask)

            # Calculates the percentage of the ROI that matches the color
            percentage = np.sum(mask) / 255 / mask.size
            color_presence[color] = percentage

        # Finds the color with the highest percentage presence
        dominant_color = max(color_presence, key=color_presence.get)

        # Returns the dominant color if it's presence is above threshold, otherwise 'none' TODO: add threshold as parameter and CLI argument
        if color_presence[dominant_color] > 0.05:
            return dominant_color
        return 'none'
