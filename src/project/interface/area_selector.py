# =============================================================================
# File: area_selector.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module provides a dialog for interactively selecting an area of interest
# in a video frame. The user can click and drag to create a selection rectangle,
# and the selected area can be confirmed with a button. The selected area is
# emitted as a QRect object.
#
# =============================================================================

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton    # type: ignore
from PyQt6.QtCore import Qt, QRect, pyqtSignal                           # type: ignore
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QRegion         # type: ignore
import os

class AreaSelector(QDialog):
    # Signal to emit when selection is confirmed
    area_selected = pyqtSignal(QRect)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Area of Interest")
        self.setMinimumSize(800, 600)
        
        self.layout = QVBoxLayout(self)

        # Instructions label at the top
        self.instructions = QLabel("Click and drag to select an area of interest in the video")
        self.layout.addWidget(self.instructions)

        # Space for the image
        self.layout.addStretch(1)

        # Confirm button at the bottom
        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_selection)
        self.confirm_button.setEnabled(False)
        self.layout.addWidget(self.confirm_button)

        # Initialize pixmap to None
        self.pixmap = None

        # Selection variables
        self.start_point = None
        self.end_point = None
        self.is_selecting = False
        self.selection = QRect()

    def set_pixmap(self, pixmap):
        """Set the pixmap from a QPixmap object"""
        self.pixmap = pixmap

    def paintEvent(self, event):
        super().paintEvent(event)
    
        if not self.pixmap:
            return
            
        painter = QPainter(self)
        
        # Reserve space for UI elements at the bottom
        ui_reserved_height = 100 
        
        # Calculate available space
        available_width = self.width()
        available_height = self.height() - ui_reserved_height
        
        # Calculate scaled dimensions
        scaled_pixmap = self.pixmap.scaled(
            available_width,
            available_height,
            Qt.AspectRatioMode.KeepAspectRatio
        )
        
        # Center the image in the available space
        x = (available_width - scaled_pixmap.width()) // 2
        y = (available_height - scaled_pixmap.height()) // 2
        
        # Draw the image
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # Store image position and scale for coordinate translation
        self.image_rect = QRect(x, y, scaled_pixmap.width(), scaled_pixmap.height())
        
        # Draw selection rectangle if the user is selecting
        if not self.selection.isEmpty():         
            # Calculate the selection rectangle in window coordinates
            rel_x = self.selection.x() / self.pixmap.width() * scaled_pixmap.width()
            rel_y = self.selection.y() / self.pixmap.height() * scaled_pixmap.height()
            rel_w = self.selection.width() / self.pixmap.width() * scaled_pixmap.width()
            rel_h = self.selection.height() / self.pixmap.height() * scaled_pixmap.height()
            
            window_selection = QRect(
                int(x + rel_x),
                int(y + rel_y),
                int(rel_w),
                int(rel_h)
            )

            image_region = QRegion(self.image_rect)
            selection_region = QRegion(window_selection)
            # Region for the area outside the selection
            outside_selection = image_region.subtracted(selection_region)
                        
            # Draw semi-transparent overlay to everything except selected area
            painter.setClipRegion(outside_selection)
            overlay = QColor(0, 0, 0, 120)  # Semi-transparent black
            painter.fillRect(self.image_rect, overlay)
            painter.setClipping(False)  # Disable clipping

            # Draw selection border
            pen = QPen(QColor(0, 0, 0, 64))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(window_selection)
    
    def mousePressEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton and 
                self.image_rect.contains(event.position().toPoint())):
            self.is_selecting = True
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point
            self.update_selection()
    
    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.end_point = event.position().toPoint()
            self.update_selection()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.end_point = event.position().toPoint()
            self.update_selection()
            self.confirm_button.setEnabled(not self.selection.isEmpty())
            self.update()
    
    def update_selection(self):
        if not self.start_point or not self.end_point or not self.image_rect:
            return
            
        # Calculate selection coordinates relative to the window
        window_selection = QRect(
            min(self.start_point.x(), self.end_point.x()),
            min(self.start_point.y(), self.end_point.y()),
            abs(self.start_point.x() - self.end_point.x()),
            abs(self.start_point.y() - self.end_point.y())
        )
        
        # Convert window coordinates to image coordinates
        rel_x = (window_selection.x() - self.image_rect.x()) / self.image_rect.width()
        rel_y = (window_selection.y() - self.image_rect.y()) / self.image_rect.height()
        rel_w = window_selection.width() / self.image_rect.width()
        rel_h = window_selection.height() / self.image_rect.height()
        
        # Apply to original image dimensions
        self.selection = QRect(
            int(rel_x * self.pixmap.width()),
            int(rel_y * self.pixmap.height()),
            int(rel_w * self.pixmap.width()),
            int(rel_h * self.pixmap.height())
        )
        
        # Ensure the selection is within image bounds
        self.selection = self.selection.intersected(
            QRect(0, 0, self.pixmap.width(), self.pixmap.height())
        )
    
    def confirm_selection(self):
        if not self.selection.isEmpty():
            self.area_selected.emit(self.selection)
            self.accept()