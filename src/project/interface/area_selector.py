from PyQt6.QtWidgets import QDialog, QVBoxLayout
from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QLabel, QPushButton
from PyQt6.QtGui import QPixmap
import os

class AreaSelector(QDialog):
    # Signal to emit when selection is confirmed
    area_selected = pyqtSignal(QRect)
    
    def __init__(self, parent=None, image_path=None):
        super().__init__(parent)
        self.setWindowTitle("Select Area of Interest")
        self.setMinimumSize(800, 600)
        
        self.layout = QVBoxLayout(self)
        
        # Load the first frame
        self.image_path = image_path
        self.pixmap = None
        if image_path and os.path.exists(image_path):
            self.pixmap = QPixmap(image_path)
        
        # Selection variables
        self.start_point = None
        self.end_point = None
        self.is_selecting = False
        self.selection = QRect()
        
        # Confirm button
        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_selection)
        self.confirm_button.setEnabled(False)
        self.layout.addWidget(self.confirm_button)
        
        # Instructions label
        self.instructions = QLabel("Click and drag to select an area of interest in the video")
        self.layout.addWidget(self.instructions)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if not self.pixmap:
            return
            
        painter = QPainter(self)
        
        # Calculate scaled dimensions to fit the window while preserving aspect ratio
        scaled_pixmap = self.pixmap.scaled(
            self.width(), 
            self.height() - self.confirm_button.height() - self.instructions.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        
        # Draw the image
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height() - 
             self.confirm_button.height() - self.instructions.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # Store image position and scale for coordinate translation
        self.image_rect = QRect(x, y, scaled_pixmap.width(), scaled_pixmap.height())
        
        # Draw selection rectangle if the user is selecting
        if not self.selection.isEmpty():
            # Draw semi-transparent overlay
            overlay = QColor(0, 0, 0, 100)
            painter.fillRect(self.image_rect, overlay)
            
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
            
            # Clear the selection area
            painter.eraseRect(window_selection)
            
            # Draw selection border
            pen = QPen(QColor(255, 0, 0))
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