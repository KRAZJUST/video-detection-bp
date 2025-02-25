import os
import re
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QSlider, QFileDialog, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap

class FrameSlideshow(QMainWindow):
    def __init__(self, parent, starting_frame_path, frame_dir, interval=500):
        super().__init__()
        self.parent = parent
        self.frame_dir = frame_dir
        self.interval = interval  # milliseconds between frames
        
        self.setWindowTitle("Frame Slideshow")
        self.resize(800, 600)
        
        # Create main container and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_label)
        
        # Create slider
        slider_layout = QHBoxLayout()
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.valueChanged.connect(self.show_frame_at_index)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("Frame: 0")
        slider_layout.addWidget(self.frame_label)
        
        main_layout.addLayout(slider_layout)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_slideshow)
        controls_layout.addWidget(self.play_button)
        
        speed_label = QLabel("Speed:")
        controls_layout.addWidget(speed_label)
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Slow (1 fps)", "Medium (2 fps)", "Fast (5 fps)"])
        # default speed to medium
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self.update_speed)
        controls_layout.addWidget(self.speed_combo)
        
        # Export button
        export_button = QPushButton("Export Frame")
        export_button.clicked.connect(self.export_current_frame)
        controls_layout.addWidget(export_button)
        
        main_layout.addLayout(controls_layout)
        
        # Set up timer for slideshow
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        # Find all frames in directory
        self.find_frames()
        
        # Load starting frame
        self.current_frame_index = self.find_starting_index(starting_frame_path)
        self.show_frame_at_index(self.current_frame_index)
        
        # Update speed from combo box
        self.update_speed()
    
    def find_frames(self):
        """Find all frame images in the directory and sort them"""
        self.frame_paths = []
        
        # Get all jpg/png files in the directory
        if os.path.exists(self.frame_dir):
            for filename in sorted(os.listdir(self.frame_dir)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.frame_paths.append(os.path.join(self.frame_dir, filename))
        
        # Set slider range
        self.frame_slider.setRange(0, len(self.frame_paths) - 1)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(5)
    
    def find_starting_index(self, starting_frame_path):
        """Find the index of the starting frame in our frame list"""
        if starting_frame_path in self.frame_paths:
            return self.frame_paths.index(starting_frame_path)
        else:
            # If not found, try to find the frame number from the filename
            try:
                frame_num = int(os.path.basename(starting_frame_path).split('_')[1])
                
                # Look for a frame with similar number
                for i, path in enumerate(self.frame_paths):
                    if f"_{frame_num:04d}" in path or f"_{frame_num}" in path:
                        return i
            except (ValueError, IndexError):
                pass
            
            # If still not found, start at beginning
            return 0
    
    def show_frame_at_index(self, index):
        """Display the frame at the given index"""
        if 0 <= index < len(self.frame_paths):
            self.current_frame_index = index
            frame_path = self.frame_paths[index]
            
            # Load and display image
            pixmap = QPixmap(frame_path)
            
            # Scale pixmap to fit the label while preserving aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
            # Update frame label and slider
            self.frame_label.setText(f"Frame: {index + 1}/{len(self.frame_paths)}")
            
            # Block signals to prevent recursive calls
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(index)
            self.frame_slider.blockSignals(False)
    
    def next_frame(self):
        """Show the next frame in the sequence"""
        next_index = (self.current_frame_index + 1) % len(self.frame_paths)
        self.show_frame_at_index(next_index)
    
    def toggle_slideshow(self):
        """Start or stop the slideshow"""
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(self.interval)
            self.play_button.setText("Pause")
    
    def update_speed(self):
        """Update the slideshow speed based on combo box selection"""
        index = self.speed_combo.currentIndex()
        
        if index == 0:  # Slow
            self.interval = 1000  # 1 fps
        elif index == 1:  # Medium
            self.interval = 500   # 2 fps
        elif index == 2:  # Fast
            self.interval = 200   # 5 fps
        
        # If timer is active, restart with new interval
        if self.timer.isActive():
            self.timer.stop()
            self.timer.start(self.interval)
    
    def export_current_frame(self):
        """Export the current frame to a user-selected location"""
        if 0 <= self.current_frame_index < len(self.frame_paths):
            source_path = self.frame_paths[self.current_frame_index]
            
            # Get save location from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Frame",
                os.path.expanduser("~/Desktop/frame.jpg"),
                "Image files (*.jpg *.png)"
            )
            
            if file_path:
                # Copy the frame to the selected location
                import shutil
                shutil.copy2(source_path, file_path)
                print(f"Exported frame to {file_path}")
    
    def resizeEvent(self, event):
        """Handle resize events to scale the image appropriately"""
        super().resizeEvent(event)
        
        # If we have an image, rescale it
        if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            self.show_frame_at_index(self.current_frame_index)