import os
import re
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QSlider, QFileDialog, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap

class FrameSlideshow(QMainWindow):
    def __init__(self, parent, starting_frame_path, frame_dir, extracted_frames_dir, 
                 context_frames=10, forward_frames=30, interval=500, fps=24,
                 frame_interval=24):
        super().__init__()
        self.parent = parent
        self.frame_dir = frame_dir
        self.extracted_frames_dir = extracted_frames_dir
        # Number of frames to show before the current frame
        self.context_frames = context_frames
        # Number of frames to show after the current frame
        self.forward_frames = forward_frames
        # milliseconds between frames in slideshow
        self.interval = interval
        
        # Frame rate and extraction interval for timestamp calculation
        self.fps = fps
        self.frame_interval = frame_interval
        
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
        
        # Context frames control
        context_label = QLabel("Context Frames:")
        controls_layout.addWidget(context_label)
        
        self.context_combo = QComboBox()
        self.context_combo.addItems(["1", "3", "5", "10", "15", "30", "50"])
        self.context_combo.setCurrentText(str(self.context_frames))
        self.context_combo.currentTextChanged.connect(self.update_context_frames)
        controls_layout.addWidget(self.context_combo)
        
        # Export button
        export_button = QPushButton("Export Frame")
        export_button.clicked.connect(self.export_current_frame)
        controls_layout.addWidget(export_button)
        
        main_layout.addLayout(controls_layout)
        
        # Set up timer for slideshow
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        # Find all frames in directories
        self.find_frames()
        
        # Load starting frame
        self.selected_frame_index = self.find_starting_index(starting_frame_path)
        self.update_visible_frames()
        
        # Update speed from combo box
        self.update_speed()
    
    def get_frame_timestamp(self, frame_num):
        """Calculate timestamp based on frame number and constant extraction rate"""
        # timestamp = (frame_num * interval) / fps
        # this gives us the time in seconds
        # only works with constant frame intervals, so if adding the dynamic 
        # frame extraction interval, refactor this to fetch timestamps from database
        seconds = (float(frame_num) * float(self.frame_interval)) / float(self.fps)
        return seconds
    
    def format_timestamp(self, seconds):
        """Format seconds into MM:SS.mmm"""
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:06.3f}"
    
    def find_frames(self):
        """Find all frames in both directories and organize them"""
        # Dictionary to store all available frames by frame number
        self.all_frames = {}
        self.annotated_frames = {}
        
        # Get all frames from the extracted frames directory
        if os.path.exists(self.extracted_frames_dir):
            for filename in sorted(os.listdir(self.extracted_frames_dir)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract frame number using regex for frame_%5d.jpg format
                    match = re.search(r'frame_(\d+)', filename)
                    if match:
                        frame_num = int(match.group(1))
                        self.all_frames[frame_num] = os.path.join(self.extracted_frames_dir, filename)
        
        # Get all annotated frames from the frame directory
        if os.path.exists(self.frame_dir):
            for filename in sorted(os.listdir(self.frame_dir)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract frame number using regex
                    match = re.search(r'frame_(\d+)', filename)
                    if match:
                        frame_num = int(match.group(1))
                        self.annotated_frames[frame_num] = os.path.join(self.frame_dir, filename)
                        # Also add to all_frames so we don't miss any frames
                        self.all_frames[frame_num] = os.path.join(self.frame_dir, filename)
        
        # Create sorted list of all frame numbers
        self.frame_numbers = sorted(self.all_frames.keys())
    
    def find_starting_index(self, starting_frame_path):
        """Find the index of the starting frame in our frame list"""
        # Extract frame number from starting_frame_path using the frame_%5d.jpg format
        match = re.search(r'frame_(\d+)', os.path.basename(starting_frame_path))
        if match:
            frame_num = int(match.group(1))
            if frame_num in self.frame_numbers:
                return self.frame_numbers.index(frame_num)
        
        # If not found, start at beginning
        return 0
    
    def update_visible_frames(self):
        """Update the list of frames visible in the slideshow based on context"""
        if not self.frame_numbers:
            return
        
        selected_frame_num = self.frame_numbers[self.selected_frame_index]
        
        # Get range of frame numbers to show
        start_idx = max(0, self.frame_numbers.index(selected_frame_num) - self.context_frames)
        end_idx = min(len(self.frame_numbers) - 1, 
                      self.frame_numbers.index(selected_frame_num) + self.forward_frames)
        
        # Create list of visible frame numbers
        self.visible_frame_numbers = self.frame_numbers[start_idx:end_idx + 1]
        
        # Update current index in visible frames
        self.current_frame_index = self.visible_frame_numbers.index(selected_frame_num)
        
        # Update visible frames paths
        self.visible_frame_paths = []
        for frame_num in self.visible_frame_numbers:
            # Check if we have an annotated version first
            if frame_num in self.annotated_frames:
                self.visible_frame_paths.append(self.annotated_frames[frame_num])
            else:
                self.visible_frame_paths.append(self.all_frames[frame_num])
        
        # Update slider to show only visible frames
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, len(self.visible_frame_paths) - 1)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
        
        # Show current frame
        self.show_frame_at_visible_index(self.current_frame_index)
    
    def show_frame_at_visible_index(self, index):
        """Display the frame at the given visible index"""
        if not self.visible_frame_paths or index < 0 or index >= len(self.visible_frame_paths):
            return
        
        self.current_frame_index = index
        frame_path = self.visible_frame_paths[index]
        frame_num = self.visible_frame_numbers[index]
        
        # Load and display image
        pixmap = QPixmap(frame_path)
        
        # Scale pixmap to fit the label while preserving aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        
        # Get timestamp for this frame using the constant interval calculation
        timestamp = self.get_frame_timestamp(frame_num)
        formatted_time = self.format_timestamp(timestamp)
        
        # Update frame label and indicate if this is an annotated frame
        is_annotated = frame_num in self.annotated_frames
        annotation_status = " (Annotated)" if is_annotated else ""
        
        # Show frame info - current/total, frame number and timestamp
        self.frame_label.setText(
            f"Frame: {index + 1}/{len(self.visible_frame_paths)} - " + 
            f"#{frame_num}{annotation_status} - Time: {formatted_time}"
        )
    
    def show_frame_at_index(self, index):
        """Handle when user changes the slider"""
        if 0 <= index < len(self.visible_frame_paths):
            self.current_frame_index = index
            # Show the frame
            self.show_frame_at_visible_index(index)
    
    def next_frame(self):
        """Show the next frame in the visible sequence"""
        next_index = (self.current_frame_index + 1) % len(self.visible_frame_paths)
        self.show_frame_at_visible_index(next_index)
        
        # update to get the next chunk when we reach the end of the visible frames
        """if next_index == len(self.visible_frame_paths) - 1:
            current_frame_num = self.visible_frame_numbers[next_index]
            if current_frame_num < self.frame_numbers[-1]:
                # Get the next frame's index in the full list
                next_full_index = self.frame_numbers.index(current_frame_num) + 1
                if next_full_index < len(self.frame_numbers):
                    self.selected_frame_index = next_full_index
                    self.update_visible_frames()"""
    
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
    
    def update_context_frames(self, value):
        """Update the number of context frames to show"""
        try:
            self.context_frames = int(value)
            self.update_visible_frames()
        except ValueError:
            pass
    
    def export_current_frame(self):
        """Export the current frame to a user-selected location"""
        if 0 <= self.current_frame_index < len(self.visible_frame_paths):
            source_path = self.visible_frame_paths[self.current_frame_index]
            frame_num = self.visible_frame_numbers[self.current_frame_index]
            timestamp = self.get_frame_timestamp(frame_num)
            formatted_time = self.format_timestamp(timestamp).replace(":", "_")
            
            # Get save location from user with suggested filename including timestamp
            suggested_name = f"frame_{frame_num:05d}_{formatted_time}.jpg"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Frame",
                os.path.expanduser(f"~/Desktop/{suggested_name}"),
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
            self.show_frame_at_visible_index(self.current_frame_index)