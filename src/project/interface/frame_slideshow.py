import os
import re
import shutil
import subprocess
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QSlider, QFileDialog, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from database.sqlite_database import Database
from database.vector_database import VectorDatabaseManager

class FrameSlideshow(QMainWindow):
    def __init__(self, 
                 parent, 
                 starting_frame_path, 
                 frame_dir, 
                 extracted_frames_dir, 
                 context_frames=10, 
                 forward_frames=30, 
                 interval=500, 
                 fps=24,
                 frame_interval=30, 
                 input_video_path=None,
                 database_path=None,
                 tracker=None,
                 metadata=None):
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
        # Path to the input video file
        self.input_video_path = input_video_path
        # Initialize database
        self.db = Database(database_path)
        # Tracker name
        self.tracker = tracker
        if metadata and len(metadata) > 0:
            self.metadata = metadata

        # Connect to the vector database
        self.vector_db = VectorDatabaseManager(
            database_path="vector_database",
            collection_name=f"embeddings_{os.path.basename(input_video_path)}",
        )
        
        # Frame rate and extraction interval for timestamp calculation
        # in .webm files, fps is unknown so to avoid type errors, set to 24
        self.fps = 24 if fps == 'unknown' or fps <= 0 else fps
        self.frame_interval = frame_interval 
        
        self.setWindowTitle("Frame Slideshow")
        self.resize(700, 600)
        
        # Create main container and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_label)
        main_layout.addSpacing(5)
        
        # Create slider
        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.valueChanged.connect(self.show_frame_at_index)
        slider_layout.addWidget(self.frame_slider)
        self.frame_label = QLabel("Frame: 0")
        slider_layout.addWidget(self.frame_label)
        # Add slider layout to main layout
        main_layout.addLayout(slider_layout)
        
        # Create controls
        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_slideshow)
        controls_layout.addWidget(self.play_button)
        self.speed_combo = QComboBox()
        self.speed_combo.setToolTip("Select slideshow speed")
        self.speed_combo.addItems(["Slow (1 fps)", "Medium (2 fps)", "Fast (5 fps)"])
        # default speed to medium
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self.update_speed)
        controls_layout.addWidget(self.speed_combo)
        # Add controls to main layout
        main_layout.addLayout(controls_layout)

        # Create context and forward frames controls
        context_layout = QHBoxLayout()
        context_label = QLabel("Context Frames:")
        context_layout.addWidget(context_label)
        self.context_combo = QComboBox()
        self.context_combo.setToolTip("Number of frames to show before the current frame")
        self.context_combo.addItems(["1", "3", "5", "10", "15", "30", "50"])
        self.context_combo.setCurrentText(str(self.context_frames))
        self.context_combo.currentTextChanged.connect(self.update_context_frames)
        context_layout.addWidget(self.context_combo)

        forward_label = QLabel("Forward Frames:")
        context_layout.addWidget(forward_label)
        self.forward_combo = QComboBox()
        self.forward_combo.setToolTip("Number of frames to show after the current frame")
        self.forward_combo.addItems(["15", "30", "50"])
        self.forward_combo.setCurrentText(str(self.forward_frames))
        self.forward_combo.currentTextChanged.connect(self.update_forward_frames)
        context_layout.addWidget(self.forward_combo)
        # Add context and forward frames controls to main layout
        main_layout.addLayout(context_layout)
        
        # Create action buttons layout
        actions_layout = QHBoxLayout()
        export_button = QPushButton('Export Frame')
        export_button.clicked.connect(self.export_current_frame)
        actions_layout.addWidget(export_button)
        open_button = QPushButton('Open in System Player')
        open_button.clicked.connect(self.open_in_system_player)
        actions_layout.addWidget(open_button)
        # Add action buttons to main layout
        main_layout.addLayout(actions_layout)
        
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
        if self.tracker == 'yolo' or self.tracker == 'bytetrack':
            return self.db.get_frame_timestamp(video_name=self.input_video_path,
                                                frame_number=frame_num,
                                                tracker=self.tracker)
        elif self.tracker == 'xclip-32' or self.tracker == 'xclip-16':
            if self.metadata:
                for batch in self.metadata:
                    # Parse frame numbers from the string
                    frame_numbers = [int(n) for n in batch['frame_numbers_str'].split(',')]
                    timestamps = [float(t) for t in batch['timestamps'].split(',')]
                    
                    # Check if the requested frame is in this batch
                    try:
                        index = frame_numbers.index(frame_num)
                        return timestamps[index]
                    except ValueError:
                        # Frame not found in this batch, continue to next
                        continue
            # If not found in metadata, return a default timestamp
            # this is a safety fallback
            return float(frame_num) * float(self.frame_interval) / float(self.fps)
        else:
            # Default to a constant interval based on fps 
            # this should never be used but is here for safety
            return float(frame_num) * float(self.frame_interval) / float(self.fps)
    
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
        print(f"Frame {frame_num} timestamp: {timestamp}")
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
        #if next_index == len(self.visible_frame_paths) - 1:
        #    current_frame_num = self.visible_frame_numbers[next_index]
        #    if current_frame_num < self.frame_numbers[-1]:
        #        # Get the next frame's index in the full list
        #        next_full_index = self.frame_numbers.index(current_frame_num) + 1
        #        if next_full_index < len(self.frame_numbers):
        #            self.selected_frame_index = next_full_index
        #            self.update_visible_frames()
    
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

    def update_forward_frames(self, value):
        """Update the number of forward frames to show"""
        try:
            self.forward_frames = int(value)
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
        
        # Update the displayed image to fit the new size
        if self.image_label.pixmap():
            self.show_frame_at_visible_index(self.current_frame_index)

    def open_in_system_player(self):
        """
        Opne the video at the current frame in the system's default video player.
        This will attempt to use commonly used Ubuntu video players
        that support seeking to a specific timestamp and open the video.

        If none are found, it will fall back to xdg-open which will open the
        video in the default video player, but not at the specific timestamp.
        """
        if 0 <= self.current_frame_index < len(self.visible_frame_paths):
            if os.path.exists(self.input_video_path):
                timestamp = self.get_frame_timestamp(self.visible_frame_numbers[self.current_frame_index])
                if shutil.which('mpv'):
                    # MPV format
                    subprocess.Popen(['mpv', f'--start={timestamp}', self.input_video_path])
                elif shutil.which('vlc'):
                    # VLC format
                    subprocess.Popen(['vlc', f'--start-time={timestamp}', self.input_video_path])
                elif shutil.which('mplayer'):
                    # MPlayer format
                    subprocess.Popen(['mplayer', f'-ss', f'{timestamp}', self.input_video_path])
                elif shutil.which('ffplay'):
                    # FFplay (part of ffmpeg) format
                    subprocess.Popen(['ffplay', f'-ss', f'{timestamp}', self.input_video_path])
                elif shutil.which('smplayer'):
                    # SMPlayer format
                    subprocess.Popen(['smplayer', f'-start', f'{timestamp}', self.input_video_path])
                else:
                    # Fallback to xdg-open (won't start at specific timestamp)
                    subprocess.Popen(['xdg-open', self.input_video_path])
                    print("Warning: No supported video player found for timestamp seeking. Opening with default player.")
            else:
                print(f"Video file not found: {self.input_video_path}")
        else:
            print("No valid frame selected to open in system player.")