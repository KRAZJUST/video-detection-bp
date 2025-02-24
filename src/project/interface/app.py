import sys
import os
import threading
import time
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QComboBox, QCheckBox, QFileDialog, QProgressBar,
                            QScrollArea, QGridLayout, QGroupBox, QFrame, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PIL import Image, ImageQt
from processors.video_processor import VideoProcessor
from parsers.detection_parser import DetectionParser
from xclip.xclip_parser import XClipParser
from .video_info import VideoInfoUtils

class VideoInfoWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            video_metadata = VideoInfoUtils.get_video_info(self.video_path)
            if video_metadata:
                self.finished.emit(vars(video_metadata))
            else:
                self.error.emit("Error reading video information")
        except Exception as e:
            self.error.emit(str(e))

class VideoProcessingWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, app, video_path, database_path, output_dir, interval, tracker):
        super().__init__()
        self.app = app
        self.video_path = video_path
        self.database_path = database_path
        self.output_dir = output_dir
        self.interval = interval
        self.tracker = tracker

    def run(self):
        try:
            start_time = time.time()
            processor = VideoProcessor(
                video_path=self.video_path,
                database_path=self.database_path,
                output_dir=self.output_dir,
                interval=self.interval,
                tracker_arg=self.tracker
            )
            processor.process_video()
            print(f'Time taken to process video: {time.time() - start_time}')
            
            # Store results in the main app
            if self.tracker == "-":
                self.app.log_results = processor.initial_yolo_results_log
            elif self.tracker in ["bytetrack"]:
                self.app.log_results = processor.log_entries
            elif self.tracker == "xclip":
                self.app.temp_embeddings = processor.temp_embeddings
                
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class QueryWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, app, query):
        super().__init__()
        self.app = app
        self.query = query

    def run(self):
        try:
            if self.app.tracker_combo.currentText() in ["-", "bytetrack"]:
                log_parser = DetectionParser(
                    log_entries=self.app.log_results,
                    query=self.query,
                    output_dir=self.app.output_dir,
                    database_path=self.app.database_path,
                    tracker=self.app.tracker_combo.currentText(),
                    use_segmentation=self.app.use_segmentation.isChecked()
                )
                log_parser.parse_detections()
                self.finished.emit(log_parser.found_log_entries)
                
            elif self.app.tracker_combo.currentText() == "xclip":
                xclip_parser = XClipParser(
                    video_path=self.app.video_path,
                    query=self.query,
                    output_dir=self.app.output_dir,
                )
                similarities, metadata = xclip_parser.search_embeddings(top_k=5)
                self.finished.emit(xclip_parser.top_frames)
        except Exception as e:
            self.error.emit(str(e))

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

class VideoProcessingApp(QMainWindow):
    def __init__(self, database_path: str):
        super().__init__()
        self.database_path = database_path
        self.video_path = ""
        self.output_dir = os.getcwd()
        self.interval = 30
        self.query = ""
        self.log_results = []
        self.found_frames_dir = ""
        self.found_log_entries = {}
        self.temp_embeddings = None
        self.video_info = None

        self.setWindowTitle("Video Processing Application")
        self.setMinimumSize(1400, 900)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create top section with horizontal layout
        top_section = QHBoxLayout()
        
        # Setup all GUI components
        self.setup_io_section(top_section)
        self.setup_query_section(top_section)
        self.setup_settings_section(top_section)
        self.setup_video_info_section(top_section)
        
        # Add top section to main layout
        main_layout.addLayout(top_section)
        
        # Setup results section
        self.setup_results_section(main_layout)

    def setup_io_section(self, parent_layout):
        io_group = QGroupBox("Input & Output")
        layout = QVBoxLayout()
        
        # Video Path
        layout.addWidget(QLabel("Video Path:"))
        self.video_path_entry = QLineEdit()
        layout.addWidget(self.video_path_entry)
        browse_video_btn = QPushButton("Browse Video")
        browse_video_btn.clicked.connect(self.select_video)
        layout.addWidget(browse_video_btn)
        
        # Output Directory
        layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_entry = QLineEdit()
        self.output_dir_entry.setText(self.output_dir)
        layout.addWidget(self.output_dir_entry)
        browse_dir_btn = QPushButton("Browse Directory")
        browse_dir_btn.clicked.connect(self.select_output_dir)
        layout.addWidget(browse_dir_btn)
        
        io_group.setLayout(layout)
        parent_layout.addWidget(io_group)

    def setup_query_section(self, parent_layout):
        query_group = QGroupBox("Query Builder")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Query:"))
        self.query_entry = QLineEdit()
        layout.addWidget(self.query_entry)
        
        # Query controls layout
        controls_layout = QHBoxLayout()
        
        self.query_button = QPushButton("Search Query")
        self.query_button.clicked.connect(self.start_query)
        controls_layout.addWidget(self.query_button)
        
        self.use_segmentation = QCheckBox("Use Segmentation")
        self.use_segmentation.setChecked(True)
        controls_layout.addWidget(self.use_segmentation)
        
        layout.addLayout(controls_layout)
        
        # Progress bar
        self.query_progress = QProgressBar()
        self.query_progress.setVisible(False)
        layout.addWidget(self.query_progress)
        
        query_group.setLayout(layout)
        parent_layout.addWidget(query_group)

    def setup_settings_section(self, parent_layout):
        settings_group = QGroupBox("Settings")
        layout = QVBoxLayout()
        
        # Tracker selection
        layout.addWidget(QLabel("Tracker:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["-", "bytetrack", "deepsort", "xclip"])
        self.tracker_combo.currentTextChanged.connect(self.update_interval_entry)
        layout.addWidget(self.tracker_combo)
        
        # Interval
        layout.addWidget(QLabel("Frame Interval:"))
        self.interval_entry = QLineEdit()
        self.interval_entry.setText("30")
        layout.addWidget(self.interval_entry)
        
        # Process button
        self.process_button = QPushButton("Process Video")
        self.process_button.clicked.connect(self.start_video_processing)
        layout.addWidget(self.process_button)
        
        # Progress bar
        self.processing_progress = QProgressBar()
        self.processing_progress.setVisible(False)
        layout.addWidget(self.processing_progress)
        
        settings_group.setLayout(layout)
        parent_layout.addWidget(settings_group)

    def setup_video_info_section(self, parent_layout):
        video_info_group = QGroupBox("Video Information")
        layout = QVBoxLayout()
        
        self.video_info_label = QLabel("Video Information:\n")
        layout.addWidget(self.video_info_label)
        
        self.video_info_progress = QProgressBar()
        self.video_info_progress.setVisible(False)
        layout.addWidget(self.video_info_progress)
        
        video_info_group.setLayout(layout)
        parent_layout.addWidget(video_info_group)

    def setup_results_section(self, parent_layout):
        # Create scroll area for results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create widget to hold the grid of results
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        
        scroll_area.setWidget(self.results_widget)
        parent_layout.addWidget(scroll_area)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video files (*.mp4 *.avi *.mkv *.mov);;All files (*.*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.relative_video_path = os.path.relpath(file_path)
            if self.relative_video_path.startswith('..'):
                self.video_path_entry.setText(self.relative_video_path)
            
            # Start video info worker
            self.show_loading('video_info', True)
            self.video_info_worker = VideoInfoWorker(file_path)
            self.video_info_worker.finished.connect(self.update_video_info)
            self.video_info_worker.error.connect(self.handle_video_info_error)
            self.video_info_worker.start()

    def update_video_info(self, metadata):
        minutes = int(metadata['duration'] // 60)
        seconds = int(metadata['duration'] % 60)
        
        bitrate_str = "unknown"
        if metadata['bitrate'] != "unknown":
            bitrate_mbps = float(metadata['bitrate']) / 1_000_000
            bitrate_str = f"{bitrate_mbps:.2f} Mbps"
        
        info_text = (
            f"Video Information:\n"
            f"Duration: {minutes:02d}:{seconds:02d}\n"
            f"FPS: {metadata['fps']}\n"
            f"Resolution: {metadata['width']}x{metadata['height']}\n"
            f"Total Frames: {metadata['frame_count']:,}\n"
            f"Codec: {metadata['codec']}\n"
            f"Bitrate: {bitrate_str}\n"
            f"File Size: {metadata['size']}"
        )
        
        self.video_info_label.setText(info_text)
        self.show_loading('video_info', False)

    def handle_video_info_error(self, error_message):
        self.video_info_label.setText(f"Error: {error_message}")
        self.show_loading('video_info', False)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        
        if dir_path:
            self.output_dir = dir_path
            self.found_frames_dir = os.path.join(self.output_dir, "found_frames")
            self.relative_output_dir = os.path.relpath(dir_path)
            if self.relative_output_dir:
                self.output_dir_entry.setText(self.relative_output_dir)

    def update_interval_entry(self, tracker):
        if tracker == "-" or tracker == "xclip":
            self.interval_entry.setText("30")
        else:
            self.interval_entry.setText("10")
        
        # Update segmentation checkbox visibility
        self.use_segmentation.setVisible(tracker in ["-", "bytetrack", "deepsort"])

    def show_loading(self, section, show):
        progress_bar = None
        button = None
        
        if section == 'video_info':
            progress_bar = self.video_info_progress
        elif section == 'video_processing':
            progress_bar = self.processing_progress
            button = self.process_button
        elif section == 'query':
            progress_bar = self.query_progress
            button = self.query_button
            
        if progress_bar:
            progress_bar.setVisible(show)
            if show:
                progress_bar.setRange(0, 0)  # Indeterminate progress
            else:
                progress_bar.setRange(0, 100)
                
        if button:
            button.setEnabled(not show)

    def start_video_processing(self):
        self.show_loading('video_processing', True)
        
        # Start processing worker
        self.processing_worker = VideoProcessingWorker(
            self,
            self.video_path,
            self.database_path,
            self.output_dir,
            int(self.interval_entry.text()),
            self.tracker_combo.currentText()
        )
        self.processing_worker.finished.connect(
            lambda: self.show_loading('video_processing', False)
        )
        self.processing_worker.error.connect(self.handle_processing_error)
        self.processing_worker.start()

    def handle_processing_error(self, error_message):
        print(f"Error processing video: {error_message}")
        self.show_loading('video_processing', False)

    def start_query(self):
        self.show_loading('query', True)
        
        # Start query worker
        self.query_worker = QueryWorker(self, self.query_entry.text())
        self.query_worker.finished.connect(self.display_query_results)
        self.query_worker.error.connect(self.handle_query_error)
        self.query_worker.start()

    def display_query_results(self, results):
        # Clear existing results
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get availible width for columns
        max_cols = 2
        availible_width = self.results_widget.width()
        target_width = int(availible_width / max_cols)

        # Display new results
        row = 0
        col = 0

        for frame_path, metadata in results.items():
            # Ensure frame_path is a string and exists
            if isinstance(frame_path, int):
                print(frame_path)  # Debug print
                # Convert frame number to actual path if needed
                frame_path = os.path.join(self.found_frames_dir, f"frame_{frame_path:04d}_annotated.jpg")
                        
            if os.path.exists(frame_path):
                try:
                    frame_widget = self.create_frame_widget(frame_path, metadata, target_width)
                    self.results_layout.addWidget(frame_widget, row, col)
                    
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                except Exception as e:
                    print(f"Error creating widget for {frame_path}: {str(e)}")
            else:
                print(f"Frame not found: {frame_path}")

        self.show_loading('query', False)

    def create_frame_widget(self, frame_path, metadata, target_width):
        # Create a frame container
        frame_container = QFrame()
        frame_container.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(frame_container)

        # Load and display the image
        pixmap = self.load_frame_image(frame_path, target_width)
        if pixmap.isNull():
            print(f"Error: Pixmap is null for {frame_path}")
        image_label = QLabel()
        # Create a copy of the pixmap to prevent memory issues and overwrites - if not copied, all previous images will be overwritten by the last one
        image_label.setPixmap(pixmap.copy())
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
         # Make the image clickable
        image_label.setCursor(Qt.CursorShape.PointingHandCursor)
        image_label.mousePressEvent = lambda event: self.open_video_at_frame(frame_path, metadata)

        layout.addWidget(image_label)
        return frame_container

    def load_frame_image(self, frame_path, target_width):
        try:
            # Ensure frame_path is a string
            if not isinstance(frame_path, str):
                raise ValueError(f"Invalid frame path type: {type(frame_path)}")
                
            print(f"Loading image from: {frame_path}")  # Debug print
            
            # Check if file exists
            if not os.path.exists(frame_path):
                raise FileNotFoundError(f"Image file not found: {frame_path}")
                
            # Load image using PIL
            pil_image = Image.open(frame_path)
            
            # Calculate aspect ratio preserving resize dimensions
            aspect_ratio = pil_image.width / pil_image.height
            # Adjust height based on aspect ratio
            target_height = int(target_width / aspect_ratio)

            # Resize image
            pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Convert PIL image to QPixmap
            qimage = ImageQt.ImageQt(pil_image)
            pixmap = QPixmap.fromImage(qimage)
            
            return pixmap
        except Exception as e:
            print(f"Error loading image {frame_path}: {str(e)}")
            # Return a blank or error pixmap
            return QPixmap((target_width), target_height)

    def open_video_at_frame(self, frame_path, metadata):
        """Open frame slideshow starting from the selected frame"""
        # Determine the directory containing the frames
        frame_dir = os.path.dirname(frame_path)
        
        # If frame_dir is empty, use the default found_frames directory
        if not frame_dir:
            frame_dir = self.found_frames_dir
        
        # Create and show slideshow window
        self.slideshow_window = FrameSlideshow(self, frame_path, frame_dir)
        self.slideshow_window.show()

    def handle_query_error(self, error_message):
        print(f"Error running query: {error_message}")
        self.show_loading('query', False)

    def closeEvent(self, event):
        """Handle application close event."""
        # Stop any running workers
        if hasattr(self, 'video_info_worker') and self.video_info_worker.isRunning():
            self.video_info_worker.terminate()
        if hasattr(self, 'processing_worker') and self.processing_worker.isRunning():
            self.processing_worker.terminate()
        if hasattr(self, 'query_worker') and self.query_worker.isRunning():
            self.query_worker.terminate()
        
        event.accept()
