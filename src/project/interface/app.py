import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,  # type: ignore
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QComboBox, QCheckBox, QFileDialog, QProgressBar,
                            QScrollArea, QGridLayout, QGroupBox, QFrame, )
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image, ImageQt
from .video_info_worker import VideoInfoWorker
from .video_processing_worker import VideoProcessingWorker
from .query_worker import QueryWorker
from .frame_slideshow import FrameSlideshow
from .area_selector import AreaSelector

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
        self.extracted_frames_dir = ""
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
        
        # Segmentation option checkbox
        self.use_segmentation = QCheckBox("Use Segmentation")
        self.use_segmentation.setChecked(True)
        self.use_segmentation.setToolTip("Use segmentation masks for object detection")
        controls_layout.addWidget(self.use_segmentation)

        # Deduplication option checkbox
        self.deduplicate_frames = QCheckBox("Deduplicate Frames")
        self.deduplicate_frames.setChecked(True)
        self.deduplicate_frames.setToolTip("Deduplicate frames with identical objects")
        controls_layout.addWidget(self.deduplicate_frames)

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
        
         # AoI selection
        self.area_selector_button = QPushButton("Select Area of Interest")
        self.area_selector_button.clicked.connect(self.select_area_of_interest)
        # Disable the button until a video is selected
        self.area_selector_button.setEnabled(False)
        layout.addWidget(self.area_selector_button)
        # Label to display the selected area
        self.area_label = QLabel("Area of Interest: Not Selected")
        layout.addWidget(self.area_label)
        # Reset button to clear the selected area
        self.reset_area_button = QPushButton("Reset Area")
        self.reset_area_button.clicked.connect(self.reset_area_of_interest)
        self.reset_area_button.setEnabled(False)
        layout.addWidget(self.reset_area_button)

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

        # Store the video metadata for later use
        self.video_info = metadata
        # Enable the AoI selection button
        self.area_selector_button.setEnabled(True)
        # Extract first frame of the video for AoI selection
        self.extract_first_frame()

    def extract_first_frame(self):
        """Get the first frame of the video for Area of Interest selection"""
        try:
            # Extract the first frame of the video
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Store the frame in memory
                self.first_frame = frame
                return True
            else:
                print("Error extracting first frame")
                return False
        except Exception as e:
            print(f"Error extracting first frame: {str(e)}")
            return False
    
    def select_area_of_interest(self):
        """ Open the area selector dialog to select an area of interest """
        if not hasattr(self, 'first_frame'):
            if not self.extract_first_frame():
                print("Error: Could not extract first frame")
                return
                        
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage from numpy array
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Create QPixmap from QImage
        pixmap = QPixmap.fromImage(q_image)
        
        # Pass pixmap directly to AreaSelector
        self.area_selector = AreaSelector(self)
        self.area_selector.set_pixmap(pixmap)
        self.area_selector.area_selected.connect(self.on_area_selected)
        self.area_selector.exec()
        
    def on_area_selected(self, area):
        """ Handle the area selected by the user"""
        # Store the selected area
        self.aoi = area
        # Update the area label
        self.area_label.setText(f"Area of Interest: ({area.x()}, {area.y()}, "
                                f"{area.width()}, {area.height()})")
        # Enable the reset button
        self.reset_area_button.setEnabled(True)

    def reset_area_of_interest(self):
        """ Reset the selected area of interest """
        if hasattr(self, 'aoi'):
            delattr(self, 'aoi')
        self.area_label.setText("Area of Interest: Not Selected")
        self.reset_area_button.setEnabled(False)

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
            self.extracted_frames_dir = os.path.join(self.output_dir, "extracted_frames")
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
        # TODO: Need to pass the resized video to the processing worker
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
        # Pass deduplication option to the query worker
        self.query_worker.deduplicate = self.deduplicate_frames.isChecked()
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
        self.slideshow_window = FrameSlideshow(self, starting_frame_path=frame_path, 
                                               frame_dir=frame_dir, extracted_frames_dir=self.extracted_frames_dir, 
                                               context_frames=20)
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
