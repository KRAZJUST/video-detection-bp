# =============================================================================
# File: app.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module implements the main application window. It is divded into several
# sections and uses PyQt6 for the UI. It also uses several workers classes from
# the interface directory to handle video processing, querying, and displaying
# results.
#
# =============================================================================

import os
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,  # type: ignore
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QComboBox, QSpinBox, QFileDialog, QProgressBar,
                            QScrollArea, QGridLayout, QGroupBox, QFrame)
from PyQt6.QtCore import Qt, QEvent, QMetaObject, Q_ARG, QRect
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image, ImageQt
from .video_info_worker import VideoInfoWorker
from .video_processing_worker import VideoProcessingWorker
from .query_worker import QueryWorker
from .frame_slideshow import FrameSlideshow
from .area_selector import AreaSelector
from .advanced_settings import AdvancedSettings
from .processing_settings import ProcessingSettings

class VideoProcessingApp(QMainWindow):
    def __init__(self, database_path: str):
        """
        Initialize the main application window.

        Args:
            database_path (str): Path to the database
        """
        super().__init__()
        self.database_path = database_path
        self.video_path = ""
        self.output_dir = os.getcwd()
        self.interval = 30
        self.query = ""
        self.found_frames_dir = ""
        self.extracted_frames_dir = ""
        self.found_log_entries = {}
        self.temp_embeddings = None
        self.video_info = None
        self.results_batch_count = 5
        self.results_frames_count = 40
        self.deduplicate_frames_value = True
        self.use_segmentation_value = False
        self.processing_segmentation_value = False
        self.skip_siglip_with_yolo_value = False

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
        """
        Setup the input and output section of the UI.

        Args:
            parent_layout: The layout to add the section to
        """
        io_group = QGroupBox("Input and Output")
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
        """
        Setup the query builder section of the UI.

        Args:
            parent_layout: The layout to add the section to
        """
        query_group = QGroupBox("Query Builder")
        layout = QVBoxLayout()
        
        # Query input
        query_row = QHBoxLayout()
        # Left side - Query text entry
        layout.addWidget(QLabel("Query:"))
        self.query_entry = QLineEdit()
        query_row.addWidget(self.query_entry)
        # Right side - Advanced settings button
        self.advanced_settings_button = QPushButton("⚙")  # Gear icon
        self.advanced_settings_button.setToolTip("Query Settings")
        self.advanced_settings_button.setMaximumSize(30, 30)  # Make it small
        self.advanced_settings_button.clicked.connect(self.open_advanced_settings)
        query_row.addWidget(self.advanced_settings_button)
        # Add the query row to the main layout
        layout.addLayout(query_row)

        # Query controls layout
        controls_layout = QHBoxLayout()
        # Search button
        self.query_button = QPushButton("Search Query")
        self.query_button.clicked.connect(self.start_query)
        controls_layout.addWidget(self.query_button)
        # add controls layout to the main layout
        layout.addLayout(controls_layout)
        
        # Progress bar
        self.query_progress = QProgressBar()
        self.query_progress.setVisible(False)
        layout.addWidget(self.query_progress)

        # Query feedback message
        self.query_feedback = QLabel("")
        self.query_feedback.setWordWrap(True)
        layout.addWidget(self.query_feedback)

        # Results count label
        self.results_count_label = QLabel("Number of frames: -")
        layout.addWidget(self.results_count_label)
        
        query_group.setLayout(layout)
        parent_layout.addWidget(query_group)

    def setup_settings_section(self, parent_layout):
        """
        Setup the processing settings section of the UI.

        Args:
            parent_layout: The layout to add the section to
        """
        settings_group = QGroupBox("Processing Settings")
        layout = QVBoxLayout()
        
        # horizontal layout for tracker and interval
        tracker_row = QHBoxLayout()
        # Left side - Tracker selection
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["yolo", "bytetrack", "xclip-32", "siglip"])
        self.tracker_combo.currentTextChanged.connect(self.update_interval_entry)
        self.tracker_combo.setToolTip("Select the tracker to use for processing.")
        tracker_row.addWidget(self.tracker_combo)
        # Centre - Interval
        self.interval_entry = QSpinBox()
        self.interval_entry.setRange(1, 200)
        self.interval_entry.setValue(30)
        self.interval_entry.setToolTip("Interval for frames extraction.")
        tracker_row.addWidget(self.interval_entry)
        # Right side - Advanced settings button
        self.processing_settings_button = QPushButton("⚙")  # Gear icon
        self.processing_settings_button.setToolTip("Processing Settings")
        self.processing_settings_button.setMaximumSize(30, 30)  # Make it small
        self.processing_settings_button.clicked.connect(self.open_processing_settings)
        tracker_row.addWidget(self.processing_settings_button)
        # Add the query row to the main layout
        layout.addLayout(tracker_row)

        process_buttons_layout = QHBoxLayout()
        # Process button
        self.process_button = QPushButton("Process Video")
        self.process_button.setMinimumSize(310, 30)
        self.process_button.clicked.connect(self.start_video_processing)
        process_buttons_layout.addWidget(self.process_button)
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMaximumSize(self.process_button.width(), 30)
        self.cancel_button.clicked.connect(self.cancel_video_processing)
        self.cancel_button.setVisible(False) # Initially invisible
        process_buttons_layout.addWidget(self.cancel_button)
        # Add the process buttons layout to the main layout
        layout.addLayout(process_buttons_layout)
        
        # Progress bar
        self.processing_progress = QProgressBar()
        self.processing_progress.setMaximumSize(self.cancel_button.width(), 30)
        self.processing_progress.setVisible(False)
        layout.addWidget(self.processing_progress)
        # Status message
        self.status_message = QLabel("Ready to process video")
        self.status_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_message)
        
        settings_group.setLayout(layout)
        parent_layout.addWidget(settings_group)

    def setup_video_info_section(self, parent_layout):
        """
        Setup the video information section of the UI.

        Args:
            parent_layout: The layout to add the section to
        """
        video_info_group = QGroupBox("Video Information")
        layout = QVBoxLayout()
        
        self.video_info_label = QLabel()
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
        """
        Setup the results section of the UI for displaying found frames.

        Args:
            parent_layout: The layout to add the section to
        """
        # Create scroll area for results
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Create widget to hold the grid of results
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setSpacing(10)  # Add some spacing between items
        
        # Set column stretch to make them equal width
        self.results_layout.setColumnStretch(0, 1)
        self.results_layout.setColumnStretch(1, 1)
        
        self.scroll_area.setWidget(self.results_widget)
        parent_layout.addWidget(self.scroll_area)
        
        # Connect resize event to handle responsive layout
        self.scroll_area.viewport().installEventFilter(self)
        
        # For dynamic loading
        self.loaded_images = []
        self.all_results = {}
        # For batch loading - 10 rows at a time
        self.batch_size = 10
        self.current_batch = 0

    def select_video(self):
        """
        Open a file dialog to select a video file and start the video info worker.
        """
        # Clear previous results
        self.clear_results_layout()
        
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
        """
        Update the video information label with metadata from the video.
        
        Args:
            metadata: Dictionary containing video metadata
        """
        minutes = int(metadata['duration'] // 60)
        seconds = int(metadata['duration'] % 60)
        
        bitrate_str = "unknown"
        if metadata['bitrate'] != "unknown":
            bitrate_mbps = float(metadata['bitrate']) / 1_000_000
            bitrate_str = f"{bitrate_mbps:.2f} Mbps"
        
        info_text = (
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

        # Update the interval entry based on the tracker
        self.interval_entry.setValue(10) if self.tracker_combo.currentText() == "bytetrack" else self.interval_entry.setValue(30)
        # Store the video metadata for later use
        self.video_info = metadata
        # Enable the AoI selection button
        self.area_selector_button.setEnabled(True)
        # Extract first frame of the video for AoI selection
        self.extract_first_frame()

    def open_advanced_settings(self):
        """
        Open the advanced settings dialog for query settings.
        """
        settings_dialog = AdvancedSettings(self)
        if settings_dialog.exec():
            # Apply the settings when OK button is clicked
            self.apply_advanced_settings(settings_dialog)

    def apply_advanced_settings(self, dialog):
        """
        Apply the advanced settings from the dialog.
        """
        # Store all settings
        self.results_batch_count = dialog.batch_count_spinbox.value()
        self.results_frames_count = dialog.frame_count_spinbox.value()

        # Store the deduplication and segmentation settings
        self.deduplicate_frames_value = dialog.deduplicate_frames_checkbox.isChecked()
        self.use_segmentation_value = dialog.use_segmentation_checkbox.isChecked()

    def open_processing_settings(self):
        """
        Open the processing settings dialog for video processing.
        """
        settings_dialog = ProcessingSettings(self)
        if settings_dialog.exec():
            # Apply the settings when OK button is clicked
            self.apply_processing_settings(settings_dialog)
    
    def apply_processing_settings(self, dialog):
        """
        Apply the processing settings from the dialog.

        Args:
            dialog: The dialog containing the settings
        """
        # Store the processing segmentation value and skip siglip value
        self.processing_segmentation_value = dialog.use_segmentation_checkbox.isChecked()
        self.skip_siglip_with_yolo_value = dialog.skip_siglip_frames_checkbox.isChecked()

    def extract_first_frame(self):
        """
        Get the first frame of the video for Area of Interest selection
        """
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
        """
        Open the area selector dialog to select an area of interest
        """
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
        #self.area_selector.area_selected.connect(self.on_area_selected)
        self.area_selector.corners_selected.connect(self.on_area_selected)
        self.area_selector.exec()
        
    #def on_area_selected(self, area):
    def on_area_selected(self, x1, y1, x2, y2):
        """
        Handle the area selected by the user

        Args:
            area: The selected area of interest
        """
        # Store the selected area
        self.aoi = QRect(x1, y1, x2, y2)
        self.area_label.setText(f"Area of Interest: ({x1}, {y1}) ({x2}, {y2})")
        # Enable the reset button
        self.reset_area_button.setEnabled(True)

    def reset_area_of_interest(self):
        """
        Reset the selected area of interest
        """
        if hasattr(self, 'aoi'):
            delattr(self, 'aoi')
        self.area_label.setText("Area of Interest: Not Selected")
        self.reset_area_button.setEnabled(False)
        self.aoi = None

    def handle_video_info_error(self, error_message):
        """
        Handle errors during video information retrieval.
        Args:
            error_message: Error message from the worker
        """
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
        """
        Update the interval entry based on the selected tracker.
        
        Args:
            tracker: The selected tracker
        """
        if tracker in ["yolo", "xclip-32", "siglip"]:
            self.interval_entry.setValue(30)
        else:
            self.interval_entry.setValue(10)

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
            
        if progress_bar and section != 'video_processing':
            progress_bar.setVisible(show)
            if show:
                progress_bar.setRange(0, 0)  # Indeterminate progress
            else:
                progress_bar.setRange(0, 100)
                
        if button:
            button.setEnabled(not show)

    def start_video_processing(self):
        """
        Start the video processing worker to process the video.
        """
        # Start processing worker
        self.processing_worker = VideoProcessingWorker(
            self,
            video_path=self.video_path,
            database_path=self.database_path,
            output_dir=self.output_dir,
            interval=int(self.interval_entry.value()),
            tracker=self.tracker_combo.currentText()
        )
        self.processing_worker.error.connect(self.handle_processing_error)
        self.processing_worker.progress.connect(self.update_processing_progress)
        self.processing_worker.finished.connect(self.handle_processing_finished)
        self.processing_worker.start()
        # Update UI
        self.processing_progress.setValue(0)
        self.processing_progress.setVisible(True)
        self.process_button.setVisible(False)
        self.cancel_button.setVisible(True)

    def cancel_video_processing(self):
        """
        Cancel the video processing worker if it is running and the cancel
        button is clicked.
        """
        if hasattr(self, "processing_worker") and self.processing_worker.isRunning():
            # Don't force terminate the thread, just set the flag to stop
            self.processing_worker.terminate_processing = True
            self.status_message.setText("Canceling processing...")
            self.reset_processing_ui()

    def update_processing_progress(self, message, percentage):
        """
        Update the processing status message in the UI

        Args:
            message: The status message to display
            percentage: The percentage of completion
        """
        if hasattr(self, 'status_message'):
            self.status_message.setText(f"{message}")
        if hasattr(self, 'processing_progress'):
            self.processing_progress.setValue(percentage)

    def handle_processing_error(self, error_message):
        """
        Handle errors during video processing.

        Args:
            error_message: Error message from the worker
        """
        print(f"Error processing video: {error_message}")
        self.show_loading('video_processing', False)
        self.reset_processing_ui()

    def handle_processing_finished(self):
        """
        Handle the completion of video processing.
        """
        self.reset_processing_ui()

    def reset_processing_ui(self):
        """
        Reset UI elements after processing completes or is cancelled
        """
        if hasattr(self, 'processing_progress'):
            self.processing_progress.setVisible(False)
        if hasattr(self, 'process_button'):
            self.process_button.setVisible(True)
        if hasattr(self, 'cancel_button'):
            self.cancel_button.setVisible(False)

    def eventFilter(self, obj, event):
        """
        Event filter to handle resize events and scroll events in the results section.

        Args:
            obj: The object that received the event
            event: The event that occurred
        """
        # Respond to resize events
        if obj == self.scroll_area.viewport() and event.type() == QEvent.Type.Resize:
            # Do not resize previous results while the new video is being selected
            # This is necessary to avoid resizing issues when the video is being loaded
            # as the previous results are being cleared
            if not hasattr(self, 'is_changing_video') or not self.is_changing_video:
                # safety check to make sure frame containers exist
                if self.findChildren(QFrame, "result_frame"):
                    self.adjust_image_sizes()
            return False
        
        # For dynamic loading - detect when scrolling nears the bottom
        elif obj == self.scroll_area.viewport() and event.type() == QEvent.Type.Wheel:
            # Check if we're near the bottom of the scroll area
            scrollbar = self.scroll_area.verticalScrollBar()
            
            # Only trigger load if we're near the bottom AND we have more to load
            more_to_load = self.current_batch * self.batch_size < len(self.all_results)
            near_bottom = scrollbar.value() > scrollbar.maximum() - 200
            
            if more_to_load and near_bottom:
                # Throttle loading to prevent multiple calls
                current_time = time.time()
                if not hasattr(self, '_last_load_time') or current_time - self._last_load_time > 0.5:
                    self._last_load_time = current_time
                    print(f"Near bottom, loading next batch. Scrollbar: {scrollbar.value()}/{scrollbar.maximum()}")
                    self.load_next_batch()
            
            return False
        
        return super().eventFilter(obj, event)

    def adjust_image_sizes(self):
        """
        Adjust the sizes of the images in the results section based on the 
        current viewport size. This method is called when the window is resized or when the
        scroll area is resized.
        """
        # Calculate new target width
        available_width = self.scroll_area.viewport().width()
        effective_width = available_width - 5
        target_width = int(effective_width / 2)

        # Check if there are any loaded images
        if not hasattr(self, 'loaded_images') or self.loaded_images is None or len(self.loaded_images) == 0:
            print("No loaded images to resize.")
            return
        # Check if target width is valid
        if target_width <= 0:
            print("Invalid target width for resizing.")
            return
        
        # Resize all currently displayed images
        for frame_path, metadata, frame_container in self.loaded_images:
            # Find the image label in the container
            for child in frame_container.children():
                if isinstance(child, QLabel) and hasattr(child, 'pixmap') and child.pixmap() is not None:
                    pixmap = self.load_frame_image(frame_path, target_width)
                    child.setPixmap(pixmap.copy())
                    break

    def start_query(self):
        """
        Start the query worker to process the query and display results.
        """
        self.show_loading('query', True)

        # Convert the area of interest to a tuple and add margin to it
        if hasattr(self, 'aoi') and self.aoi:
            # Margin 20 pixels TODO: Make this configurable
            aoi = (self.aoi.x(), self.aoi.y(), self.aoi.width(), self.aoi.height(), 20)
        else:
            aoi = None
        
        # Start query worker
        self.query_worker = QueryWorker(self, 
                                        query=self.query_entry.text(), 
                                        area_of_interest=aoi)
        self.query_worker.finished.connect(self.display_query_results)
        self.query_worker.error.connect(self.handle_query_error)
        self.query_worker.feedback.connect(self.update_query_feedback)
        # Pass deduplication option to the query worker
        self.query_worker.deduplicate = self.deduplicate_frames_value
        self.query_worker.start()

    def update_query_feedback(self, message):
        """
        Update the query feedback message in the UI.

        Args:
            message: The feedback message to display
        """
        if hasattr(self, 'query_feedback'):
            self.query_feedback.setText(message)

    def display_query_results(self, results, metadata=None):
        """
        Display the results of the query in the results section.
        
        Args:
            results: The results of the query
            metadata: Metadata associated with the results
        """
        print(f"Displaying {len(results)} results")

        # Ensure results is not None and has a valid length
        result_count = len(results) if results else 0
        print(f"Results count: {result_count}")
        # the updates must be done in the main thread
        # otherwise the text won't update properly
        QMetaObject.invokeMethod(self.results_count_label, "setText", 
                                Qt.ConnectionType.QueuedConnection, 
                                Q_ARG(str, f"Number of frames: {result_count}"))
        
        # Store all results
        self.all_results = list(results.items())
        self.all_metadata = metadata
        self.loaded_images = []
        self.current_batch = 0
        
        # Clear existing results first
        self.clear_results_layout()
        
        # Only load the first batch initially
        self.load_next_batch()
        
        self.show_loading('query', False)

    def clear_results_layout(self):
        """
        Clear the results layout and remove all widgets.
        """
        # Clear existing results
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def load_next_batch(self):
        """
        Load the next batch of results into the results section.
        This method is called when the user scrolls to the bottom of the 
        results section.
        """
        # If no more results to load, return
        if self.current_batch * self.batch_size >= len(self.all_results):
            print(f"All batches loaded. Current batch: {self.current_batch}, Total items: {len(self.all_results)}")
            return
        
        print(f"Loading batch {self.current_batch}")
        
        # Calculate available width for columns
        max_cols = 2
        available_width = self.scroll_area.viewport().width()
        # Subtract the margin from the width
        effective_width = available_width - 80
        target_width = int(effective_width / max_cols)
        
        # Get the starting and ending indices for this batch
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.all_results))
        
        # Get items to process in this batch
        batch_items = self.all_results[start_idx:end_idx]
        print(f"Batch items: {batch_items}")
        
        # Process each item in the batch
        for i, (frame_path, metadata) in enumerate(batch_items):
            # Calculate position
            absolute_idx = start_idx + i
            row = absolute_idx // max_cols
            col = absolute_idx % max_cols
            
            print(f"Adding image at position ({row}, {col}): {frame_path}")
            
            # Ensure frame_path is a string and exists
            if isinstance(frame_path, int):
                frame_path = os.path.join(self.found_frames_dir, f"frame_{frame_path:06d}.jpg")
            
            if os.path.exists(frame_path):
                try:
                    # Create widget for this frame
                    frame_widget = self.create_frame_widget(frame_path, metadata, target_width)
                    
                    # Add to grid layout at the calculated position
                    self.results_layout.addWidget(frame_widget, row, col)
                    
                    # Store reference for resizing
                    self.loaded_images.append((frame_path, metadata, frame_widget))
                except Exception as e:
                    print(f"Error creating widget for {frame_path}: {str(e)}")
            else:
                print(f"Frame not found: {frame_path}")
        
        # Increment batch counter
        self.current_batch += 1

    def create_frame_widget(self, frame_path, metadata, target_width):
        """
        Create a widget for a single frame.

        Args:
            frame_path: Path to the frame image
            metadata: Metadata associated with the frame
            target_width: Target width for resizing the image
        """

        # Create a frame container
        frame_container = QFrame()
        frame_container.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(frame_container)

        # Load and display the image
        pixmap = self.load_frame_image(frame_path, target_width)
        if pixmap.isNull():
            print(f"Error: Pixmap is null for {frame_path}")
            return frame_container

        image_label = QLabel()
        # Create a copy of the pixmap to prevent memory issues and overwrites - if not copied, all previous images will be overwritten by the last one
        image_label.setPixmap(pixmap.copy())
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Don't scale contents automatically
        image_label.setScaledContents(False)

        # Make the image clickable
        image_label.setCursor(Qt.CursorShape.PointingHandCursor)
        image_label.mousePressEvent = lambda event: self.open_video_at_frame(frame_path, metadata)

        layout.addWidget(image_label)
        return frame_container

    def load_frame_image(self, frame_path, target_width):
        """
        Load an image from the given path and resize it to the target width.
        This method uses PIL to load and resize the image, then converts it to QPixmap.
        This is done to avoid issues with QPixmap loading large images directly.

        Args:
            frame_path: Path to the image file
            target_width: Target width for resizing the image
        Returns:
            QPixmap: Resized image as QPixmap
        """
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
            # Return a blank or error pixmap with fixed height
            return QPixmap((target_width), 100)

    def open_video_at_frame(self, frame_path, metadata):
        """
        Open frame slideshow starting from the selected frame

        Args:
            frame_path: Path to the selected frame
            metadata: Metadata associated with the selected frame
        """
        # Determine the directory containing the frames
        frame_dir = os.path.dirname(frame_path)
        
        # If frame_dir is empty, use the default found_frames directory
        if not frame_dir:
            frame_dir = self.found_frames_dir

        if self.tracker_combo.currentText() == "bytetrack":
            extracted_frames_dir = os.path.join(self.output_dir, "extracted_frames_b") 
        else:
            extracted_frames_dir = os.path.join(self.output_dir, "extracted_frames_yx")
        
        # Create and show slideshow window
        self.slideshow_window = FrameSlideshow(self, 
                                               starting_frame_path=frame_path, 
                                               frame_dir=frame_dir, 
                                               extracted_frames_dir=extracted_frames_dir, 
                                               context_frames=15, 
                                               forward_frames=30, 
                                               interval=500, 
                                               fps=int(self.video_info['fps']), 
                                               frame_interval=self.interval_entry.value(),
                                               input_video_path=self.video_path,
                                               database_path=self.database_path,
                                               tracker=self.tracker_combo.currentText(),
                                               metadata=self.all_metadata)
        self.slideshow_window.show()

    def handle_query_error(self, error_message):
        """
        Handle errors during query processing.

        Args:
            error_message: Error message from the worker
        """
        print(f"Error running query: {error_message}")
        self.show_loading('query', False)

    def closeEvent(self, event):
        """
        Handle application close event.
        This method is called when the user closes the application window.
        It ensures that any running workers are stopped gracefully.

        Args:
            event: The close event
        """
        # Stop any running workers
        if hasattr(self, 'video_info_worker') and self.video_info_worker.isRunning():
            self.video_info_worker.terminate()
        if hasattr(self, 'processing_worker') and self.processing_worker.isRunning():
            self.processing_worker.terminate()
        if hasattr(self, 'query_worker') and self.query_worker.isRunning():
            self.query_worker.terminate()
        
        event.accept()
