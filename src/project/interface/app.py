"""


"""

import tkinter as tk
import traceback
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import os
import threading
from processors.video_processor import VideoProcessor
from parsers.detection_parser import DetectionParser
from xclip.xclip_parser import XClipParser
from .video_info import VideoInfoUtils
import time

class VideoProcessingApp:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.video_path = ""
        self.output_dir = os.getcwd()
        self.interval = 30
        self.query = ""
        self.log_results = []
        self.found_frames_dir = ""
        self.found_log_entries = {}
        self.temp_embeddings = None

        # GUI colors
        self.bg_color = "#2E2E2E"  # Dark gray background
        self.fg_color = "#D3D3D3"  # Light gray text
        self.button_bg_color = "#4A4A4A"  # Dark button background
        self.button_fg_color = "#FFFFFF"  # White button text

        self.root = tk.Tk()
        self.use_segmentation = tk.BooleanVar(value=True)
        self.root.title("Video Processing Application")
        self.root.geometry("1400x900")
        self.root.configure(bg=self.bg_color)
        # Bind window close event to stop processing
        #self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Container for dynamically added comboboxes
        self.combobox_rows = []
        # Stop event for the separate threads
        #self.stop_event = threading.Event()

    def setup_gui(self):
        """Sets up the main GUI layout."""
        # Create a top-level frame to hold the three sections
        top_frame = tk.Frame(self.root, bg=self.bg_color)
        top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Configure columns for the three sections
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=1)
        top_frame.grid_columnconfigure(2, weight=1)
        top_frame.grid_columnconfigure(3, weight=1)

        # === Input and Output Section ===
        io_frame = tk.LabelFrame(top_frame, text="Input & Output", bg=self.bg_color, fg=self.fg_color, padx=10, pady=10)
        io_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Video Path
        tk.Label(io_frame, text="Video Path:", bg=self.bg_color, fg=self.fg_color).grid(row=0, column=0, sticky="w")
        self.video_path_entry = tk.Entry(io_frame, width=50, bg="#3C3C3C", fg=self.fg_color)
        self.video_path_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        tk.Button(io_frame, text="Browse Video", command=self.select_video, bg=self.button_bg_color, fg=self.button_fg_color).grid(row=2, column=0, pady=2, sticky="w")

        # Output Directory
        tk.Label(io_frame, text="Output Directory:", bg=self.bg_color, fg=self.fg_color).grid(row=3, column=0, sticky="w")
        self.output_dir_entry = tk.Entry(io_frame, width=50, bg="#3C3C3C", fg=self.fg_color)
        self.output_dir_entry.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        tk.Button(io_frame, text="Browse Directory", command=self.select_output_dir, bg=self.button_bg_color, fg=self.button_fg_color).grid(row=5, column=0, pady=2, 
                                                                                                                                            sticky="w")

        # === Query Section ===
        query_frame = tk.LabelFrame(top_frame, text="Query Builder", bg=self.bg_color, fg=self.fg_color, padx=10, pady=10)
        query_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Add text input for query
        tk.Label(query_frame, text="Query:", bg=self.bg_color, fg=self.fg_color).grid(row=0, column=0, sticky="w")
        self.query_entry = tk.Entry(query_frame, width=50, bg="#3C3C3C", fg=self.fg_color)
        self.query_entry.grid(row=1, column=0, sticky="ew", pady=5)

        # Search Query Button (bottom-right)
        self.query_button = tk.Button(query_frame, text="Search Query", command=self.start_query, bg=self.button_bg_color, fg=self.button_fg_color)
        self.query_button.grid(row=2, column=0, sticky="s", padx=5, pady=5)

        # Conditional Segmentation Checkbox
        self.segmentation_checkbox = tk.Checkbutton(
            query_frame,
            text="Use Segmentation",
            variable=self.use_segmentation,
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor=self.button_bg_color,
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        )
        self.segmentation_checkbox.grid(row=2, column=0, sticky="se", pady=5, padx=5)

        # Configure Grid Weights for Query Section
        query_frame.grid_rowconfigure(1, weight=1) 
        query_frame.grid_columnconfigure(0, weight=1)  # Query section uses full width

        # === Tracker and Interval Section ===
        settings_frame = tk.LabelFrame(top_frame, text="Settings", bg=self.bg_color, fg=self.fg_color, padx=10, pady=10)
        settings_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        # Tracker Selection
        tk.Label(settings_frame, text="Tracker:", bg=self.bg_color, fg=self.fg_color).grid(row=0, column=0, sticky="w")
        self.tracker_combobox = ttk.Combobox(settings_frame, values=["-", "bytetrack", "deepsort","xclip"], state="readonly")
        self.tracker_combobox.set("-")
        self.tracker_combobox.grid(row=1, column=0, sticky="ew", pady=5)
        self.tracker_combobox.bind("<<ComboboxSelected>>", self.update_interval_entry)

        # Frame Interval
        tk.Label(settings_frame, text="Frame Interval:", bg=self.bg_color, fg=self.fg_color).grid(row=2, column=0, sticky="w")
        self.interval_entry = tk.Entry(settings_frame, width=20, bg="#3C3C3C", fg=self.fg_color)
        self.interval_entry.insert(0, "30")
        self.interval_entry.grid(row=3, column=0, sticky="ew", pady=5)

        # Process Button
        self.process_button = tk.Button(settings_frame, text="Process Video", command=self.start_video_processing, bg=self.button_bg_color, fg=self.button_fg_color)
        self.process_button.grid(row=4, column=0, pady=10, sticky="ew")

        # === Video Information Section ===
        video_info_frame = tk.LabelFrame(top_frame, text="Video Information", bg=self.bg_color, fg=self.fg_color, padx=10, pady=10)
        video_info_frame.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")
        
        self.video_info_label = tk.Label(video_info_frame, text="Video Information:\n", bg=self.bg_color, fg=self.fg_color, justify="left")
        self.video_info_label.grid(row=0, column=0, sticky="w")
        
        # === Results section ===
        results_frame = tk.Frame(self.root, bg=self.bg_color)
        results_frame.grid(row=1, column=0, padx=10, pady=3, sticky="nsew")

        # Configure results_frame to expand
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        # Main Frame Display Section
        self.canvas_frame = tk.Frame(results_frame, bg=self.bg_color)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure canvas_frame to expand
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        # Create the canvas and scrollbar
        self.canvas = tk.Canvas(self.canvas_frame, bg=self.bg_color)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas to hold the grid
        self.frame_in_canvas = tk.Frame(self.canvas, bg=self.bg_color)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.frame_in_canvas, anchor="nw")

        # Bind resize events
        self.root.bind('<Configure>', self.on_window_resize)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        
        # Store the last known canvas size to detect actual size changes
        self.last_canvas_width = 0
        self.last_canvas_height = 0
        
        # Video information section progress bar
        self.video_info_progress = ttk.Progressbar(
            video_info_frame, 
            mode='indeterminate', 
            length=150
        )
        self.video_info_progress.grid(row=1, column=0, sticky="ew", pady=5)
        self.video_info_progress.grid_remove()

        # Processing section progress bar
        self.processing_progress = ttk.Progressbar(
            settings_frame, 
            mode='indeterminate', 
            length=180
        )
        self.processing_progress.grid(row=5, column=0, sticky="ew", pady=5)
        self.processing_progress.grid_remove()

        # Query section progress bar
        self.query_progress = ttk.Progressbar(
            query_frame, 
            mode='indeterminate', 
            length=200
        )
        self.query_progress.grid(row=3, column=0, sticky="ew", pady=5)
        self.query_progress.grid_remove()
        
    def update_interval_entry(self, event):
        """Update the interval entry based on the selected tracker."""
        selected_tracker = self.tracker_combobox.get()
        if selected_tracker == "-":
            self.interval_entry.delete(0, tk.END)
            self.interval_entry.insert(0, "30")
        elif selected_tracker == "xclip":
            self.interval_entry.delete(0, tk.END)
            self.interval_entry.insert(0, "30")
        else:
            self.interval_entry.delete(0, tk.END)
            self.interval_entry.insert(0, "10")

    def show_loading(self, section=None, show=True):
        """
        Show/hide loading indicator for specific sections.
        
        Args:
            section (str): Section to show loading indicator for
            show (bool): Whether to show or hide the loading
        """
        button = None
        
        if section == 'video_info':
            progress_bar = self.video_info_progress
        elif section == 'video_processing':
            progress_bar = self.processing_progress
            button = self.process_button
        elif section == 'query':
            progress_bar = self.query_progress
            button = self.query_button
        else:
            return

        if show:
            progress_bar.grid()
            progress_bar.start()
            if button:
                button.config(state=tk.DISABLED)
        else:
            progress_bar.stop()
            progress_bar.grid_remove()
            if button:
                button.config(state=tk.NORMAL)

    def select_video(self):
        """Open file dialog to select video and display its information in a non-blocking manner."""
        def process_video_info():
            
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=(("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*"))
            )
            if file_path:
                # Show loading indicator
                self.root.after(0, lambda: self.show_loading('video_info', True))
                
                # Update UI elements in main thread
                self.video_path = file_path
                self.video_path_entry.delete(0, tk.END)
                self.video_path_entry.insert(0, file_path)

                # Perform video info processing in a separate thread
                try:
                    video_metadata = VideoInfoUtils.get_video_info(file_path)
                    
                    if video_metadata:
                        # Format video info
                        minutes = int(video_metadata.duration // 60)
                        seconds = int(video_metadata.duration % 60)
                        
                        bitrate_str = "unknown"
                        if video_metadata.bitrate != "unknown":
                            bitrate_mbps = float(video_metadata.bitrate) / 1_000_000
                            bitrate_str = f"{bitrate_mbps:.2f} Mbps"
                        
                        info_text = (
                            f"Video Information:\n"
                            f"Duration: {minutes:02d}:{seconds:02d}\n"
                            f"FPS: {video_metadata.fps}\n"
                            f"Resolution: {video_metadata.width}x{video_metadata.height}\n"
                            f"Total Frames: {video_metadata.frame_count:,}\n"
                            f"Codec: {video_metadata.codec}\n"
                            f"Bitrate: {bitrate_str}\n"
                            f"File Size: {video_metadata.size}"
                        )
                        
                        # .after() to update UI from background thread safely
                        self.root.after(0, lambda: [
                            self.video_info_label.config(text=info_text),
                            self.show_loading('video_info', False)
                        ])
                    else:
                        self.root.after(0, lambda: [
                            self.video_info_label.config(text="Error reading video information"),
                            self.show_loading('video_info', False)
                        ])
                
                except Exception as e:
                    self.root.after(0, lambda: [
                        self.video_info_label.config(text=f"Error: {str(e)}"),
                        self.show_loading('video_info', False)
                    ])

        # Start processing in a separate thread
        threading.Thread(target=process_video_info, daemon=True).start()

    def select_output_dir(self):
        """Open directory dialog to select output directory."""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.found_frames_dir = os.path.join(self.output_dir, "found_frames")
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, dir_path)

    def update_checkbox_visibility(self, event=None):
        """Show or hide the segmentation checkbox based on tracker selection."""
        tracker = self.tracker_combobox.get()
        if tracker in ["-", "bytetrack", "deepsort"]:
            self.segmentation_checkbox.grid()
        else:
            self.segmentation_checkbox.grid_remove()

    def start_video_processing(self):
        """Start the video processing in a separate thread."""
        # Reset stop event
        #self.stop_event.clear()
        # Show loading indicator and disable button while processing
        self.root.after(0, lambda: self.show_loading('video_processing', True))
        # Run video processing in a separate thread to avoid freezing the GUI
        video_thread = threading.Thread(target=self.process_video)
        video_thread.start()

    def process_video(self):
        """Process the video with VideoProcessor."""        
        try:
            start_time = time.time()
            processor = VideoProcessor(
                video_path=self.video_path,
                database_path=self.database_path,
                output_dir=self.output_dir,
                interval=int(self.interval_entry.get()),
                tracker_arg=self.tracker_combobox.get()
            )
            # TODO: maybe add thread termination here if stop_event is set - migh slow down the process though
            processor.process_video()
            print(f'Time taken to process video: {time.time() - start_time}')
            
            # Get the correct log results if the tracker was selected or not
            if self.tracker_combobox.get() == "-":
                self.log_results = processor.initial_yolo_results_log
            elif self.tracker_combobox.get() == "bytetrack" or self.tracker_combobox.get() == "deepsort":
                self.log_results = processor.log_entries
            elif self.tracker_combobox.get() == "xclip":
                self.temp_embeddings = processor.temp_embeddings

        except Exception as e:
            print(f"Error processing video: {e}")
            # Hide loading indicator
            self.root.after(0, lambda: self.show_loading('video_processing', False))

        # Hide loading indicator
        self.root.after(0, lambda: self.show_loading('video_processing', False))

    def start_query(self):
        """Start the query process when the button is clicked and disable the button."""
        # Show loading indicator
        self.root.after(0, lambda: self.show_loading('query', True))

        # Run query parsing in a separate thread to avoid freezing the GUI
        query_thread = threading.Thread(target=self.run_query)
        query_thread.start()

    def run_query(self):
        """Run query on parsed results."""
        self.query = self.query_entry.get()

        try:
            if self.tracker_combobox.get() in ["-", "bytetrack", "deepsort"]:
                log_parser = DetectionParser(
                    log_entries=self.log_results,
                    query=self.query,
                    output_dir=self.output_dir,
                    database_path=self.database_path,
                    tracker=self.tracker_combobox.get(),
                    use_segmentation=self.use_segmentation.get()
                )
                log_parser.parse_detections()

                # After query, load and display frames in grid
                self.display_frames_in_grid(log_parser.found_log_entries)
                
            elif self.tracker_combobox.get() == "xclip":
                print('Running XClip query...')
                # XClip query processing
                xclip_parser = XClipParser(
                    video_path=self.video_path,
                    query=self.query,
                    output_dir=self.output_dir,
                )
                print('Getting query embeddings...')
                similarities, metadata = xclip_parser.search_embeddings(top_k=5)
                print(f"Similarities: {similarities}")
                print(f"Metadata: {metadata}")
                self.display_frames_in_grid(xclip_parser.top_frames)

        except Exception as e:
            print(f"Error running query: {e}")
            self.root.after(0, lambda: self.show_loading('query', False))
            traceback.print_exc()

        # Hide loading indicator
        self.root.after(0, lambda: self.show_loading('query', False))

    def clear_canvas(self):
        """Remove all widgets from the canvas."""
        for widget in self.frame_in_canvas.winfo_children():
            widget.destroy()


    def on_window_resize(self, event):
        """Handle window resize events."""
        if event.widget == self.root:
            # Update canvas size after a short delay
            self.root.after(100, self.update_canvas_size)

    def on_canvas_configure(self, event):
        """Handle canvas configure events."""
        # Check if the canvas size actually changed
        if (event.width != self.last_canvas_width or 
            event.height != self.last_canvas_height):
            
            # Update stored dimensions
            self.last_canvas_width = event.width
            self.last_canvas_height = event.height
            
            # Update the frame_in_canvas width to match the canvas
            self.canvas.itemconfig(self.canvas_window, width=event.width)
            
            # Resize images with the new dimensions
            self.resize_images()

    def update_canvas_size(self):
        """Update the canvas size to fill available space."""
        # Get the available height and width
        top_height = self.root.winfo_children()[0].winfo_height()
        available_height = self.root.winfo_height() - top_height - 40  # 40 for padding
        available_width = self.root.winfo_width() - 40  # 40 for padding
        
        # Update canvas dimensions
        self.canvas.configure(
            height=max(available_height, 100),  # Minimum height of 100
            width=max(available_width, 200)  # Minimum width of 200
        )

    def resize_images(self):
        """Resize images to fit the current canvas width."""
        if not hasattr(self, 'frame_in_canvas') or not self.frame_in_canvas.winfo_children():
            return

        # Get the actual canvas width
        canvas_width = self.canvas.winfo_width()
        
        # Calculate new image dimensions
        padding = 20  # Total horizontal padding between images
        scrollbar_width = self.scrollbar.winfo_width()
        available_width = canvas_width - scrollbar_width - padding
        
        # Calculate image width (2 images per row)
        new_image_width = available_width // 2
        new_image_height = int(new_image_width * 0.75)  # Maintain 4:3 aspect ratio

        # Resize all images in the grid
        for label in self.frame_in_canvas.winfo_children():
            if isinstance(label, tk.Label) and hasattr(label, 'original_image'):
                # Resize the image
                resized_image = label.original_image.resize(
                    (new_image_width, new_image_height),
                    Image.Resampling.LANCZOS
                )
                photo = ImageTk.PhotoImage(resized_image)
                label.configure(image=photo)
                label.image = photo  # Keep a reference

        # Update the scroll region after resizing
        self.frame_in_canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def display_frames_in_grid(self, results):
        """Load frames and display in a 2xN scrollable grid."""
        self.clear_canvas()

        # Get the list of images from the found frames directory
        image_files = sorted(os.listdir(self.found_frames_dir))

        # Initial canvas width and calculate image dimensions
        canvas_width = self.canvas.winfo_width()
        scrollbar_width = self.scrollbar.winfo_width()
        padding = 20
        available_width = canvas_width - scrollbar_width - padding
        
        # Calculate initial image dimensions
        image_width = available_width // 2
        image_height = int(image_width * 0.75)  # Maintain 4:3 aspect ratio

        # Set up the grid
        for idx, image_file in enumerate(image_files):
            row = idx // 2
            col = idx % 2
            
            # Load and resize the image
            image_path = os.path.join(self.found_frames_dir, image_file)
            original_image = Image.open(image_path)
            
            # Store original image for later resizing
            resized_image = original_image.resize(
                (image_width, image_height),
                Image.Resampling.LANCZOS
            )
            photo = ImageTk.PhotoImage(resized_image)

            # Create label and store original image for resizing
            label = tk.Label(self.frame_in_canvas, image=photo, bg="#3C3C3C")
            label.original_image = original_image  # Store original for resizing
            label.image = photo  # Keep a reference
            label.grid(row=row, column=col, padx=5, pady=5)

        # Configure grid columns to be equal width
        self.frame_in_canvas.grid_columnconfigure(0, weight=1)
        self.frame_in_canvas.grid_columnconfigure(1, weight=1)

        # Update the scroll region
        self.frame_in_canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # TODO: use this if needed to stop the video processing
    def on_close(self):
        """Handle GUI close event."""
        print("Closing application... Stopping threads.")
        # Signal thread to stop
        self.stop_event.set()
        if self.video_thread and self.video_thread.is_alive():
            # Wait for thread to finish
            self.video_thread.join()
        # Close the GUI
        self.root.destroy()

    def run(self):
        """Start the Tkinter mainloop."""
        self.setup_gui()
        self.root.mainloop()