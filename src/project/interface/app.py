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

class VideoProcessingApp:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.video_path = ""
        self.output_dir = os.getcwd()
        self.interval = 30
        self.tracker = "bytetrack"
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

        # Container for dynamically added comboboxes
        self.combobox_rows = []

    def setup_gui(self):
        """Sets up the main GUI layout."""
        # Create a top-level frame to hold the three sections
        top_frame = tk.Frame(self.root, bg=self.bg_color)
        top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Configure columns for the three sections
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=1)
        top_frame.grid_columnconfigure(2, weight=1)

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
        self.tracker_combobox = ttk.Combobox(settings_frame, values=["-", "bytetrack", "xclip"], state="readonly")
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

        # === Results section ===
        results_frame = tk.Frame(self.root, bg=self.bg_color)
        results_frame.grid(row=1, column=0, padx=10, pady=3, sticky="nsew")

        # Main Frame Display Section (left-center, scrollable 2xN grid for frames)
        self.canvas_frame = tk.Frame(results_frame, bg=self.bg_color)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Create the canvas and vertical scrollbar
        self.canvas = tk.Canvas(self.canvas_frame, bg=self.bg_color, height=600, width=1250)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Make the canvas scrollable
        self.canvas.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Create a frame inside the canvas to hold the 2xN grid
        self.frame_in_canvas = tk.Frame(self.canvas, bg=self.bg_color)
        self.canvas.create_window((0, 0), window=self.frame_in_canvas, anchor="nw")

        # Set row and column weights for resizing
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        # Add a Button to show the log in a new window
        self.show_log_button = tk.Button(results_frame, text="Show Log", command=self.show_log, bg=self.button_bg_color, fg=self.button_fg_color)
        self.show_log_button.grid(row=0, column=2, pady=10, sticky="e")

        # Set row and column weights for resizing
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

    def update_interval_entry(self, event):
        """Update the interval entry based on the selected tracker."""
        selected_tracker = self.tracker_combobox.get()
        if selected_tracker == "-":
            self.interval_entry.delete(0, tk.END)
            self.interval_entry.insert(0, "30")
        elif selected_tracker == "xclip":
            self.interval_entry.delete(0, tk.END)
            self.interval_entry.insert(0, "20")
        else:
            self.interval_entry.delete(0, tk.END)
            self.interval_entry.insert(0, "10")

    def select_video(self):
        """Open file dialog to select video."""
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
        if file_path:
            self.video_path = file_path
            self.video_path_entry.delete(0, tk.END)
            self.video_path_entry.insert(0, file_path)

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
        if tracker in ["-", "bytetrack"]:
            self.segmentation_checkbox.grid()
        else:
            self.segmentation_checkbox.grid_remove()

    def start_video_processing(self):
        """Disable process button and show loading during video processing."""
        # Disable the buttons during processing
        self.process_button.config(state=tk.DISABLED)

        # Run video processing in a separate thread to avoid freezing the GUI
        video_thread = threading.Thread(target=self.process_video)
        video_thread.start()

    def process_video(self):
        """Process the video with VideoProcessor."""
        try:
            processor = VideoProcessor(
                video_path=self.video_path,
                database_path=self.database_path,
                output_dir=self.output_dir,
                interval=int(self.interval_entry.get()),
                tracker_arg=self.tracker_combobox.get()
            )
            processor.process_video()
            
            # Get the correct log results if the tracker was selected or not
            if self.tracker_combobox.get() == "-":
                self.log_results = processor.initial_yolo_results_log
            elif self.tracker_combobox.get() == "bytetrack":
                self.log_results = processor.log_entries
            elif self.tracker_combobox.get() == "xclip":
                self.temp_embeddings = processor.temp_embeddings

        except Exception as e:
            print(f"Error processing video: {e}")

        # Re-enable buttons and hide loading indicator after processing
        self.process_button.config(state=tk.NORMAL)

    def start_query(self):
        """Start the query process when the button is clicked and disable the button."""
        
        # Disable search query button
        self.query_button.config(state=tk.DISABLED)

        # Run query parsing in a separate thread to avoid freezing the GUI
        query_thread = threading.Thread(target=self.run_query)
        query_thread.start()

    def run_query(self):
        """Run query on parsed results."""
        
        # Disable the "Show Log" button while processing query
        self.show_log_button.config(state=tk.DISABLED)
        self.query = self.query_entry.get()

        try:
            if self.tracker_combobox.get() == "-" or self.tracker_combobox.get() == "bytetrack":
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
                
                # Save the found log entries for showing in the log window
                self.found_log_entries = log_parser.found_log_entries
            
            elif self.tracker_combobox.get() == "xclip":
                print('Running XClip query...')
                # XClip query processing
                xclip_parser = XClipParser(
                    video_path=self.video_path,
                    query=self.query,
                    output_dir=self.output_dir,
                )
                print('Getting query embeddings...')
                similarities, metadata = xclip_parser.search_embeddings(top_k=3)
                print(f"Similarities: {similarities}")
                print(f"Metadata: {metadata}")
                self.display_frames_in_grid(xclip_parser.top_frames)

        except Exception as e:
            print(f"Error running query: {e}")
            traceback.print_exc()

        # Re-enable buttons and hide loading indicator after processing
        self.query_button.config(state=tk.NORMAL)

        # Re-enable the "Show Log" button after query is complete
        self.show_log_button.config(state=tk.NORMAL)

    def clear_canvas(self):
        """Remove all widgets from the canvas."""
        for widget in self.frame_in_canvas.winfo_children():
            widget.destroy()


    def display_frames_in_grid(self, results):
        """Load frames from found_frames_dir and display in a 2x2 scrollable grid."""
        # Clear the canvas before adding new images
        self.clear_canvas()

        # Get the list of images from the found frames directory
        image_files = sorted(os.listdir(self.found_frames_dir))

        # Set up the grid on the canvas
        row = 0
        col = 0

        # Number of images to show per row and column (2x2 grid)
        images_per_row = 2
        images_per_column = 2
        total_images = len(image_files)
        
        # Calculate the number of rows needed based on the number of images
        rows_needed = (total_images + images_per_row - 1) // images_per_row

        # Set the canvas height to only show 2 rows at a time
        self.canvas.config(height=600)

        # Add all images in a 2xN grid, fitting in the scrollable area
        for image_file in image_files:
            image_path = os.path.join(self.found_frames_dir, image_file)
            image = Image.open(image_path)
            # Resize the image to fit in the grid
            image = image.resize((610, 450))
            photo = ImageTk.PhotoImage(image)

            # Create a Label for each image
            label = tk.Label(self.frame_in_canvas, image=photo, bg="#3C3C3C")
            # Keep a reference to avoid garbage collection
            label.image = photo

            # Position the image in the 2x2 grid
            label.grid(row=row, column=col, padx=5, pady=5)

            # Update grid position for the next image
            col += 1
            # Move to the next row after 2 columns
            if col == images_per_row:
                col = 0
                row += 1

        # Force the canvas to update the scroll region
        self.frame_in_canvas.update_idletasks()

        # Update the scroll region to make the canvas scrollable
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def show_log(self):
        """Open the secondary results section in a new window."""
        if not hasattr(self, 'found_log_entries') or not self.found_log_entries:
            print("No log entries to display.")
            # If no entries were found, don't open the log window
            return

        # Create a new window for the log
        log_window = tk.Toplevel(self.root)
        log_window.title("Detection Log")
        log_window.configure(bg=self.bg_color)
        log_window.geometry("1100x400")

        # Create a frame to hold the log components
        secondary_results_frame = tk.Frame(log_window, bg=self.bg_color)
        secondary_results_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create the Treeview for Frame Results in the new window
        self.results_tree = ttk.Treeview(secondary_results_frame, columns=("Frame", "Class Name", "Confidence", "Bounding Box", "Dominant Color"), show="headings")

        # Style the Treeview
        style = ttk.Style()
        style.configure("Treeview",
                        background="#2E2E2E",  
                        foreground="#FFFFFF",  
                        fieldbackground="#2E2E2E")
        style.configure("Treeview.Heading",
                        background="#444444",
                        foreground="#FFFFFF")
        
        self.results_tree.heading("Frame", text="Frame")
        self.results_tree.heading("Class Name", text="Class Name")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.heading("Bounding Box", text="Bounding Box")
        self.results_tree.heading("Dominant Color", text="Dominant Color")
        self.results_tree.grid(row=0, column=0, sticky="nsew")

        # Add a vertical scrollbar to the log window
        scrollbar = ttk.Scrollbar(secondary_results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Insert data into the Treeview from the found log entries
        for frame, entries in self.found_log_entries.items():
            for entry in entries:
                self.results_tree.insert("", "end", values=(
                    frame,
                    entry["class_name"],
                    entry["confidence"],
                    str(entry["bbox"]),
                    entry["dominant_color"]
                ))

        # Set row and column weights for resizing the new window
        log_window.grid_rowconfigure(0, weight=1)
        log_window.grid_columnconfigure(0, weight=1)


    def run(self):
        """Start the Tkinter mainloop."""
        self.setup_gui()
        self.root.mainloop()

# Running the app
if __name__ == "__main__":
    app = VideoProcessingApp()
    app.run()
