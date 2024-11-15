import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import os
import threading
from processors.video_processor import VideoProcessor
from parsers.detection_parser import DetectionParser

class VideoProcessingApp:
    def __init__(self):
        self.video_path = ""
        self.output_dir = os.getcwd()
        self.interval = 30
        self.tracker = "bytetrack"
        self.query = ""
        self.log_results = []
        self.found_frames_dir = ""

        # GUI colors
        self.bg_color = "#2E2E2E"  # Dark gray background
        self.fg_color = "#D3D3D3"  # Light gray text
        self.button_bg_color = "#4A4A4A"  # Dark button background
        self.button_fg_color = "#FFFFFF"  # White button text

        self.root = tk.Tk()
        self.root.title("Video Processing Application")
        self.root.geometry("1200x860")
        self.root.configure(bg=self.bg_color)


    def setup_gui(self):
        """Sets up the main GUI layout."""
        # Top Frame for Video Selection and Query
        top_frame = tk.Frame(self.root, padx=10, pady=10, bg=self.bg_color)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        # Video Selection Section
        video_frame = tk.LabelFrame(top_frame, text="Video Selection", padx=10, pady=10, bg=self.bg_color, fg=self.fg_color)
        video_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

        tk.Label(video_frame, text="Video Path:", bg=self.bg_color, fg=self.fg_color).grid(row=0, column=0, sticky="w")
        self.video_path_entry = tk.Entry(video_frame, width=60, bg="#3C3C3C", fg=self.fg_color)
        self.video_path_entry.grid(row=1, column=0)
        tk.Button(video_frame, text="Browse Video", command=self.select_video, bg=self.button_bg_color, fg=self.button_fg_color).grid(row=1, column=1)

        tk.Label(video_frame, text="Output Directory:", bg=self.bg_color, fg=self.fg_color).grid(row=2, column=0, sticky="w")
        self.output_dir_entry = tk.Entry(video_frame, width=60, bg="#3C3C3C", fg=self.fg_color)
        self.output_dir_entry.grid(row=3, column=0)
        tk.Button(video_frame, text="Browse Directory", command=self.select_output_dir, bg=self.button_bg_color, fg=self.button_fg_color).grid(row=3, column=1)

        # Query and Settings Section
        settings_frame = tk.LabelFrame(top_frame, text="Query and Settings", padx=10, pady=10, bg=self.bg_color, fg=self.fg_color)
        settings_frame.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(settings_frame, text="Search Query:", bg=self.bg_color, fg=self.fg_color).grid(row=0, column=0, sticky="w")
        self.query_entry = tk.Entry(settings_frame, width=40, bg="#3C3C3C", fg=self.fg_color)
        self.query_entry.grid(row=1, column=0)

        tk.Label(settings_frame, text="Frame Interval:", bg=self.bg_color, fg=self.fg_color).grid(row=2, column=0, sticky="w")
        self.interval_entry = tk.Entry(settings_frame, width=40, bg="#3C3C3C", fg=self.fg_color)
        self.interval_entry.insert(0, "30")  # Default value
        self.interval_entry.grid(row=3, column=0)

        tk.Label(settings_frame, text="Tracker:", bg=self.bg_color, fg=self.fg_color).grid(row=4, column=0, sticky="w")
        self.tracker_combobox = ttk.Combobox(settings_frame, values=["bytetrack", "deepsort"], state="readonly", 
                                              background="#3C3C3C", foreground=self.bg_color)
        self.tracker_combobox.set("bytetrack")
        self.tracker_combobox.grid(row=5, column=0)

        self.process_button = tk.Button(settings_frame, text="Process Video", command=self.start_video_processing, bg=self.button_bg_color, fg=self.button_fg_color)
        self.process_button.grid(row=6, column=0)

        self.query_button = tk.Button(settings_frame, text="Run Query", command=self.start_query, bg=self.button_bg_color, fg=self.button_fg_color)
        self.query_button.grid(row=6, column=1)

        # Add a loading indicator
        self.loading_label = tk.Label(self.root, text="Processing...", bg=self.bg_color, fg=self.fg_color)
        self.loading_label.grid(row=1, column=0, padx=10, pady=10)
        self.loading_label.grid_forget()  # Hide initially

        # Main Results Section (Left Center)
        results_frame = tk.Frame(self.root, bg=self.bg_color)
        results_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Main Frame Display Section (left-center, scrollable 2xN grid for frames)
        self.canvas_frame = tk.Frame(results_frame, bg=self.bg_color)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Create the canvas and vertical scrollbar
        self.canvas = tk.Canvas(self.canvas_frame, bg=self.bg_color, height=500, width=1000)
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
        tk.Button(results_frame, text="Show Log", command=self.show_log, bg=self.button_bg_color, fg=self.button_fg_color).grid(row=2, column=0, pady=10)

        # Set row and column weights for resizing
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)


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

    def start_video_processing(self):
        """Disable process button and show loading during video processing."""
        # Disable the buttons during processing
        self.process_button.config(state=tk.DISABLED)
        self.loading_label.grid(row=1, column=0, padx=10, pady=10)  # Show loading indicator

        # Run video processing in a separate thread to avoid freezing the GUI
        video_thread = threading.Thread(target=self.process_video)
        video_thread.start()

    def process_video(self):
        """Process the video with VideoProcessor."""
        try:
            processor = VideoProcessor(
                video_path=self.video_path,
                output_dir=self.output_dir,
                interval=int(self.interval_entry.get()),
                tracker_arg=self.tracker_combobox.get()
            )
            processor.process_video()
            self.log_results = processor.initial_yolo_results_log
        except Exception as e:
            print(f"Error processing video: {e}")

        # Re-enable buttons and hide loading indicator after processing
        self.process_button.config(state=tk.NORMAL)
        self.loading_label.grid_forget()

    def start_query(self):
        """Disable the buttons and show loading during query processing."""
        if not self.log_results:
            # No results to query
            return 
        
        # Disable search query button
        self.query_button.config(state=tk.DISABLED)
        self.loading_label.config(text="Parsing Query...")
        self.loading_label.grid(row=1, column=0, padx=10, pady=10)

        # Run query parsing in a separate thread to avoid freezing the GUI
        query_thread = threading.Thread(target=self.run_query)
        query_thread.start()

    def run_query(self):
        """Run query on parsed results."""
        if not self.log_results:
            return  # No results to query

        self.query = self.query_entry.get()
        try:
            log_parser = DetectionParser(
                log_entries=self.log_results,
                query=self.query,
                output_dir=self.output_dir
            )
            log_parser.parse_log_entries()
            print(log_parser.found_log_entries)
            # After query, load and display frames in grid
            self.display_frames_in_grid(log_parser.found_log_entries)
        except Exception as e:
            print(f"Error running query: {e}")

        # Re-enable buttons and hide loading indicator after processing
        self.query_button.config(state=tk.NORMAL)
        self.loading_label.grid_forget()

    def display_frames_in_grid(self, results):
        """Load frames from found_frames_dir and display in a 2x2 scrollable grid."""
        # Get the list of images from the found frames directory
        image_files = sorted(os.listdir(self.found_frames_dir))  # Get all frame images

        # Clear the canvas before adding new images
        for widget in self.frame_in_canvas.winfo_children():
            widget.destroy()

        # Set up the grid on the canvas
        row = 0
        col = 0

        # Number of images to show per row and column (2x2 grid)
        images_per_row = 2
        images_per_column = 2
        total_images = len(image_files)
        
        # Calculate the number of rows needed based on the number of images
        rows_needed = (total_images + images_per_row - 1) // images_per_row  # Ceiling division

        # Set the canvas height to only show 2 rows at a time
        self.canvas.config(height=500)

        # Add all images in a 2xN grid, fitting in the scrollable area
        for image_file in image_files:
            image_path = os.path.join(self.found_frames_dir, image_file)
            image = Image.open(image_path)
            image = image.resize((480, 320))  # Resize the image to fit in the grid
            photo = ImageTk.PhotoImage(image)

            # Create a Label for each image
            label = tk.Label(self.frame_in_canvas, image=photo, bg="#3C3C3C")
            label.image = photo  # Keep a reference to avoid garbage collection

            # Position the image in the 2x2 grid
            label.grid(row=row, column=col, padx=5, pady=5)

            # Update grid position for the next image
            col += 1
            if col == images_per_row:  # Move to the next row after 2 columns
                col = 0
                row += 1

        # Update the scroll region to make the canvas scrollable
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def show_log(self):
        """Open the secondary results section in a new window."""
        # Create a new window for the log
        log_window = tk.Toplevel(self.root)
        log_window.title("Detection Log")
        log_window.configure(bg=self.bg_color)

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

        # Set row and column weights for resizing the new window
        log_window.grid_rowconfigure(0, weight=1)
        log_window.grid_columnconfigure(0, weight=1)

    def display_results_in_treeview(self, results):
        """Display results in Treeview widget."""
        for result in results:
            self.results_tree.insert("", "end", values=(result['frame'], result['class_name'], result['confidence'], result['bounding_box'], result['dominant_color']))

    def display_detection_log(self, results):
        """Display detection log in the text widget."""
        self.detection_log_text.delete(1.0, tk.END)
        print(results)
        for result in results:
            self.detection_log_text.insert(tk.END, f"Frame: {result['frame']}, {result['class_name']} - Confidence: {result['confidence']}\n")

    def run(self):
        """Start the Tkinter mainloop."""
        self.setup_gui()
        self.root.mainloop()

# Running the app
if __name__ == "__main__":
    app = VideoProcessingApp()
    app.run()
