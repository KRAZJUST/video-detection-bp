import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
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

        # GUI colors
        self.bg_color = "#2E2E2E"  # Dark gray background
        self.fg_color = "#D3D3D3"  # Light gray text
        self.button_bg_color = "#4A4A4A"  # Dark button background
        self.button_fg_color = "#FFFFFF"  # White button text

        self.root = tk.Tk()
        self.root.title("Video Processing Application")
        self.root.geometry("1400x800")
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
        self.video_path_entry = tk.Entry(video_frame, width=40, bg="#3C3C3C", fg=self.fg_color)
        self.video_path_entry.grid(row=1, column=0)
        tk.Button(video_frame, text="Browse Video", command=self.select_video, bg=self.button_bg_color, fg=self.button_fg_color).grid(row=1, column=1)

        tk.Label(video_frame, text="Output Directory:", bg=self.bg_color, fg=self.fg_color).grid(row=2, column=0, sticky="w")
        self.output_dir_entry = tk.Entry(video_frame, width=40, bg="#3C3C3C", fg=self.fg_color)
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

        tk.Button(settings_frame, text="Start Processing", command=self.start_processing, bg=self.button_bg_color, fg=self.button_fg_color).grid(row=6, column=0)
        tk.Button(settings_frame, text="Run Query", command=self.run_query, bg=self.button_bg_color, fg=self.button_fg_color).grid(row=6, column=1)

        # Main Results Section (Left Center)
        results_frame = tk.Frame(self.root, bg=self.bg_color)
        results_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Main Frame Display Section (left-center)
        main_results_frame = tk.Frame(results_frame, bg=self.bg_color)
        main_results_frame.grid(row=0, column=0, sticky="nsew")

        # Secondary Results Section for Detection Log (Right side)
        secondary_results_frame = tk.Frame(results_frame, bg=self.bg_color)
        secondary_results_frame.grid(row=0, column=1, padx=10, sticky="nsew")

        # Create the Treeview for Frame Results
        self.results_tree = ttk.Treeview(main_results_frame, columns=("Frame", "Class Name", "Confidence", "Bounding Box", "Dominant Color"), show="headings")
        self.results_tree.heading("Frame", text="Frame")
        self.results_tree.heading("Class Name", text="Class Name")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.heading("Bounding Box", text="Bounding Box")
        self.results_tree.heading("Dominant Color", text="Dominant Color")
        self.results_tree.grid(row=0, column=0, sticky="nsew")

        # Add a vertical scrollbar to the main results section
        scrollbar = ttk.Scrollbar(main_results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Secondary Text Widget for Detection Log
        self.detection_log_text = tk.Text(secondary_results_frame, width=40, height=20, bg="#3C3C3C", fg=self.fg_color)
        self.detection_log_text.grid(row=0, column=0)

        # Set row and column weights for resizing
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=2)
        results_frame.grid_columnconfigure(1, weight=1)


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
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, dir_path)

    def start_processing(self):
        """Start processing video in a separate thread."""
        if not self.video_path:
            return  # No video selected, do nothing

        thread = threading.Thread(target=self.process_video)
        thread.start()

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
            parsed_results = log_parser.parse_log_entries()
            self.display_results_in_treeview(parsed_results)
            #self.display_detection_log(parsed_results)
        except Exception as e:
            print(f"Error running query: {e}")

    def display_results_in_treeview(self, results):
        """Display results in Treeview widget."""
        for result in results:
            self.results_tree.insert("", "end", values=(result['frame'], result['class_name'], result['confidence'], result['bounding_box'], result['dominant_color']))

    def display_detection_log(self, results):
        """Display detection log in the text widget."""
        self.detection_log_text.delete(1.0, tk.END)
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
