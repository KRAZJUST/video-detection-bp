import tkinter as tk
from PIL import Image, ImageTk
import os
from collections import deque

class FrameDisplay:
    def __init__(self, root, canvas, scrollbar, found_frames_dir):
        self.root = root
        self.canvas = canvas
        self.scrollbar = scrollbar
        self.found_frames_dir = found_frames_dir
        
        # Configuration
        self.images_per_row = 2
        # Extra rows to load above and below visible area
        self.buffer_rows = 4
        # Store PhotoImage references by row
        self.photo_references = {}
        # Cache of original images
        self.original_images = {} 
        # Track which rows are currently loaded
        self.current_visible_rows = set()
        # Store all image filenames
        self.all_image_files = []
        self.last_canvas_width = 0
        self.last_canvas_height = 0
        self.total_rows = 0
        
        # Initialize frame in canvas
        self.frame_in_canvas = tk.Frame(self.canvas, bg="#3C3C3C")
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.frame_in_canvas,
            anchor="nw"
        )
        
    def initialize_grid(self):
        """Initialize the grid and prepare for scrolling."""
        self.last_loaded_row = -1  # Track the last row loaded

        # Ensure the directory exists
        if not os.path.exists(self.found_frames_dir):
            os.makedirs(self.found_frames_dir)

        self.all_image_files = sorted([f for f in os.listdir(self.found_frames_dir) 
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
        
        # Clear existing widgets
        for widget in self.frame_in_canvas.winfo_children():
            widget.destroy()

        if not self.all_image_files:
            no_images_label = tk.Label(
                self.frame_in_canvas, 
                text="No images found", 
                bg="#3C3C3C", 
                fg="#D3D3D3"
            )
            no_images_label.pack(expand=True, fill=tk.BOTH)
            return

        self.total_rows = (len(self.all_image_files) + self.images_per_row - 1) // self.images_per_row

        for col in range(self.images_per_row):
            self.frame_in_canvas.grid_columnconfigure(col, weight=1)

        # Create empty labels for all positions
        for row in range(self.total_rows):
            for col in range(self.images_per_row):
                label = tk.Label(self.frame_in_canvas, bg="#3C3C3C")
                label.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

        self.frame_in_canvas.update_idletasks()
        
        # Update scroll region
        self.update_scroll_region()

        # Bind scrolling event to update frames when the canvas scrolls
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.canvas.bind('<MouseWheel>', self.on_mouse_scroll)  # Windows
        self.canvas.bind('<Button-4>', self.on_mouse_scroll)    # Linux up
        self.canvas.bind('<Button-5>', self.on_mouse_scroll)    # Linux down
        self.canvas.bind('<Configure>', lambda e: self.update_scroll_region())
        # Helps detect movement
        self.canvas.bind('<Motion>', lambda e: self.update_visible_frames())

        # Bind to vertical scrolling
        self.canvas.bind('<Configure>', lambda e: self.update_visible_frames())
        # Reset scroll position
        self.canvas.yview_moveto(0)

        # Load the first row initially
        self.update_visible_frames()

    def update_scroll_region(self):
        """Update the canvas scroll region."""
        self.frame_in_canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def get_visible_row_range(self):
        """Calculate which rows are currently visible in the canvas."""
        # Ensure we have a valid total_rows
        if self.total_rows <= 0:
            return 0, 0

        # Get canvas scroll position and dimensions
        try:
            scroll_pos = self.canvas.yview()
            canvas_height = self.canvas.winfo_height()
            total_height = self.frame_in_canvas.winfo_height()
            
            # Prevent division by zero
            if total_height <= 0 or self.total_rows <= 0:
                return 0, 0
            
            # Calculate visible rows
            row_height = total_height / self.total_rows
            start_y = scroll_pos[0] * total_height
            end_y = scroll_pos[1] * total_height
            
            # Convert to row numbers
            start_row = max(0, int(start_y / row_height) - self.buffer_rows)
            end_row = min(self.total_rows - 1, int(end_y / row_height) + self.buffer_rows)
            
            return start_row, end_row
        except Exception as e:
            print(f"Error calculating visible rows: {e}")
            return 0, 0

    def update_visible_frames(self, event=None):
        """Update which frames are visible and load/unload as needed."""

        self.update_scroll_region()
        # Skip if no rows or frames
        if self.total_rows <= 0:
            return

        try:
            start_row, end_row = self.get_visible_row_range()
        
            # Only load frames if scrolling down beyond last loaded row
            for row in range(self.last_loaded_row + 1, end_row + 1):
                if row < self.total_rows:
                    self.load_row(row)
                    self.last_loaded_row = row  # Update last loaded row

        except Exception as e:
            print(f"Error updating visible frames: {e}")

    def load_row(self, row):
        """Load images for a specific row."""
        if row in self.photo_references:
            return
            
        self.photo_references[row] = []
        for col in range(self.images_per_row):
            idx = row * self.images_per_row + col
            if idx >= len(self.all_image_files):
                break
                
            image_file = self.all_image_files[idx]
            image_path = os.path.join(self.found_frames_dir, image_file)
            
            # Load and resize image
            if idx not in self.original_images:
                self.original_images[idx] = Image.open(image_path)
            
            resized_image = self.resize_single_image(self.original_images[idx])
            photo = ImageTk.PhotoImage(resized_image)
            self.photo_references[row].append(photo)
            
            # Update label
            label = self.frame_in_canvas.grid_slaves(row=row, column=col)[0]
            label.configure(image=photo)

    def unload_row(self, row):
        """Unload images for a specific row."""
        if row not in self.photo_references:
            return
            
        for col in range(self.images_per_row):
            slaves = self.frame_in_canvas.grid_slaves(row=row, column=col)
            if slaves:
                label = slaves[0]
                label.configure(image='')
            
        del self.photo_references[row]

    def resize_single_image(self, original_image):
        """Resize a single image to fit the current canvas width more accurately."""
        try:
            canvas_width = self.canvas.winfo_width()
            scrollbar_width = self.scrollbar.winfo_width()
            padding = 20
            
            # Check if canvas width is valid
            if canvas_width <= 0:
                # Default fallback width
                canvas_width = 800
            
            available_width = max(canvas_width - scrollbar_width - padding, 200)
            
            image_width = available_width // self.images_per_row
            # Maintain aspect ratio
            image_height = int(image_width * (original_image.height / original_image.width))
            
            return original_image.resize(
                (image_width, image_height),
                Image.Resampling.LANCZOS
            )
        except Exception as e:
            print(f"Error resizing image: {e}")
            return original_image

    def resize_all_visible_images(self):
        """Resize all currently visible images."""
        for row in self.current_visible_rows:
            if row in self.photo_references:
                self.unload_row(row)
                self.load_row(row)

    def on_canvas_configure(self, event):
        """Handle canvas resize events."""
        if (event.width != self.last_canvas_width or 
            event.height != self.last_canvas_height):
            
            self.last_canvas_width = event.width
            self.last_canvas_height = event.height
            self.canvas.itemconfig(self.canvas_window, width=event.width)
            
            # Resize visible images
            self.resize_all_visible_images()
            
            # Update scroll region
            self.frame_in_canvas.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_mouse_scroll(self, event):
        """Handle mouse scroll events."""
        # Determine scroll direction
        if event.num == 4 or event.delta == 120:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta == -120:
            self.canvas.yview_scroll(1, "units")
        # Trigger frame change event
        self.update_visible_frames()

    def clear_canvas(self):
        """Clear all images and references."""
        for row in list(self.current_visible_rows):
            self.unload_row(row)
        
        self.photo_references.clear()
        self.original_images.clear()
        self.current_visible_rows.clear()
        
        for widget in self.frame_in_canvas.winfo_children():
            widget.destroy()

    def display_frames_in_grid(self, results=None):
        """Initialize and display the frame grid."""
        self.clear_canvas()
        self.initialize_grid()
        self.update_visible_frames()