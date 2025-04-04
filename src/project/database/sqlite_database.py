import sqlite3
from typing import List, Tuple, Any, Dict, Optional
from detectors.detection import Detection

class Database:
    def __init__(self, db_path: str = "detections.db"):
        self.db_path = db_path
        self.connection = None
        self.connect()
        self.create_tables()

    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """Connect to the database."""
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

    def create_tables(self):
        """Create the necessary tables in the database."""
        # New videos table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Modified frames table with video_id reference
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp REAL,
                FOREIGN KEY (video_id) REFERENCES videos (video_id),
                UNIQUE (video_id, frame_number)
            );
        """)

        # Modified detections table to reference frame_id
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER,
                class_name TEXT,
                confidence REAL,
                xmin INTEGER,
                ymin INTEGER,
                xmax INTEGER,
                ymax INTEGER,
                dominant_color TEXT,
                track_id INTEGER DEFAULT NULL,
                direction TEXT DEFAULT NULL,
                FOREIGN KEY (frame_id) REFERENCES frames (frame_id)
            );
        """)

        # Modified refined_detections table to reference frame_id
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS refined_detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER,
                track_id INTEGER,
                class_name TEXT,
                confidence REAL,
                xmin INTEGER,
                ymin INTEGER,
                xmax INTEGER,
                ymax INTEGER,
                dominant_color TEXT,
                direction TEXT,
                FOREIGN KEY (frame_id) REFERENCES frames (frame_id)
            );
        """)

        self.connection.commit()

    def add_video(self, video_name: str) -> int:
        """
        Add a new video to the database or get the ID if it already exists.
        Returns the video_id.
        """
        try:
            self.cursor.execute("""
                INSERT INTO videos (video_name) VALUES (?)
            """, (video_name,))
            self.connection.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            # Video already exists, get its ID
            self.cursor.execute("""
                SELECT video_id FROM videos WHERE video_name = ?
            """, (video_name,))
            return self.cursor.fetchone()[0]

    def get_video_id(self, video_name: str) -> Optional[int]:
        """Get video_id by video_name."""
        self.cursor.execute("""
            SELECT video_id FROM videos WHERE video_name = ?
        """, (video_name,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def insert_frame(self, video_name: str, frame_number: int, timestamp: float) -> int:
        """
        Insert a frame for a specific video.
        If the frame already exists for this video, it will be replaced.
        """
        # Ensure the video exists
        video_id = self.add_video(video_name)
        
        # Check if frame already exists
        frame_id = self.get_frame_id(video_id, frame_number)
        
        if frame_id:
            # Remove the frame if it already exists
            self.cursor.execute("DELETE FROM frames WHERE frame_id = ?", (frame_id,))
            self.connection.commit()

        # Insert the new frame
        self.cursor.execute("""
            INSERT INTO frames (video_id, frame_number, timestamp)
            VALUES (?, ?, ?)
        """, (video_id, frame_number, timestamp))
        self.connection.commit()
        
        return self.cursor.lastrowid

    def get_frame_id(self, video_id: int, frame_number: int) -> Optional[int]:
        """
        Get the frame_id for a specific video_id and frame_number.
        Returns frame_id if found, None otherwise.
        """
        self.cursor.execute("""
            SELECT frame_id FROM frames 
            WHERE video_id = ? AND frame_number = ?
        """, (video_id, frame_number))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_frame_by_number(self, video_name: str, frame_number: int):
        """
        Check if a frame number exists for a specific video.
        Returns the frame_id if it exists, or None otherwise.
        """
        video_id = self.get_video_id(video_name)
        if not video_id:
            return None
            
        self.cursor.execute("""
            SELECT frame_id FROM frames 
            WHERE video_id = ? AND frame_number = ?
        """, (video_id, frame_number))
        return self.cursor.fetchone()

    def insert_detection(self, video_name: str, frame_number: int, detection: Detection):
        """Insert detection for a specific video frame."""
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            raise ValueError(f"Video '{video_name}' not found in database")
            
        # Get frame_id
        frame_id = self.get_frame_id(video_id, frame_number)
        if not frame_id:
            raise ValueError(f"Frame {frame_number} not found for video '{video_name}'")
            
        self.cursor.execute("""
            INSERT INTO detections (frame_id, class_name, confidence, xmin, ymin, xmax, ymax, dominant_color, track_id, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            frame_id,
            detection.class_name,
            detection.confidence,
            detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3],
            detection.dominant_color,
            None,
            None
        ))
        self.connection.commit()

    def bulk_insert_detections(self, video_name: str, detections: List[dict]):
        """Bulk insert detections for a specific video."""
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            raise ValueError(f"Video '{video_name}' not found in database")
            
        # Prepare data with frame_id instead of frame_number
        processed_data = []
        for det in detections:
            frame_id = self.get_frame_id(video_id, det['frame_number'])
            if not frame_id:
                raise ValueError(f"Frame {det['frame_number']} not found for video '{video_name}'")
                
            processed_data.append(
                (frame_id, det['class_name'], det['confidence'], det['bbox'][0], 
                det['bbox'][1], det['bbox'][2], det['bbox'][3], det['dominant_color'], 
                det['track_id'], det['direction'])
            )
            
        self.cursor.executemany("""
            INSERT INTO detections (frame_id, class_name, confidence, xmin, ymin, xmax, ymax, dominant_color, track_id, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, processed_data)

        self.connection.commit()

    def bulk_insert_refined_detections(self, video_name: str, detections: List[dict]):
        """
        Bulk insert refined detections from tracker for a specific video.
        Args:
            video_name: Name of the video
            detections: List of dictionaries with detection data.
        """
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            raise ValueError(f"Video '{video_name}' not found in database")
            
        # Prepare data with frame_id instead of frame_number
        processed_data = []
        for det in detections:
            frame_id = self.get_frame_id(video_id, det['frame_number'])
            if not frame_id:
                raise ValueError(f"Frame {det['frame_number']} not found for video '{video_name}'")
                
            processed_data.append(
                (frame_id, det['track_id'], det['class_name'], det['confidence'],
                det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3],
                det['dominant_color'], det['direction'])
            )
            
        self.cursor.executemany("""
            INSERT INTO refined_detections (frame_id, track_id, class_name, confidence, xmin, ymin, xmax, ymax, dominant_color, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, processed_data)
        self.connection.commit()

    def get_bbox(self, detection_id: int) -> Tuple[int]:
        """Retrieve the bounding box for a specific detection."""
        self.cursor.execute("""
            SELECT xmin, ymin, xmax, ymax FROM detections WHERE detection_id = ?
        """, (detection_id,))
        return self.cursor.fetchone()

    def get_detections_by_frame(self, video_name: str, frame_number: int) -> List[Tuple[Any]]:
        """Retrieve detections for a specific frame of a video."""
        # Get video_id and frame_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            return []
            
        self.cursor.execute("""
            SELECT d.class_name, d.confidence, d.xmin, d.ymin, d.xmax, d.ymax, d.dominant_color
            FROM detections d
            JOIN frames f ON d.frame_id = f.frame_id
            WHERE f.video_id = ? AND f.frame_number = ?
        """, (video_id, frame_number))
        return self.cursor.fetchall()

    def get_refined_detections_by_frame(self, video_name: str, frame_number: int) -> List[Tuple[Any]]:
        """Retrieve refined (tracked) detections for a specific frame of a video."""
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            return []
            
        self.cursor.execute("""
            SELECT r.class_name, r.confidence, r.xmin, r.ymin, r.xmax, r.ymax, r.dominant_color, r.track_id, r.direction
            FROM refined_detections r
            JOIN frames f ON r.frame_id = f.frame_id
            WHERE f.video_id = ? AND f.frame_number = ?
        """, (video_id, frame_number))
        return self.cursor.fetchall()

    def reset_video(self, video_name: str):
        """Function to clear detections for a specific video."""
        video_id = self.get_video_id(video_name)
        if not video_id:
            print(f"Video '{video_name}' not found in database")
            return
            
        # Get all frame_ids for this video
        self.cursor.execute("""
            SELECT frame_id FROM frames WHERE video_id = ?
        """, (video_id,))
        frame_ids = [row[0] for row in self.cursor.fetchall()]
        
        if not frame_ids:
            print(f"No frames found for video '{video_name}'")
            return
            
        # Delete detections and refined_detections for these frames
        placeholders = ', '.join(['?'] * len(frame_ids))
        self.cursor.execute(f"DELETE FROM detections WHERE frame_id IN ({placeholders})", frame_ids)
        self.cursor.execute(f"DELETE FROM refined_detections WHERE frame_id IN ({placeholders})", frame_ids)
        
        # Delete the frames
        self.cursor.execute("DELETE FROM frames WHERE video_id = ?", (video_id,))
        
        # Keep the video record for future use
        self.connection.commit()
        print(f"Detections for video '{video_name}' cleared successfully.")

    def get_frames_with_detections(self, video_name: str, table_name: str, filter_objects=None, filter_colors=None,
                                  filter_directions=None, area_filter=None):
        """
        Fetch unique frames with matching detections from the database for a specific video.
        Allows conditional filtering based on object type, color, direction, logic and area.
        
        Args:
            video_name (str): Name of the video to query.
            table_name (str): Name of the table to query (detections or refined_detections).
            filter_objects (list, optional): List of object classes to filter by.
            filter_colors (list, optional): List of colors to filter by. Defaults to None.
            filter_directions (list, optional): List of directions to filter by. Defaults to None.
            area_filter (tuple, optional): (x1, y1, x2, y2, margin) defining area of interest. Defaults to None.
            
        Returns:
            frames: Dictionary containing frame number and detections.
        """
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            return {}
            
        where_clauses = [f"f.video_id = ?"]
        params = [video_id]

        # Add conditions based on the filters
        if filter_objects:
            object_placeholders = ', '.join(['?'] * len(filter_objects))
            where_clauses.append(f"d.class_name IN ({object_placeholders})")
            params.extend(filter_objects)

        if filter_colors:
            color_placeholders = ', '.join(['?'] * len(filter_colors))
            where_clauses.append(f"d.dominant_color IN ({color_placeholders})")
            params.extend(filter_colors)

        if filter_directions:
            direction_placeholders = ', '.join(['?'] * len(filter_directions))
            where_clauses.append(f"d.direction IN ({direction_placeholders})")
            params.extend(filter_directions)

        if area_filter:
            print(f'Area Filter: {area_filter}')
            area_x1, area_y1, area_x2, area_y2, margin = area_filter
            # Expand area by margin
            area_x1 -= margin
            area_y1 -= margin
            area_x2 += margin
            area_y2 += margin
            # Filter based on centroid being within the area
            where_clauses.append(f"((d.xmin + d.xmax)/2 BETWEEN ? AND ?) AND ((d.ymin + d.ymax)/2 BETWEEN ? AND ?)")
            params.extend([area_x1, area_x2, area_y1, area_y2])

        # Join the clauses
        where_clause = " AND ".join(where_clauses)
        print(f"Where Clause: {where_clause}")

        query = f"""
        SELECT DISTINCT f.frame_number, d.class_name, d.dominant_color, d.xmin, d.xmax, d.ymin, d.ymax, d.confidence, d.track_id
        FROM {table_name} d
        JOIN frames f ON d.frame_id = f.frame_id
        WHERE {where_clause}
        """

        print(f"Constructed Query: {query}")
        print(f"Query Parameters: {params}")

        # Execute the query
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()

        # Group detections by frame_number
        frames = {}
        for row in rows:
            frame_number, class_name, dominant_color, xmin, xmax, ymin, ymax, confidence, track_id = row
            detection = {
                'class_name': class_name,
                'dominant_color': dominant_color,
                'bbox': (xmin, ymin, xmax, ymax),
                'confidence': confidence,
                'track_id': track_id
            }
            if frame_number not in frames:
                frames[frame_number] = {'frame_number': frame_number, 'detections': []}
            frames[frame_number]['detections'].append(detection)

        return frames

    def get_videos(self) -> List[Tuple[int, str]]:
        """Get a list of all videos in the database."""
        self.cursor.execute("""
            SELECT video_id, video_name FROM videos ORDER BY created_at DESC
        """)
        return self.cursor.fetchall()

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()

    def reset_database(self):
        """Function to clear all data from the database."""
        self.cursor.execute("DELETE FROM detections")
        self.cursor.execute("DELETE FROM refined_detections")
        self.cursor.execute("DELETE FROM frames")
        self.cursor.execute("DELETE FROM videos")
        self.connection.commit()
        print("Database reset complete.")

    def drop_tables(self):
        """Drop the tables in the database."""
        self.cursor.execute("DROP TABLE IF EXISTS detections")
        self.cursor.execute("DROP TABLE IF EXISTS refined_detections")
        self.cursor.execute("DROP TABLE IF EXISTS frames")
        self.cursor.execute("DROP TABLE IF EXISTS videos")
        self.connection.commit()
        print("Tables dropped successfully.")