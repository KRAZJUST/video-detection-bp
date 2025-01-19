import sqlite3
from typing import List, Tuple, Any
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

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_number INTEGER PRIMARY KEY NOT NULL,
                timestamp REAL
            );
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_number INTEGER,
                class_name TEXT,
                confidence REAL,
                xmin INTEGER,
                ymin INTEGER,
                xmax INTEGER,
                ymax INTEGER,
                dominant_color TEXT,
                track_id INTEGER DEFAULT NULL,
                direction TEXT DEFAULT NULL,
                FOREIGN KEY (frame_number) REFERENCES frames (frame_number)
            );
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS refined_detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_number INTEGER,
                track_id INTEGER,
                class_name TEXT,
                confidence REAL,
                xmin INTEGER,
                ymin INTEGER,
                xmax INTEGER,
                ymax INTEGER,
                dominant_color TEXT,
                direction TEXT,
                FOREIGN KEY (frame_number) REFERENCES frames (frame_number)
            );
        """)

        self.connection.commit()

    def insert_frame(self, frame_number: int, timestamp: float) -> int:
        """
        If the frame does not exist, insert it into the database.
        If it already exists, remove the frame and re-insert it.
        """
        existing_frame = self.get_frame_number(frame_number)
        if existing_frame:
            # Remove the frame if it already exists and re-insert it
            self.cursor.execute("DELETE FROM frames WHERE frame_number = ?", (frame_number,))
            self.connection.commit()

        self.cursor.execute("""
            INSERT INTO frames (frame_number, timestamp)
            VALUES (?, ?)
        """, (frame_number, timestamp))
        self.connection.commit()
        return self.cursor.lastrowid

    def insert_detection(self, frame_number: int, detection: Detection):
        """Insert detection for a specific frame."""
        self.cursor.execute("""
            INSERT INTO detections (frame_number, class_name, confidence, xmin, ymin, xmax, ymax, dominant_color, track_id, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            frame_number,
            detection.class_name,
            detection.confidence,
            detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3],
            detection.dominant_color,
            None,
            None
        ))
        self.connection.commit()

    def bulk_insert_detections(self, detections: List[dict]):
        """Bulk insert detections for a specific frame."""
        data = [
            (det['frame_number'], det['class_name'], det['confidence'], det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3], det['dominant_color'], det['track_id'], 
             det['direction'])
            for det in detections
        ]
        self.cursor.executemany("""
            INSERT INTO detections (frame_number, class_name, confidence, xmin, ymin, xmax, ymax, dominant_color, track_id, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

        self.connection.commit()

    def bulk_insert_refined_detections(self, detections: List[dict]):
        """
        Bulk insert refined detections from tracker.
        Args:
            detections: List of dictionaries with detection data.
        """
        data = [
            (det['frame_number'], det['track_id'], det['class_name'], det['confidence'],
            det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3],
            det['dominant_color'], det['direction'])
            for det in detections
        ]
        self.cursor.executemany("""
            INSERT INTO refined_detections (frame_number, track_id, class_name, confidence, xmin, ymin, xmax, ymax, dominant_color, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        self.connection.commit()


    def update_detection(self, detection_id: int, track_id: int, direction: str):
        """Update the track ID and direction for a specific detection."""
        self.cursor.execute("""
            UPDATE detections SET track_id = ?, direction = ? WHERE detection_id = ?
        """, (track_id, direction, detection_id))
        self.connection.commit()

    def get_bbox(self, detection_id: int) -> Tuple[int]:
        """Retrieve the bounding box for a specific detection."""
        self.cursor.execute("""
            SELECT xmin, ymin, xmax, ymax FROM detections WHERE detection_id = ?
        """, (detection_id,))
        return self.cursor.fetchone()

    def get_frame_number(self, frame_number: int):
        """
        Check if a frame number exists in the database.
        Returns the frame ID and number if it exists, or None otherwise.
        """
        self.cursor.execute("""
            SELECT frame_number FROM frames WHERE frame_number = ?
        """, (frame_number,))
        return self.cursor.fetchone()


    def get_detections_by_frame(self, frame_number: int) -> List[Tuple[Any]]:
        """Retrieve detections for a specific frame."""
        self.cursor.execute("""
            SELECT d.class_name, d.confidence, d.xmin, d.ymin, d.xmax, d.ymax, d.dominant_color
            FROM detections d
            JOIN frames f ON d.frame_number = f.frame_number
            WHERE f.frame_number = ?
        """, (frame_number,))
        return self.cursor.fetchall()

    def get_refined_detections_by_frame(self, frame_number: int) -> List[Tuple[Any]]:
        """Retrieve refined (tracked) detections for a specific frame."""
        self.cursor.execute("""
            SELECT class_name, confidence, xmin, ymin, xmax, ymax, dominant_color, track_id, direction
            FROM refined_detections
            WHERE frame_number = ?
        """, (frame_number,))
        return self.cursor.fetchall()

    def reset_database(self):
        """ Function to clear previous detections from the database. """
        self.cursor.execute("DELETE FROM detections")
        self.cursor.execute("DELETE FROM frames")
        self.connection.commit()
        print("Database reset complete.")

    def get_frames_with_detections(self, table_name: str, filter_objects, filter_colors=None, filter_directions=None):
        """
        Fetch unique frames with matching detections from the database.
        Allows conditional filtering based on object type, color, direction, and logic.
        """
        where_clauses = []
        params = []

        # Add conditions based on the filters
        if filter_objects:
            object_placeholders = ', '.join(['?'] * len(filter_objects))
            where_clauses.append(f"class_name IN ({object_placeholders})")
            params.extend(filter_objects)

        if filter_colors:
            color_placeholders = ', '.join(['?'] * len(filter_colors))
            where_clauses.append(f"dominant_color IN ({color_placeholders})")
            params.extend(filter_colors)

        if filter_directions:
            direction_placeholders = ', '.join(['?'] * len(filter_directions))
            where_clauses.append(f"direction IN ({direction_placeholders})")
            params.extend(filter_directions)

        # Join the clauses
        where_clause = f" AND ".join(where_clauses) if where_clauses else "1=1"

        query = f"""
        SELECT DISTINCT frame_number, class_name, dominant_color, xmin, xmax, ymin, ymax, confidence
        FROM {table_name}
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
            frame_number, class_name, dominant_color, xmin, xmax, ymin, ymax, confidence = row
            detection = {
                'class_name': class_name,
                'dominant_color': dominant_color,
                'bbox': (xmin, ymin, xmax, ymax),
                'confidence': confidence
            }
            if frame_number not in frames:
                frames[frame_number] = {'frame_number': frame_number, 'detections': []}
            frames[frame_number]['detections'].append(detection)

        return frames

    def close(self):
        """Close the database connection."""
        self.connection.close()

    def drop_tables(self):
        """Drop the tables in the database."""
        self.cursor.execute("DROP TABLE IF EXISTS frames")
        self.cursor.execute("DROP TABLE IF EXISTS detections")
        self.cursor.execute("DROP TABLE IF EXISTS refined_detections")
        self.connection.commit()
        print("Tables dropped successfully.")