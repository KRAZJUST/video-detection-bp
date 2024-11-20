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
        self.connection.commit()

    def insert_frame(self, frame_number: int, timestamp: float) -> int:
        """Insert frame metadata and return the frame ID."""
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

    def get_frame_number(self, frame_number: int) -> int:
        """Retrieve the frame ID for a specific frame number."""
        self.cursor.execute("""
            SELECT frame_number FROM frames WHERE frame_number = ?
        """, (frame_number,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_detections_by_frame(self, frame_number: int) -> List[Tuple[Any]]:
        """Retrieve detections for a specific frame."""
        self.cursor.execute("""
            SELECT d.class_name, d.confidence, d.xmin, d.ymin, d.xmax, d.ymax, d.dominant_color
            FROM detections d
            JOIN frames f ON d.frame_number = f.frame_number
            WHERE f.frame_number = ?
        """, (frame_number,))
        return self.cursor.fetchall()

    def reset_database(self):
        """ Function to clear previous detections from the database. """
        self.cursor.execute("DELETE FROM detections")
        self.connection.commit()
        print("Database reset complete.")

    def get_frames_with_detections(self, filter_objects, filter_colors):
        """Fetch unique frames with matching detections from the database."""
        query = """
        SELECT DISTINCT frame_number, class_name, dominant_color, xmin, xmax, ymin, ymax, confidence
        FROM detections
        WHERE class_name IN (?) AND dominant_color IN (?)
        """
        object_filter_str = ', '.join([f"'{obj}'" for obj in filter_objects])
        color_filter_str = ', '.join([f"'{color}'" for color in filter_colors])

        self.cursor.execute(query, (object_filter_str, color_filter_str))
        rows = self.cursor.fetchall()

        # Group detections by frame_number
        frames = {}
        for row in rows:
            frame_number, class_name, dominant_color, xmin, xmax, ymin, ymax, confidence = row
            if frame_number not in frames:
                frames[frame_number] = {'frame_number': frame_number, 'detections': []}
            frames[frame_number]['detections'].append({
                'class_name': class_name,
                'dominant_color': dominant_color,
                'bbox': (xmin, ymin, xmax, ymax),
                'confidence': confidence
            })

        return frames


    def close(self):
        """Close the database connection."""
        self.connection.close()
