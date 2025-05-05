# =============================================================================
# File: sqlite_database.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module provides a SQLite database interface for storing and managing
# video detection data. It includes methods for creating tables, inserting
# data, and querying the database. The database is designed to handle
# video metadata, frames, YOLO detections, and ByteTrack refined detections.
#
# =============================================================================

import sqlite3
from typing import List, Tuple, Any, Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='database_operations.log',  # Log to file
    filemode='a'  # Append mode
)
logger = logging.getLogger("DatabaseOperations")

class Database:
    def __init__(self, db_path: str = "detections.db"):
        """
        Initialize the database connection and create tables if they don't exist.
        Args:
            db_path (str): Path to the SQLite database file.
        """

        self.db_path = db_path
        self.connection = None
        self.connect()
        self.create_tables()

    def __enter__(self):
        """Context manager for database connection."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the database connection when exiting the context."""
        self.close()

    def connect(self):
        """Connect to the database."""

        logger.info(f"Connecting to database at {self.db_path}")
        self.connection = sqlite3.connect(self.db_path)
        # add PRAGMA statements
        self.cursor = self.connection.cursor()
        # Configure SQLite for better concurrency
        # This is important for write-heavy applications
        # and is here to minimize the errors related to database being locked
        logger.debug("Setting PRAGMA statements for database optimization")
        self.cursor.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
        self.cursor.execute("PRAGMA synchronous = NORMAL")  # Less rigorous but faster durability
        self.cursor.execute("PRAGMA busy_timeout = 5000")  # Wait up to 5 seconds when database is locked

    def create_tables(self):
        """Create the necessary tables in the database."""

        # Videos table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create two tables for frames so it can store them
        # separately for YOLO and ByteTrack as they use different
        # interval and the frame numbers would be different
        # and needed to be overwritten
        # YOLO frames table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS yolo_frames (
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp REAL,
                FOREIGN KEY (video_id) REFERENCES videos (video_id),
                UNIQUE (video_id, frame_number)
            );
        """)

        # ByteTrack frames table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS bytetrack_frames (
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp REAL,
                FOREIGN KEY (video_id) REFERENCES videos (video_id),
                UNIQUE (video_id, frame_number)
            );
        """)

        # Detections table (linked to YOLO frames)
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
                FOREIGN KEY (frame_id) REFERENCES yolo_frames (frame_id)
            );
        """)

        # Refined detections table (linked to ByteTrack frames)
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
                FOREIGN KEY (frame_id) REFERENCES bytetrack_frames (frame_id)
            );
        """)

        self.connection.commit()

    def add_video(self, video_name: str) -> int:
        """
        Add a new video to the database or get the ID if it already exists.

        Args:
            video_name (str): The name of the video to add.
        Returns:
            int: The ID of the video in the database.
        """
        try:
            self.cursor.execute("""
                INSERT INTO videos (video_name) VALUES (?)
            """, (video_name,))
            logger.info(f"Video '{video_name}' added successfully.")
            self.connection.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            # Video already exists, get its ID
            self.cursor.execute("""
                SELECT video_id FROM videos WHERE video_name = ?
            """, (video_name,))
            return self.cursor.fetchone()[0]

    def get_video_id(self, video_name: str) -> Optional[int]:
        """
        Get video_id by video_name.
        
        Args:
            video_name (str): The name of the video to search for.
        Returns:
            Optional[int]: The ID of the video if found, otherwise None.
        """
        self.cursor.execute("""
            SELECT video_id FROM videos WHERE video_name = ?
        """, (video_name,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    # YOLO frame methods
    def insert_yolo_frame(self, video_name: str, frame_number: int, timestamp: float) -> int:
        """
        Insert a frame for YOLO detection processing.

        Args:
            video_name (str): The name of the video.
            frame_number (int): The frame number to insert.
            timestamp (float): The timestamp of the frame.
        Returns:
            int: The ID of the inserted frame.
        """
        # Ensure the video exists
        video_id = self.add_video(video_name)
        
        # Check if frame already exists
        frame_id = self.get_yolo_frame_id(video_id, frame_number)
        
        if frame_id:
            # Remove the frame before inserting it again
            self.cursor.execute("""
                DELETE FROM yolo_frames WHERE frame_id = ?
            """, (frame_id,))
            self.connection.commit()
        
        # Insert the new frame
        self.cursor.execute("""
            INSERT INTO yolo_frames (video_id, frame_number, timestamp)
            VALUES (?, ?, ?)
        """, (video_id, frame_number, timestamp))
        self.connection.commit()
        
        return self.cursor.lastrowid

    def get_yolo_frame_id(self, video_id: int, frame_number: int) -> Optional[int]:
        """
        Get the frame_id for a YOLO frame.

        Args:
            video_id (int): The ID of the video.
            frame_number (int): The frame number to search for.
        Returns:
            Optional[int]: The ID of the frame if found, otherwise None.
        """
        self.cursor.execute("""
            SELECT frame_id FROM yolo_frames 
            WHERE video_id = ? AND frame_number = ?
        """, (video_id, frame_number))
        result = self.cursor.fetchone()
        return result[0] if result else None

    # ByteTrack frame methods
    def insert_bytetrack_frame(self, video_name: str, frame_number: int, timestamp: float) -> int:
        """
        Insert a frame for ByteTrack processing.

        Args:
            video_name (str): The name of the video.
            frame_number (int): The frame number to insert.
            timestamp (float): The timestamp of the frame.
        Returns:
            int: The ID of the inserted frame.
        """
        # Ensure the video exists
        video_id = self.add_video(video_name)
        
        # Check if frame already exists
        frame_id = self.get_bytetrack_frame_id(video_id, frame_number)
        
        if frame_id:
            # Remove the frame before inserting it again
            self.cursor.execute("""
                DELETE FROM bytetrack_frames WHERE frame_id = ?
            """, (frame_id,))
            self.connection.commit()
        
        # Insert the new frame
        self.cursor.execute("""
            INSERT INTO bytetrack_frames (video_id, frame_number, timestamp)
            VALUES (?, ?, ?)
        """, (video_id, frame_number, timestamp))
        self.connection.commit()
        
        return self.cursor.lastrowid

    def get_bytetrack_frame_id(self, video_id: int, frame_number: int) -> Optional[int]:
        """
        Get the frame_id for a ByteTrack frame.

        Args:
            video_id (int): The ID of the video.
            frame_number (int): The frame number to search for.
        Returns:
            Optional[int]: The ID of the frame if found, otherwise None.
        """
        self.cursor.execute("""
            SELECT frame_id FROM bytetrack_frames 
            WHERE video_id = ? AND frame_number = ?
        """, (video_id, frame_number))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def bulk_insert_detections(self, video_name: str, detections: List[dict]):
        """
        Bulk insert detections for YOLO frames.
        
        Args:
            video_name (str): The name of the video.
            detections (List[dict]): List of detection dictionaries containing
                'frame_number', 'class_name', 'confidence', 'bbox', etc.
        
        Returns:
            None
        """
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            raise ValueError(f"Video '{video_name}' not found in database")
            
        # Prepare data with frame_id
        processed_data = []
        for det in detections:
            frame_id = self.get_yolo_frame_id(video_id, det['frame_number'])
            if not frame_id:
                # Automatically create the frame if it doesn't exist
                frame_id = self.insert_yolo_frame(video_name, det['frame_number'], 0)
                
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
        Bulk insert refined detections from ByteTrack.

        Args:
            video_name (str): The name of the video.
            detections (List[dict]): List of detection dictionaries containing
                'frame_number', 'track_id', 'class_name', 'confidence', 'bbox', etc.
        
        Returns:
            None
        """
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            raise ValueError(f"Video '{video_name}' not found in database")
            
        # Prepare data with frame_id
        processed_data = []
        for det in detections:
            frame_id = self.get_bytetrack_frame_id(video_id, det['frame_number'])
            if not frame_id:
                # Automatically create the frame if it doesn't exist
                frame_id = self.insert_bytetrack_frame(video_name, det['frame_number'], 0)
                
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

    def reset_video_yolo(self, video_name: str):
        """
        Function to clear YOLO detections for a specific video.
        
        Args:
            video_name (str): The name of the video to clear.
        Returns:
            None
        """
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            # if the video is not in the database, add it
            logger.debug(f"Video '{video_name}' not found in database, adding it.")
            self.add_video(video_name)
            return
        
        logger.debug(f"Video ID: {video_id}")

        # Get all frame_ids for this video
        self.cursor.execute("""
            SELECT frame_id FROM yolo_frames WHERE video_id = ?
        """, (video_id,))
        frame_ids = [row[0] for row in self.cursor.fetchall()]
        
        if not frame_ids:
            print(f"No YOLO frames found for video '{video_name}'")
            return
            
        # Delete detections for these frames
        logger.debug(f"Deleting {len(frame_ids)} detections from YOLO frames")
        placeholders = ', '.join(['?'] * len(frame_ids))
        self.cursor.execute(f"DELETE FROM detections WHERE frame_id IN ({placeholders})", frame_ids)
        
        # Delete the frames
        self.cursor.execute("DELETE FROM yolo_frames WHERE video_id = ?", (video_id,))
        
        self.connection.commit()
        print(f"YOLO detections for video '{video_name}' cleared successfully.")

    def reset_video_bytetrack(self, video_name: str):
        """
        Function to clear ByteTrack refined detections for a specific video.
        
        Args:
            video_name (str): The name of the video to clear.
        Returns:
            None
        """
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            # if the video is not in the database, add it
            logger.debug(f"Video '{video_name}' not found in database, adding it.")
            self.add_video(video_name)
            return
            
        # Get all frame_ids for this video
        self.cursor.execute("""
            SELECT frame_id FROM bytetrack_frames WHERE video_id = ?
        """, (video_id,))
        frame_ids = [row[0] for row in self.cursor.fetchall()]
        
        if not frame_ids:
            print(f"No ByteTrack frames found for video '{video_name}'")
            return
            
        # Delete refined detections for these frames
        placeholders = ', '.join(['?'] * len(frame_ids))
        self.cursor.execute(f"DELETE FROM refined_detections WHERE frame_id IN ({placeholders})", frame_ids)
        
        # Delete the frames
        self.cursor.execute("DELETE FROM bytetrack_frames WHERE video_id = ?", (video_id,))
        
        self.connection.commit()
        print(f"ByteTrack detections for video '{video_name}' cleared successfully.")

    def get_yolo_frames_with_detections(self, video_name: str, filter_objects=None, filter_colors=None,
                                      filter_directions=None, area_filter=None):
        """
        Fetch unique frames with matching YOLO detections from the database 
        for a specific video.

        Args:
            video_name (str): The name of the video to fetch frames for.
            filter_objects (List[str]): List of object classes to filter by.
            filter_colors (List[str]): List of colors to filter by.
            filter_directions (List[str]): List of directions to filter by.
            area_filter (Tuple[int, int, int, int, int]): Area filter defined by
                (x1, y1, x2, y2, margin).
        Returns:
            Dict[int, Dict[str, Any]]: A dictionary where keys are frame numbers
                and values are dictionaries containing detection data.
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
        FROM detections d
        JOIN yolo_frames f ON d.frame_id = f.frame_id
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
    
    def get_frames_with_yolo_detections(self, video_name: str):
        """
        Function to fetch the numbers of frames containing YOLO detections
        for a specific video.
        Used to filter out frames with no detections when processing with SigLIP.

        Args:
            video_name (str): The name of the video to fetch frames for.
        Returns:
            Set[int]: A set of frame numbers containing YOLO detections.
        """
        # Get video_id
        video_id = self.get_video_id(video_name)
        if not video_id:
            return set()
        
        # Query to get all frames with detections for the video
        query = """
        SELECT *
        FROM yolo_frames f
        JOIN detections d ON d.frame_id = f.frame_id
        WHERE f.video_id = ?
        """
        
        self.cursor.execute(query, [video_id])
        rows = self.cursor.fetchall()
        
        # Extract just the frame numbers and return as a set
        frame_numbers = set()
        for row in rows:
            frame_numbers.add(row[2])

        return frame_numbers

    def get_bytetrack_frames_with_detections(self, video_name: str, filter_objects=None, filter_colors=None,
                                           filter_directions=None, area_filter=None):
        """
        Fetch unique frames with matching ByteTrack refined detections from 
        the database for a specific video.

        Args:
            video_name (str): The name of the video to fetch frames for.
            filter_objects (List[str]): List of object classes to filter by.
            filter_colors (List[str]): List of colors to filter by.
            filter_directions (List[str]): List of directions to filter by.
            area_filter (Tuple[int, int, int, int, int]): Area filter defined by
                (x1, y1, x2, y2, margin).
        Returns:
            Dict[int, Dict[str, Any]]: A dictionary where keys are frame numbers
                and values are dictionaries containing detection data.
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
        SELECT DISTINCT f.frame_number, d.class_name, d.dominant_color, d.xmin, d.xmax, d.ymin, d.ymax, d.confidence, d.track_id, d.direction
        FROM refined_detections d
        JOIN bytetrack_frames f ON d.frame_id = f.frame_id
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
            frame_number, class_name, dominant_color, xmin, xmax, ymin, ymax, confidence, track_id, direction = row
            detection = {
                'class_name': class_name,
                'dominant_color': dominant_color,
                'bbox': (xmin, ymin, xmax, ymax),
                'confidence': confidence,
                'track_id': track_id,
                'direction': direction
            }
            if frame_number not in frames:
                frames[frame_number] = {'frame_number': frame_number, 'detections': []}
            frames[frame_number]['detections'].append(detection)

        return frames
    
    def get_number_of_frames_yolo(self, video_name: str) -> int:
        """
        Get the number of frames for a specific video in YOLO frames.
        
        Args:
            video_name (str): The name of the video to count frames for.
        Returns:
            int: The number of frames for the video.
        """
        video_id = self.get_video_id(video_name)
        if not video_id:
            return 0
            
        self.cursor.execute("""
            SELECT COUNT(*) FROM yolo_frames WHERE video_id = ?
        """, (video_id,))
        return self.cursor.fetchone()[0]
    
    def get_number_of_frames_bytetrack(self, video_name: str) -> int:
        """
        Get the number of frames for a specific video in ByteTrack frames.
        
        Args:
            video_name (str): The name of the video to count frames for.
        Returns:
            int: The number of frames for the video.
        """
        video_id = self.get_video_id(video_name)
        if not video_id:
            return 0
            
        self.cursor.execute("""
            SELECT COUNT(*) FROM bytetrack_frames WHERE video_id = ?
        """, (video_id,))
        return self.cursor.fetchone()[0]

    def get_videos(self) -> List[Tuple[int, str]]:
        """
        Get a list of all videos in the database.
        
        Returns:
            List[Tuple[int, str]]: A list of tuples containing video_id and video_name.
        """
        self.cursor.execute("""
            SELECT video_id, video_name FROM videos ORDER BY created_at DESC
        """)
        return self.cursor.fetchall()
    
    def get_frame_timestamp(self, video_name: str, frame_number: int, tracker: str) -> Optional[float]:
        """
        Get the timestamp of a specific frame in a video.

        Args:
            video_name (str): The name of the video.
            frame_number (int): The frame number to get the timestamp for.
            tracker (str): The tracker used ('yolo' or 'bytetrack').
        Returns:
            Optional[float]: The timestamp of the frame if found, otherwise None.
        """
        video_id = self.get_video_id(video_name)
        if not video_id:
            return None
        
        if tracker == 'yolo' or tracker == 'xclip-32' or tracker == 'xclip-16':
            self.cursor.execute("""
                SELECT timestamp FROM yolo_frames WHERE video_id = ? AND frame_number = ?
            """, (video_id, frame_number))
        elif tracker == 'bytetrack':
            self.cursor.execute("""
                SELECT timestamp FROM bytetrack_frames WHERE video_id = ? AND frame_number = ?
            """, (video_id, frame_number))
        else:
            raise ValueError("Invalid tracker specified. Use 'yolo' or 'bytetrack'.")
        result = self.cursor.fetchone()
        return result[0] if result else None
      
    def close(self):
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()

    def reset_database(self):
        """
        Clear all data from the database.
        """
        self.cursor.execute("DELETE FROM detections")
        self.cursor.execute("DELETE FROM refined_detections")
        self.cursor.execute("DELETE FROM yolo_frames")
        self.cursor.execute("DELETE FROM bytetrack_frames")
        self.cursor.execute("DELETE FROM videos")
        self.connection.commit()
        print("Database reset complete.")

    def drop_tables(self):
        """
        Drop the tables in the database.
        """
        self.cursor.execute("DROP TABLE IF EXISTS detections")
        self.cursor.execute("DROP TABLE IF EXISTS refined_detections")
        self.cursor.execute("DROP TABLE IF EXISTS yolo_frames")
        self.cursor.execute("DROP TABLE IF EXISTS bytetrack_frames")
        self.cursor.execute("DROP TABLE IF EXISTS videos")
        self.connection.commit()
        print("Tables dropped successfully.")