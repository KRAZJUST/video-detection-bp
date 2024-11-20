from interface.app import VideoProcessingApp
import argparse
import time
from database.database import Database

if __name__ == '__main__':
    db_path = "detections.db"
    
    # Create an instance to reset the database
    db = Database(db_path)
    # Reset the database
    db.reset_database()
    # Close the database
    db.close()

    # Create an instance of the VideoProcessingApp and pass the database path
    app = VideoProcessingApp(db_path)
    app.run()
