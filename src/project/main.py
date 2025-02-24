# This is the main file to run the application
import argparse
import time
import sys
from database.sqlite_database import Database
from interface.app import VideoProcessingApp
from PyQt6.QtWidgets import QApplication

if __name__ == '__main__':
    # Set up database
    db_path = "detections.db"
    db = Database(db_path)
    db.reset_database()
    db.close()

    # Create QApplication instance
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Create and show the main window
    main_window = VideoProcessingApp(db_path)
    main_window.show()

    # Start the event loop
    sys.exit(app.exec())