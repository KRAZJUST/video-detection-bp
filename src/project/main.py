# This is the main file to run the application
import sys
from database.sqlite_database import Database
from interface.app import VideoProcessingApp
from PyQt6.QtWidgets import QApplication
from profiling_utils.profiling_utils import setup_memory_logging, profile_memory_usage
import cProfile
import pstats
from pstats import SortKey

def main():
    # Set up database
    db_path = "detections.db"
    db = Database(db_path)
    #db.reset_database()
    db.close()

    # Create QApplication instance
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Setup memory logging - will print memory usage every 20 seconds
    setup_memory_logging(app, interval=20000)
    print("Initial memory usage:")
    profile_memory_usage()

    # Create and show the main window
    main_window = VideoProcessingApp(db_path)
    main_window.show()

    # Start the event loop
    sys.exit(app.exec())

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'profile':
        cProfile.run('main()', 'app_profile')
        p = pstats.Stats('app_profile')
        p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
    else:
        main() 
