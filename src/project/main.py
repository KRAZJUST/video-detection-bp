# File: main.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This is the main entry point for the application. It initializes the
# application, sets up the database, and starts the main event loop.
# It also provides an option to run the application with profiling enabled
# for performance analysis.
#
# =============================================================================

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
    #db.drop_tables()
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
    if len(sys.argv) > 1 and sys.argv[1] == '-profile':
        # Run with profiler
        cProfile.run('main()', 'app_profile.prof')
        
        # Print summary to console
        p = pstats.Stats('app_profile.prof')
        print("\n=== TOP FUNCTIONS BY TIME ===")
        p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
        
        print("\n=== TOP FUNCTIONS BY CUMULATIVE TIME ===")
        p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
        
        print("\n=== TOP CALLERS ===")
        p.sort_stats(SortKey.TIME).print_callers(10)
        
        print("\n=== CALL HIERARCHY ===")
        p.print_callees(10)
        
        # Save detailed stats for visualization tools
        p.dump_stats('app_profile.prof')
        print("\nProfile data saved to 'app_profile.prof'")
    else:
        main()
