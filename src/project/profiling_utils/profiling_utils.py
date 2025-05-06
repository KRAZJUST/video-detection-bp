# =============================================================================
# File: profiling_utils.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module provides helper functions for profiling memory and time usage.
#
# =============================================================================

import psutil
import os
import time
import functools

# Dictionary to hold timing data
timing_data = {}

def profile_memory_usage():
    """
    Log the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 ** 2
    print(f"Memory usage: {memory_usage:.2f} MB")

def profile_time_usage(func):
    """
    Decorator to profile the execution time of a function.
    It prints the time taken to execute the function.
    This is a simple profiling decorator that can be used
    to measure the performance of specific functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function to measure execution time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} - execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Add periodic memory logging to the application with interval in milliseconds
def setup_memory_logging(app, interval=5000):
    """
    Set up periodic memory logging in the application.
    This function creates a timer that logs the memory usage
    of the application at regular intervals.

    Args:
        app: Reference to the main application
        interval: Interval in milliseconds for memory logging
    """
    timer = app.startTimer(interval)
    
    def timerEvent(event):
        if event.timerId() == timer:
            profile_memory_usage()
    
    # Monkey patch QApplication to add timer event
    original_timer_event = app.timerEvent
    app.timerEvent = lambda event: timerEvent(event) or original_timer_event(event)

def detailed_profile(func):
    """
    Decorator to profile the execution time of a function
    and store detailed timing data.
    It prints the time taken to execute the function and
    stores the timing data in a dictionary.
    This is a more detailed profiling decorator that can be used
    to measure the performance of specific functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        if name not in timing_data:
            timing_data[name] = {
                'calls': 0,
                'total_time': 0,
                'times': []
            }
        
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        # Save timing data
        timing_data[name]['calls'] += 1
        timing_data[name]['total_time'] += elapsed
        timing_data[name]['times'].append(elapsed)
        
        print(f"{name} took {elapsed:.4f} seconds")
        return result
    return wrapper