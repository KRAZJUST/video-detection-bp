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
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 ** 2
    print(f"Memory usage: {memory_usage:.2f} MB")

def profile_time_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} - execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Add periodic memory logging to the application with interval in milliseconds
def setup_memory_logging(app, interval=5000):
    timer = app.startTimer(interval)
    
    def timerEvent(event):
        if event.timerId() == timer:
            profile_memory_usage()
    
    # Monkey patch QApplication to add timer event
    original_timer_event = app.timerEvent
    app.timerEvent = lambda event: timerEvent(event) or original_timer_event(event)

def detailed_profile(func):
    """More detailed profiling decorator that collects statistics"""
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