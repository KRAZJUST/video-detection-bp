import psutil
import os
import time
import functools

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
