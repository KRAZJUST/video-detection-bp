from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
import os
import asyncio
import json
import uuid
import sys
import time
import threading
import os


# Ensure the project root is in the Python path so we can import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.sqlite_database import Database
from processors.video_processor import VideoProcessor
from interface.video_info_utils import VideoInfoUtils
from parsers.detection_parser import DetectionParser
from xclip.xclip_parser import XClipParser
from siglip.siglip_parser import SigLIPParser

app = FastAPI(title="NeuroVision API", description="API for video detection backend")

# Allow CORS for the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to track progress of tasks
task_progress = {}

# Ensure required directories exist
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DB Init
db_path = "detections.db"
# Initialize DB just to make sure tables exist
Database(db_path).close()

# Mount static files for output images
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def root():
    return {"status": "ok", "message": "NeuroVision API is running"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video and returns the path to it. Cleans up old uploads."""
    # Clean up files older than 1 hour in the uploads directory
    now = time.time()
    for f in os.listdir(UPLOAD_DIR):
        f_path = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(f_path):
            if os.stat(f_path).st_mtime < now - 3600:
                try:
                    os.remove(f_path)
                except Exception:
                    pass

    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1]
    safe_filename = f"video_{file_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    return {"filename": safe_filename, "path": file_path}

@app.delete("/api/video")
async def delete_video(path: str):
    """Deletes a specific video file."""
    # Ensure it's inside UPLOAD_DIR for safety
    abs_upload_dir = os.path.abspath(UPLOAD_DIR)
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path) and abs_path.startswith(abs_upload_dir):
        try:
            os.remove(abs_path)
            return {"message": "Video deleted"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=404, content={"error": "File not found or invalid path"})

@app.get("/api/video-info")
def get_video_info(path: str):
    """Get metadata about a video."""
    try:
        if not os.path.exists(path):
            return JSONResponse(status_code=404, content={"error": "File not found"})
        info = VideoInfoUtils.get_video_info(path)
        return vars(info)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def process_video_task(task_id: str, video_path: str, output_dir: str, interval: int, tracker: str, segmentation: bool):
    """Background task to process the video."""
    task_progress[task_id] = {"status": "running", "progress": 0, "message": "Starting processing...", "should_cancel": False}
    
    def progress_callback(message, percentage):
        if task_progress[task_id].get("should_cancel", False):
            return False  # Terminate processing
            
        task_progress[task_id]["progress"] = percentage
        task_progress[task_id]["message"] = message
        # Return True to continue processing
        return True

    try:
        processor = VideoProcessor(
            video_path=video_path,
            database_path=db_path,
            output_dir=output_dir,
            interval=interval,
            model_name=tracker,
            use_segmentation=segmentation,
            skip_siglip_with_yolo=False,
            progress_callback=progress_callback,
        )
        processor.process_video()
        task_progress[task_id]["status"] = "completed"
        task_progress[task_id]["progress"] = 100
        task_progress[task_id]["message"] = "Processing complete!"
        
        # Schedule the video for deletion after 1 hour (3600 seconds) to save space
        def delayed_delete():
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass
        
        threading.Timer(3600.0, delayed_delete).start()
            
    except Exception as e:
        task_progress[task_id]["status"] = "error"
        task_progress[task_id]["message"] = str(e)


@app.post("/api/process")
async def start_processing(
    background_tasks: BackgroundTasks,
    video_path: str = Form(...),
    output_dir: str = Form("./outputs/"),
    interval: int = Form(30),
    tracker: str = Form("yolo"),
    segmentation: bool = Form(False)
):
    """Starts video processing in the background."""
    if not os.path.exists(video_path):
        return JSONResponse(status_code=400, content={"error": "Video file not found"})
        
    task_id = str(uuid.uuid4())
    background_tasks.add_task(process_video_task, task_id, video_path, output_dir, interval, tracker, segmentation)
    return {"task_id": task_id, "message": "Processing started"}

@app.post("/api/process/cancel/{task_id}")
async def cancel_processing(task_id: str):
    """Cancels an ongoing processing task."""
    if task_id in task_progress:
        task_progress[task_id]["should_cancel"] = True
        return {"message": "Cancellation requested"}
    return JSONResponse(status_code=404, content={"error": "Task not found"})


@app.get("/api/process/status/{task_id}")
async def stream_progress(task_id: str, request: Request):
    """SSE endpoint for streaming progress."""
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
                
            task_info = task_progress.get(task_id, {"status": "unknown"})
            yield f"data: {json.dumps(task_info)}\n\n"
            
            if task_info.get("status") in ["completed", "error", "unknown"]:
                break
                
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/query")
async def run_query(
    query: str = Form(...),
    video_path: str = Form(...),
    tracker: str = Form("yolo"),
    segmentation: bool = Form(False),
    deduplicate: bool = Form(True)
):
    """Run a query on processed video frames."""
    if not os.path.exists(video_path):
        return JSONResponse(status_code=400, content={"error": "Video file not found"})
        
    try:
        results = {}
        metadata = {}
        
        # Dummy feedback callback
        def feedback_callback(msg): pass
            
        if tracker in ["yolo", "bytetrack"]:
            log_parser = DetectionParser(
                query=query,
                input_video=video_path,
                output_dir=OUTPUT_DIR,
                database_path=db_path,
                tracker=tracker,
                use_segmentation=segmentation,
                area_of_interest=None,
                feedback_callback=feedback_callback,
            )
            log_parser.parse_detections()
            
            # Simple conversion to generic dict since deduplication functions 
            # are tied to the UI class, we'll return raw for now or we could copy them over.
            results = log_parser.found_log_entries
            
        elif tracker == "xclip-32":
            xclip_parser = XClipParser(
                video_path=video_path,
                query=query,
                output_dir=OUTPUT_DIR,
                model_name=tracker
            )
            similarities, metadata = xclip_parser.search_embeddings(top_k=5)
            results = xclip_parser.top_frames
            
        elif tracker == "siglip":
            siglip_parser = SigLIPParser(
                video_path=video_path,
                query=query,
                output_dir=OUTPUT_DIR,
                model_name=tracker
            )
            similarities, metadata = siglip_parser.search_embeddings(n_results=40)
            results = siglip_parser.top_frames
            
        return {"results": results, "metadata": metadata}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/open-player")
async def open_in_system_player(video_path: str = Form(...), timestamp: float = Form(0.0)):
    """Opens the video in the system player (e.g., mpv or vlc) at the given timestamp."""
    if not os.path.exists(video_path):
        return JSONResponse(status_code=400, content={"error": "Video file not found"})
        
    try:
        import subprocess
        # Try to open with mpv first, which supports --start
        # If it fails, fallback to OS default open
        try:
            # We use Popen so it doesn't block the backend
            subprocess.Popen(['mpv', f'--start={timestamp}', video_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"message": "Opened in mpv"}
        except FileNotFoundError:
            # Fallback to system default if mpv is not installed
            if sys.platform == "win32":
                os.startfile(video_path)
            elif sys.platform == "darwin":
                subprocess.Popen(['open', video_path])
            else:
                subprocess.Popen(['xdg-open', video_path])
            return {"message": "Opened in system default player (timestamp not supported)"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
