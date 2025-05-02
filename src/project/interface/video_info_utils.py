# =============================================================================
# File: video_info_utils.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# IMPORTANT: This module uses open-source FFprobe program that is a part of
#   FFmpeg project. FFmpeg is a complete, cross-platform solution to record,
#   convert and stream audio and video. FFprobe is a tool to analyze multimedia
#   files and streams.
#   Available at: https://ffmpeg.org/
#   FFmpeg documentation: https://ffmpeg.org/documentation.html
#
# Description:
# This module provides utility functions for extracting video metadata using
# FFmpeg and FFprobe. It includes functions to get video duration, frame rate,
# frame count, resolution, codec, bitrate, and file size. The metadata is
# then returned as a structured object to be used in the application.
#
# =============================================================================

import subprocess
import json
from dataclasses import dataclass
from typing import Optional, Dict
import re

@dataclass
class VideoMetadata:
    # Video duration in seconds
    duration: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str
    bitrate: str
    size: str

class VideoInfoUtils:
    @staticmethod
    def _run_ffprobe(video_path: str) -> Optional[Dict]:
        """
        Run ffprobe command to get video information in JSON format.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Optional[Dict]: FFprobe output as dictionary if successful, None if failed
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFprobe error: {result.stderr}")
                return None
                
            return json.loads(result.stdout)
            
        except Exception as e:
            print(f"Error running ffprobe: {e}")
            return None

    @staticmethod
    def _get_frame_count(video_path: str) -> Optional[int]:
        """
        Get exact frame count using ffprobe frame counting.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Optional[int]: Frame count if successful, None if failed
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())
            
            return None
            
        except Exception as e:
            print(f"Error counting frames: {e}")
            return None

    @staticmethod
    def get_video_info(video_path: str) -> Optional[VideoMetadata]:
        """
        Extract comprehensive information from a video file using FFmpeg.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Optional[VideoMetadata]: Video metadata if successful, None if failed
        """
        try:
            probe_data = VideoInfoUtils._run_ffprobe(video_path)
            
            if not probe_data:
                return None
                
            # Find the video stream
            video_stream = next(
                (stream for stream in probe_data['streams'] 
                 if stream['codec_type'] == 'video'),
                None
            )
            print(video_stream)
            
            if not video_stream:
                return None
                
            # Parse frame rate from the ratio format (xxxxx/yyyyy)
            fps_str = video_stream.get('avg_frame_rate', '0/1')
            print(fps_str)
            fps = 'unknown'
            if fps_str and '/' in fps_str:
                try:
                    num, den = map(int, fps_str.split('/'))
                    fps = round(float(num / den), 2) if den != 0 else 'unknown'
                except (ValueError, ZeroDivisionError):
                    fps = 'unknown'
            elif fps_str:
                try:
                    fps = round(float(fps_str), 2)
                except ValueError:
                    fps = 'unknown'
                
            # Get duration from format section if available, otherwise from stream
            duration = float(probe_data['format'].get('duration', 
                           video_stream.get('duration', 0)))
                
            # Get frame count or calculate it
            frame_count = VideoInfoUtils._get_frame_count(video_path)
            if frame_count is None and fps > 0:
                frame_count = int(duration * fps)
                
            # Get file size in human-readable format
            size_bytes = int(probe_data['format'].get('size', 0))
            size = VideoInfoUtils._format_size(size_bytes)
                
            return VideoMetadata(
                duration=round(float(duration), 2),
                fps= 'unknown' if fps == 'unknown' else round(float(fps), 2),
                frame_count=frame_count,
                width=int(video_stream.get('width', 0)),
                height=int(video_stream.get('height', 0)),
                codec=video_stream.get('codec_name', 'unknown'),
                bitrate=probe_data['format'].get('bit_rate', 'unknown'),
                size=size
            )
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"