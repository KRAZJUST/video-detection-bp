from processors.video_processor import VideoProcessor
import argparse
import time


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='YOLO Object Detection with Color Filtering')
    parser.add_argument('--input', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Directory to save output frames and logs')
    parser.add_argument('--query', type=str, help='Search query in the format "color object", e.g., "red car"')
    parser.add_argument('--interval', type=int, default=30, help='Time interval in which to extract frames (default: every 30th frame)')
    parser.add_argument('--tracker', type=str, default='bytetrack', help='Which tracker to use - deepsort or bytetrack')

    args = parser.parse_args()
    argument_parsing_time = time.time()

    # Run detection on the video
    processor = VideoProcessor(
        video_path=args.input,
        output_dir=args.output,
        interval=args.interval,
        tracker_arg=args.tracker
    )
    processor.process_video()

    print(f"Total time taken: {time.time() - start_time:.2f} seconds (Argument Parsing: {argument_parsing_time - start_time:.2f} seconds)")

if __name__ == '__main__':
    main()