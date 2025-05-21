from pathlib import Path
from typing import Dict
import logging

from config.defaults import (
    DEFAULT_THREADS,
    DEFAULT_QUALITY,
    DEFAULT_FORMAT,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_FFMPEG_PATH,
    DEFAULT_FRAME_PATTERN,
    DEFAULT_OVERWRITE,
)
from config.config import get_config
from config.logger_config import logger

from ffmpeg_extr import (
    Config as FFmpegConfig,
    VideoProcessor as FFmpegProcessor,
)
from openFace_prs import process_videos as process_with_openface


# List of supported video formats
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def process_video_pipeline(
    video_path: Path,
    output_dir: Path,
    ffmpeg_config: FFmpegConfig,
    openface_config: Dict,
) -> Dict:
    """Process a single video through the complete pipeline.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to store the output
        ffmpeg_config: Configuration for FFmpeg extraction
        openface_config: Configuration for OpenFace processing

    Returns:
        Dict containing the processing results
    """
    # Step 1: Extract I-frames with FFmpeg
    ffmpeg_processor = FFmpegProcessor(ffmpeg_config)
    ffmpeg_result = ffmpeg_processor.process_video(video_path)

    if not ffmpeg_result.get("status") == "success":
        logger.error(f"Failed to extract frames from {video_path}")
        return {
            "success": False,
            "video": str(video_path),
            "error": "FFmpeg extraction failed",
        }

    # Step 2: Process frames with OpenFace
    frames_dir = Path(ffmpeg_result["output_dir"])
    openface_result = process_with_openface(
        [str(frames_dir)],
        output_dirs=[str(output_dir / "openface")],
        config=openface_config,
    )

    return {
        "success": True,
        "video": str(video_path),
        "ffmpeg_result": ffmpeg_result,
        "openface_result": openface_result[0] if openface_result else None,
        "output_dir": str(output_dir),
    }


def main() -> None:
    """Main entry point for video processing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract and analyze facial keypoints from videos"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input video file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=DEFAULT_THREADS,
        help="Number of threads for processing",
    )
    args = parser.parse_args()

    # Configure FFmpeg
    ffmpeg_config = FFmpegConfig(
        input_path=Path(args.input) if args.input else DEFAULT_INPUT_ROOT,
        output_root=Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT,
        ffmpeg_path=DEFAULT_FFMPEG_PATH,
        threads=args.threads,
        frame_pattern=DEFAULT_FRAME_PATTERN,
        output_format=DEFAULT_FORMAT,
        quality=DEFAULT_QUALITY,
        video_extensions=list(VALID_VIDEO_EXTENSIONS),
        overwrite=DEFAULT_OVERWRITE,
    )

    # Get OpenFace configuration
    openface_config = get_config()

    input_path = Path(args.input) if args.input else DEFAULT_INPUT_ROOT
    output_root = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT

    if input_path.is_file():
        # Process single video
        result = process_video_pipeline(
            input_path,
            output_root,
            ffmpeg_config,
            openface_config,
        )
        if result["success"]:
            logger.info(f"Successfully processed {input_path}")
        else:
            logger.error(f"Failed to process {input_path}")

    else:
        # Process all videos in directory
        results = []
        for video_path in input_path.rglob("*"):
            if video_path.suffix.lower() in VALID_VIDEO_EXTENSIONS:
                rel_path = video_path.relative_to(input_path)
                output_dir = output_root / rel_path.parent / video_path.stem
                result = process_video_pipeline(
                    video_path,
                    output_dir,
                    ffmpeg_config,
                    openface_config,
                )
                results.append(result)

        # Report results
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        logger.info(
            f"Processed {len(results)} videos. "
            f"Success: {successful}, Failed: {failed}"
        )


if __name__ == "__main__":
    main()
