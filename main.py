from pathlib import Path
from typing import Dict
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
from lib.face_extraction import extract_faces_from_video_processing


VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def process_video_pipeline(
    video_path: Path,
    output_dir: Path,
    ffmpeg_config: FFmpegConfig,
    openface_config: Dict,
) -> Dict:
    """Process a single video through the complete pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract I-frames with FFmpeg
    ffmpeg_processor = FFmpegProcessor(ffmpeg_config)
    ffmpeg_result = ffmpeg_processor.process_video(video_path)

    if not ffmpeg_result.get("status") == "success":
        logger.error("Failed to extract frames from %s", video_path)
        return {
            "success": False,
            "video": str(video_path),
            "error": "FFmpeg extraction failed",
        }

    # Step 2: Process original video with OpenFace (not frames)
    openface_output_dir = output_dir / "openface"
    openface_output_dir.mkdir(parents=True, exist_ok=True)

    # Process the original video file with OpenFace
    openface_result = process_with_openface(
        [str(video_path)],  # Pass video file, not frames directory
        output_dirs=[str(openface_output_dir)],
        config=openface_config,
    )

    if not openface_result or not openface_result[0].get("success"):
        logger.error("OpenFace processing failed for %s", video_path)
        return {
            "success": False,
            "video": str(video_path),
            "error": "OpenFace processing failed",
            "ffmpeg_result": ffmpeg_result,
        }

    # Step 3: Extract face crops using OpenFace landmarks and FFmpeg frames
    openface_csv = openface_result[0].get("csv_file")
    face_extraction_result = {"success": False, "error": "No CSV file"}

    if openface_csv and Path(openface_csv).exists():
        frames_dir = Path(ffmpeg_result["output_dir"])
        face_extraction_dir = output_dir / "face_extraction"
        face_extraction_dir.mkdir(parents=True, exist_ok=True)

        face_extraction_result = extract_faces_from_video_processing(
            frames_dir=frames_dir,
            openface_csv=Path(openface_csv),
            output_dir=face_extraction_dir,
            target_size=(224, 224),
            frame_format=ffmpeg_config.output_format,
        )

        if face_extraction_result.get("success"):
            logger.info("Successfully extracted faces for %s", video_path)
        else:
            logger.warning(
                "Face extraction failed for %s: %s",
                video_path,
                face_extraction_result.get("error", "Unknown error"),
            )
    else:
        logger.warning(
            "No OpenFace CSV found for face extraction: %s", openface_csv
        )

    return {
        "success": True,
        "video": str(video_path),
        "ffmpeg_result": ffmpeg_result,
        "openface_result": openface_result[0] if openface_result else None,
        "face_extraction_result": face_extraction_result,
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

    input_path = Path(args.input) if args.input else DEFAULT_INPUT_ROOT
    output_root = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT

    # Validate input path
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return

    # Ensure output directory exists
    output_root.mkdir(parents=True, exist_ok=True)

    # Configure FFmpeg
    ffmpeg_config = FFmpegConfig(
        input_path=input_path,
        output_root=output_root,
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

    if input_path.is_file():
        # Process single video
        if input_path.suffix.lower() not in VALID_VIDEO_EXTENSIONS:
            logger.error("Invalid video file extension: %s", input_path.suffix)
            return

        output_dir = output_root / input_path.stem
        result = process_video_pipeline(
            input_path,
            output_dir,
            ffmpeg_config,
            openface_config,
        )
        if result["success"]:
            logger.info("Successfully processed %s", input_path)
        else:
            logger.error(
                "Failed to process %s: %s", input_path, result.get("error")
            )

    else:
        # Process all videos in directory
        video_files = list(input_path.rglob("*"))
        video_files = [
            f
            for f in video_files
            if f.suffix.lower() in VALID_VIDEO_EXTENSIONS
        ]

        if not video_files:
            logger.warning("No valid video files found in %s", input_path)
            return

        results = []
        for video_path in video_files:
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
            "Processed %d videos. Success: %d, Failed: %d",
            len(results),
            successful,
            failed,
        )


if __name__ == "__main__":
    main()
