# Extracting I keyframes from videos using FFmpeg

__author__ = {"name": "Raghav Gupta", "username": "Raghav-56"}

import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from config.logger_config import logger
from config.defaults import (
    DEFAULT_VIDEO_EXTENSIONS,
    DEFAULT_THREADS,
    DEFAULT_QUALITY,
    DEFAULT_FORMAT,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_FFMPEG_PATH,
    DEFAULT_FRAME_PATTERN,
    DEFAULT_OVERWRITE,
    DEFAULT_MAINTAIN_STRUCTURE,
    DEFAULT_LOG_FILE,
    DEFAULT_METADATA_CSV,
)
from lib.video_filename_parser import parse_video_filename
from lib.data_handling import DataHandler
from lib.extract_i_frames import build_extract_frames_command

# logger = logger("video_processor", "logs")


@dataclass
class Config:
    """Configuration for frame extraction from videos."""

    input_path: Path = DEFAULT_INPUT_ROOT
    output_root: Optional[Path] = DEFAULT_OUTPUT_ROOT
    ffmpeg_path: Path = DEFAULT_FFMPEG_PATH
    threads: int = DEFAULT_THREADS
    frame_pattern: str = DEFAULT_FRAME_PATTERN
    output_format: str = DEFAULT_FORMAT
    video_extensions: List[str] = field(
        default_factory=lambda: DEFAULT_VIDEO_EXTENSIONS
    )
    quality: int = DEFAULT_QUALITY
    overwrite: bool = DEFAULT_OVERWRITE
    maintain_structure: bool = DEFAULT_MAINTAIN_STRUCTURE
    use_parent_dir: bool = False
    log_file: Optional[Path] = DEFAULT_LOG_FILE
    metadata_csv: Optional[Path] = DEFAULT_METADATA_CSV


class VideoProcessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_handler = DataHandler(
            log_file=cfg.log_file, metadata_csv=cfg.metadata_csv
        )

    def process_video(self, video_path: Path) -> Dict:
        """Extract I-frames from video file."""
        metadata = parse_video_filename(video_path.name)
        output_root = self.cfg.output_root or Path("output")
        output_dir = self.data_handler.get_output_dir(
            video_path,
            self.cfg.input_path,
            output_root,
            self.cfg.maintain_structure,
            self.cfg.use_parent_dir,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = build_extract_frames_command(
                self.cfg, video_path, output_dir
            )
            logger.info("Running FFmpeg command: %s", " ".join(map(str, cmd)))

            subprocess.run(cmd, capture_output=True, check=True, text=True)

            frame_count = self.data_handler.count_extracted_frames(
                output_dir, self.cfg.output_format
            )

            if frame_count == 0:
                logger.warning("No frames extracted from %s", video_path)
                return {
                    "video_path": str(video_path),
                    "status": "failed",
                    "error": "No frames extracted",
                }

            logger.info(
                "Successfully extracted %d frames from %s",
                frame_count,
                video_path.name,
            )

            return {
                "video_path": str(video_path),
                "status": "success",
                "metadata": metadata,
                "output_dir": str(output_dir),
                "frame_count": frame_count,
            }
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg failed for %s: %s", video_path.name, e.stderr)
            return {
                "video_path": str(video_path),
                "status": "failed",
                "error": f"FFmpeg error: {e.stderr}",
            }
        except OSError as e:
            logger.error("Failed to process %s: %s", video_path.name, e)
            return {
                "video_path": str(video_path),
                "status": "failed",
                "error": str(e),
            }

    def process_videos(self):
        """Process all videos in input path."""
        video_files = self.data_handler.find_video_files(
            self.cfg.input_path, self.cfg.video_extensions
        )
        results = [
            self.process_video(video_path) for video_path in video_files
        ]
        self.data_handler.save_results(results)


def main():
    cfg = Config()
    cfg.use_parent_dir = True
    VideoProcessor(cfg).process_videos()


if __name__ == "__main__":
    main()
