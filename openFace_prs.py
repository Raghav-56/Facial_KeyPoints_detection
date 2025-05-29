import time
import subprocess
from pathlib import Path
from tqdm import tqdm

import config.defaults as cfg
from config.config import get_config
from config.logger_config import logger
from lib.video import run_openface
from lib.csv_util import (
    extract_csv_columns,
    detect_key_frames,
    read_csv_with_openface_handling,
)
from lib.file_utils import get_file_list, safe_mkdir
from lib.video_filename_parser import parse_video_filename


def process_video(video_path, output_dir=None, config=None):
    video_path = Path(video_path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = video_path.parent / f"{video_path.stem}_output"

    openface_out = output_dir
    safe_mkdir(str(openface_out))

    logger.info("Processing video: %s", video_path.name)

    if not config:
        return {
            "success": False,
            "video": str(video_path),
            "error": "No config provided",
        }

    if not video_path.exists():
        return {
            "success": False,
            "video": str(video_path),
            "error": "Video file does not exist",
        }

    openface_binary = config.get("openface_binary")
    if not openface_binary or not Path(openface_binary).exists():
        return {
            "success": False,
            "video": str(video_path),
            "error": f"OpenFace binary not found: {openface_binary}",
        }

    try:
        csv_file = run_openface(
            str(video_path), str(openface_out), openface_binary
        )

        if not csv_file:
            logger.error(
                "OpenFace returned no CSV file path for %s", video_path
            )
            return {
                "success": False,
                "video": str(video_path),
                "error": "OpenFace returned no CSV file path",
            }

        csv_path = Path(csv_file)
        if not csv_path.exists():
            logger.error("OpenFace CSV file does not exist: %s", csv_file)
            return {
                "success": False,
                "video": str(video_path),
                "error": f"OpenFace CSV file does not exist: {csv_file}",
            }

        if csv_path.stat().st_size == 0:
            logger.error("OpenFace generated empty CSV file: %s", csv_file)
            return {
                "success": False,
                "video": str(video_path),
                "error": "OpenFace generated empty CSV file",
            }

        logger.info(
            "OpenFace generated CSV: %s (size: %d bytes)",
            csv_file,
            csv_path.stat().st_size,
        )

        if config.get("extract_csv") and config.get("csv_columns"):
            csv_name = f"extracted_{csv_path.name}"
            extracted_csv = str(openface_out / csv_name)
            try:
                csv_file = extract_csv_columns(
                    csv_file, extracted_csv, config["csv_columns"]
                )
                logger.info("Extracted CSV columns to: %s", extracted_csv)
            except (OSError, ValueError, KeyError) as e:
                logger.warning("Failed to extract CSV columns: %s", e)

        try:
            df = read_csv_with_openface_handling(csv_file)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            return {
                "success": False,
                "video": str(video_path),
                "error": f"Failed to read CSV file: {e}",
            }

        if df.empty:
            return {
                "success": False,
                "video": str(video_path),
                "error": "Empty CSV file generated",
            }

        logger.info(
            "Successfully read CSV with %d rows and %d columns",
            len(df),
            len(df.columns),
        )

        frame_col = next(
            (col for col in df.columns if col.lower() == "frame"), None
        )

        if "AU_sum" not in df.columns:
            au_columns = [
                col
                for col in df.columns
                if col.startswith("AU") and col.endswith("_r")
            ]
            if au_columns:
                df["AU_sum"] = df[au_columns].sum(axis=1)
                logger.info(
                    "Created AU_sum column from %d AU columns", len(au_columns)
                )
            else:
                logger.warning(
                    "No AU intensity columns found for %s", video_path
                )
                df["AU_sum"] = 0

        key_frames = detect_key_frames(
            df,
            frame_col=frame_col,
            value_col="AU_sum",
            threshold=config.get("au_sum_threshold", 0),
            method="combined",
            min_peaks=15,
            max_peaks=40,
        )

        if key_frames:
            csv_stem = csv_path.stem
            key_frames_csv = str(openface_out / f"{csv_stem}_keyframes.csv")
            try:
                df.iloc[key_frames].to_csv(key_frames_csv, index=False)
                logger.info(
                    "Saved %d key frames to %s",
                    len(key_frames),
                    key_frames_csv
                )
            except (OSError, ValueError) as e:
                logger.error("Failed to save key frames CSV: %s", e)

        return {
            "success": True,
            "video": str(video_path),
            "key_frames": key_frames,
            "output_dir": str(output_dir),
            "csv_file": csv_file,
            "metadata": parse_video_filename(video_path.name),
        }

    except subprocess.CalledProcessError as e:
        logger.error("OpenFace subprocess failed for %s: %s", video_path, e)
        return {
            "success": False,
            "video": str(video_path),
            "error": f"OpenFace subprocess failed: {e}",
        }
    except (OSError, RuntimeError) as e:
        logger.error("Unexpected error processing %s: %s", video_path, e)
        return {
            "success": False,
            "video": str(video_path),
            "error": f"Unexpected error: {e}",
        }


def process_videos(videos, output_dirs=None, config=None):
    start_time = time.time()
    results = []
    if not output_dirs:
        output_dirs = [None] * len(videos)
    video_pairs = zip(videos, output_dirs)
    for video, output_dir in tqdm(
        video_pairs, desc="Processing videos", total=len(videos)
    ):
        try:
            result = process_video(video, output_dir, config)
            results.append(result)
        except (IOError, RuntimeError) as e:
            logger.error("Error processing %s: %s", video, e)
            results.append({"success": False, "video": video, "error": str(e)})
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful
    total = len(videos)
    duration = time.time() - start_time
    logger.info(
        "Processed %d videos in %.2fs. Success: %d, Failed: %d",
        total,
        duration,
        successful,
        failed,
    )
    return results


def main():
    config = get_config()
    videos = []
    output_dirs = []

    if config.get("video"):
        videos = config["video"]
        outputs = config.get("openface_output", [None] * len(videos))
        output_dirs = [Path(out).parent if out else None for out in outputs]
    elif config.get("input_root"):
        input_root_path = Path(config["input_root"])
        if not input_root_path.exists():
            logger.error(
                "Input root directory does not exist: %s", input_root_path
            )
            return 1

        videos = get_file_list(
            config["input_root"], cfg.SETTINGS["valid_extensions"]
        )
        if not videos:
            logger.warning(
                "No valid video files found in %s", input_root_path
            )
            return 0

        output_root = Path(config["output_root"])
        output_dirs = []
        for v in videos:
            v_path = Path(v)
            rel_path = (
                v_path.parent.relative_to(input_root_path)
                if input_root_path != v_path.parent
                else Path("")
            )
            out_dir = str(output_root / rel_path / v_path.stem)
            output_dirs.append(out_dir)
    else:
        logger.error(
            "No input specified. Use --input or --video to specify input."
        )
        return 1

    if not videos:
        logger.warning("No videos to process")
        return 0

    process_videos(videos, output_dirs, config)
    return 0


if __name__ == "__main__":
    exit(main())
