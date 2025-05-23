import time
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
    frames_out = output_dir / "frames"
    safe_mkdir(str(openface_out))
    safe_mkdir(str(frames_out))
    logger.info("Processing video: %s", video_path.name)
    if not config:
        return {
            "success": False,
            "video": str(video_path),
            "error": "No config",
        }
    csv_file = run_openface(
        str(video_path), str(openface_out), config.get("openface_binary")
    )
    if config.get("extract_csv") and config.get("csv_columns"):
        csv_name = f"extracted_{Path(csv_file).name}"
        extracted_csv = str(Path(openface_out) / csv_name)
        csv_file = extract_csv_columns(csv_file, extracted_csv, config["csv_columns"])
    df = read_csv_with_openface_handling(csv_file)
    frame_col = next((col for col in df.columns if col.lower() == "frame"), None)
    key_frames = detect_key_frames(
        df,
        frame_col=frame_col,
        value_col="AU_sum",
        threshold=config.get("au_sum_threshold"),
        method="combined",
        min_peaks=15,
        max_peaks=40,
    )
    if key_frames:
        csv_stem = Path(csv_file).stem
        key_frames_csv = str(openface_out / f"{csv_stem}_keyframes.csv")
        df.iloc[key_frames].to_csv(key_frames_csv, index=False)
    return {
        "success": bool(key_frames),
        "video": str(video_path),
        "key_frames": key_frames,
        "output_dir": str(output_dir),
        "csv_file": csv_file,
        "metadata": parse_video_filename(video_path.name),
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
        videos = get_file_list(config["input_root"], cfg.SETTINGS["valid_extensions"])
        input_root = Path(config["input_root"])
        output_root = Path(config["output_root"])
        output_dirs = []
        for v in videos:
            v_path = Path(v)
            rel_path = (
                v_path.parent.relative_to(input_root)
                if input_root != v_path.parent
                else ""
            )
            out_dir = str(output_root / rel_path / v_path.stem)
            output_dirs.append(out_dir)
    else:
        logger.error("No input specified. Use --input or --video to specify input.")
        return 1
    process_videos(videos, output_dirs, config)
    return 0


if __name__ == "__main__":
    exit(main())
