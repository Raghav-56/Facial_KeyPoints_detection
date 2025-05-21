from pathlib import Path


def build_extract_frames_command(cfg, input_path: Path, output_dir: Path) -> list:
    """Build FFmpeg command for extracting I-frames."""
    cmd = [
        str(cfg.ffmpeg_path),
        "-y" if cfg.overwrite else None,
        "-i",
        str(input_path),
        "-threads",
        str(cfg.threads),
    ]
    cmd = [arg for arg in cmd if arg is not None]
    frame_pattern = (
        cfg.frame_pattern
        if "%d" in cfg.frame_pattern or "%0" in cfg.frame_pattern
        else f"frame_%03d.{cfg.output_format}"
    )
    cmd.extend(
        [
            "-vf",
            "select='eq(pict_type,I)'",
            "-vsync",
            "vfr",
            "-q:v",
            str(cfg.quality),
            "-f",
            "image2",
            str(output_dir / frame_pattern),
        ]
    )
    return cmd
