import sys
import json
from pathlib import Path
from lib.csv_util import COLUMNS_TO_EXTRACT as DEFAULT_CSV_COLUMNS

# Base directory for the application
BASE_DIR = Path(__file__).resolve().parent.parent

# Default paths
DEFAULT_INPUT_ROOT = Path(r"D:\Programming\DIC\Samvedna_Sample\Sample_vid")
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\Programming\DIC\Samvedna_Sample\Sample_vid\extracted_frames"
)
DEFAULT_FFMPEG_PATH = Path("ffmpeg")
DEFAULT_LOG_FILE = Path("extraction_log.csv")
DEFAULT_METADATA_CSV = Path("video_metadata.csv")

# Video processing settings
DEFAULT_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
DEFAULT_QUALITY = 1  # Highest quality (1-31 scale where lower is better)
DEFAULT_FORMAT = "png"
DEFAULT_FRAME_PATTERN = "frame_%04d.png"
DEFAULT_THREADS = 4
DEFAULT_OVERWRITE = False
DEFAULT_MAINTAIN_STRUCTURE = True

# OpenFace settings
OPENFACE_BINARIES = {
    "win32": "OpenFace_2.2.0_win_x64/FeatureExtraction.exe",
    "darwin": "OpenFace/build/bin/FeatureExtraction",  # macOS
    "linux": "OpenFace/build/bin/FeatureExtraction",  # Linux
}
DEFAULT_OPENFACE_BINARY = str(BASE_DIR / OPENFACE_BINARIES.get(sys.platform, ""))

# Default Action Unit thresholds
DEFAULT_AU_THRESHOLDS = {
    "AU01_r": 0.3,  # Inner brow raiser
    "AU02_r": 0.3,  # Outer brow raiser
    "AU04_r": 0.5,  # Brow lowerer
    "AU05_r": 0.3,  # Upper lid raiser
    "AU06_r": 0.3,  # Cheek raiser
    "AU07_r": 0.3,  # Lid tightener
    "AU09_r": 0.3,  # Nose wrinkler
    "AU10_r": 0.4,  # Upper lip raiser
    "AU12_r": 0.4,  # Lip corner puller (smile)
    "AU14_r": 0.3,  # Dimpler
    "AU15_r": 0.3,  # Lip corner depressor
    "AU17_r": 0.3,  # Chin raiser
    "AU20_r": 0.3,  # Lip stretcher
    "AU23_r": 0.3,  # Lip tightener
    "AU25_r": 0.3,  # Lips part
    "AU26_r": 0.3,  # Jaw drop
    "AU45_r": 0.3,  # Blink
}

# Application-wide default settings
SETTINGS = {
    "openface_binary": DEFAULT_OPENFACE_BINARY,
    "valid_extensions": DEFAULT_VIDEO_EXTENSIONS,
    "au_thresholds": DEFAULT_AU_THRESHOLDS,
    "au_sum_threshold": 1.0,
    "csv_columns": DEFAULT_CSV_COLUMNS,
    "extract_csv": False,
    "output_root": str(BASE_DIR / "output"),
    "log_level": "INFO",
}

# Load user-specific defaults if available
USER_CONFIG_PATH = BASE_DIR / "user_config.json"
try:
    if USER_CONFIG_PATH.exists():
        SETTINGS.update(json.loads(USER_CONFIG_PATH.read_text()))
except Exception as e:
    print(f"Failed to load user config: {e}", file=sys.stderr)
