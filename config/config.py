import argparse
import os
import json
from pathlib import Path
import config.defaults as defaults
from config.logger_config import logger


def parse_au_thresholds(threshold_str):
    if not threshold_str:
        return {}

    thresholds = {}
    for pair in threshold_str.split(","):
        try:
            name, value = pair.split(":")
            thresholds[name.strip()] = float(value.strip())
        except ValueError:
            logger.warning("Invalid AU threshold format: %s", pair)
    return thresholds


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=(
            "Process videos using OpenFace to extract facial expressions "
            "and key frames"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_argument_group("Input/Output")
    input_group.add_argument(
        "--input",
        help="Root directory containing input videos",
        dest="input_root"
    )
    input_group.add_argument(
        "--output",
        help="Root directory for output files",
        dest="output_root"
    )
    input_group.add_argument(
        "--video",
        action="append",
        help="Path to individual video file"
    )
    input_group.add_argument(
        "--openface_output",
        action="append",
        help="Path for OpenFace output"
    )
    input_group.add_argument(
        "--frames_output",
        action="append",
        help="Path for extracted frames"
    )
    input_group.add_argument(
        "--diagnose_csv",
        help="Run in diagnostic mode on specified CSV file"
    )

    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--openface_binary",
        help="Path to OpenFace FeatureExtraction binary",
        default=defaults.SETTINGS.get("openface_binary")
    )
    config_group.add_argument(
        "--au_thresholds",
        help="AU detection thresholds (format: AU01_r:0.5,AU02_r:0.3)",
        default=""
    )
    config_group.add_argument(
        "--au_sum_threshold",
        type=float,
        help="Threshold for AU sum-based frame detection",
        default=defaults.SETTINGS.get("au_sum_threshold", 1.0)
    )
    config_group.add_argument(
        "--extract_csv",
        action="store_true",
        help="Extract specific columns from OpenFace CSV output"
    )
    config_group.add_argument(
        "--csv_columns",
        nargs="+",
        help="Columns to extract from CSV",
        default=defaults.SETTINGS.get("csv_columns", [])
    )
    config_group.add_argument(
        "--config_file",
        help="Path to JSON configuration file"
    )

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    log_group.add_argument(
        "--rotate_logs",
        action="store_true",
        help="Enable log rotation"
    )
    log_group.add_argument(
        "--fallback_extraction",
        action="store_true",
        help="Enable fallback extraction methods"
    )
    log_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )

    return parser.parse_args()


def load_config_file(path):
    if not path or not os.path.exists(path):
        return {}
        
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load configuration file %s: %s", path, e)
        return {}


def set_log_level(level):
    try:
        from lib.log_utils import set_log_level as set_level
        set_level(level)
    except ImportError:
        numeric_level = getattr(logger, level.upper())
        logger.setLevel(numeric_level)


def get_config():
    config = defaults.SETTINGS.copy()
    args = parse_command_line()

    # Merge configurations in order of priority
    if args.config_file:
        config.update(load_config_file(args.config_file))
    
    arg_dict = {k: v for k, v in vars(args).items() if v is not None}
    config.update(arg_dict)

    # Configure logging
    set_log_level(config.get("log_level", "INFO"))
    if config.get("rotate_logs"):
        try:
            from lib.log_utils import configure_rotating_logs
            configure_rotating_logs()
        except ImportError:
            logger.warning("Log rotation not available")

    # Process AU thresholds
    thresholds = config.get("au_thresholds", "")
    config["au_thresholds_parsed"] = (
        parse_au_thresholds(thresholds) or 
        defaults.SETTINGS.get("au_thresholds", {})
    )

    # Set default output path if needed
    if args.input_root and not args.output_root:
        config["output_root"] = str(Path(args.input_root).parent / "output")

    # Validate required input
    if not any([args.input_root, args.video, args.diagnose_csv]):
        logger.error("No input specified (--input, --video, or --diagnose_csv)")
        return config

    non_private_config = {k: v for k, v in config.items() if not k.startswith("_")}
    logger.debug("Configuration: %s", non_private_config)
    return config
