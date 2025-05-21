import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from config.logger_config import logger


def compute_au_sum(df):
    """Compute the sum of all Action Unit (AU) intensity values."""
    if "AU_sum" not in df.columns:
        au_columns = [
            col for col in df.columns 
            if col.startswith("AU") and col.endswith("_r")
        ]
        df["AU_sum"] = df[au_columns].sum(axis=1) if au_columns else 0
        if not au_columns:
            logger.warning("No AU intensity columns found in DataFrame")
    return df


def read_csv_with_openface_handling(csv_path):
    """Read CSV with OpenFace comment handling."""
    try:
        return pd.read_csv(csv_path, comment="//")
    except pd.errors.ParserError as e:
        logger.debug(f"Failed to parse with comment handling: {e}")
        return pd.read_csv(csv_path)


def extract_csv_columns(csv_file, output_csv, columns_to_extract):
    """Extract and save specific columns from a CSV file."""
    csv_path = Path(csv_file)
    df = read_csv_with_openface_handling(csv_path)
    df.columns = df.columns.str.strip()
    
    available_cols = [col for col in columns_to_extract if col in df.columns]
    extracted = compute_au_sum(df[available_cols].copy())
    extracted.to_csv(output_csv, index=False)
    logger.info(f"Extracted {len(available_cols)} columns to {output_csv}")
    
    return output_csv


def smooth_signal(signal, window_length=15, polyorder=3):
    """Smooth a signal using Savitzky-Golay filter."""
    if len(signal) < 5:
        return signal
        
    window_length = min(15, max(5, len(signal) // 20))
    window_length += (window_length % 2 == 0)
    polyorder = min(polyorder, window_length - 1)
    
    try:
        pad_size = window_length // 2
        padded_signal = np.pad(signal, (pad_size, pad_size), mode="reflect")
        smoothed = savgol_filter(padded_signal, window_length, polyorder)
        return smoothed[pad_size:-pad_size]
    except Exception as e:
        logger.warning(f"Signal smoothing failed: {e}")
        return signal


def detect_key_frames(
        df,
        frame_col=None,
        value_col="AU_sum",
        threshold=None, 
        method="combined",
        min_distance=15,
        min_peaks=3,
        max_peaks=7,
        adaptive=True,
        smoothing=True,
        percentile=95
):
    """Detect key frames using specified method."""
    # Initial setup and validation
    if value_col == "AU_sum" and value_col not in df.columns:
        df = compute_au_sum(df)
    if value_col not in df.columns:
        logger.error(f"Column {value_col} not found in dataframe")
        return []

    signal = df[value_col].values
    if len(signal) < 10:
        indices = list(range(len(signal)))
        if frame_col in df.columns:
            return df.iloc[indices][frame_col].tolist()
        return indices

    # Signal processing
    smoothed_signal = smooth_signal(signal) if smoothing else signal
    
    # Calculate threshold
    if adaptive:
        percentile_val = np.percentile(smoothed_signal, max(percentile, 97))
        mean_based = np.mean(smoothed_signal) + np.std(smoothed_signal)
        actual_threshold = max(threshold or 0, percentile_val, mean_based)
    else:
        actual_threshold = threshold or np.percentile(smoothed_signal, percentile)

    min_dist = max(min_distance, len(signal) // (max_peaks * 2))
    key_frames = []

    # Frame detection logic
    if method in ["simple", "peaks", "combined"]:
        if method == "simple":
            indices = np.argsort(smoothed_signal)[::-1]
            for idx in indices:
                is_far = all(abs(idx - kf) >= min_dist for kf in key_frames)
                if not key_frames or is_far:
                    key_frames.append(idx)
                if len(key_frames) >= max_peaks:
                    break
        else:
            try:
                prominence = 0.1 * (np.max(smoothed_signal) - np.min(smoothed_signal))
                peaks, _ = find_peaks(
                    smoothed_signal,
                    height=actual_threshold,
                    distance=min_dist,
                    prominence=prominence
                )
                key_frames = peaks.tolist()
                logger.debug(f"Peak detection found {len(key_frames)} peaks")
            except Exception as e:
                logger.warning(f"Peak detection failed: {e}")

    # Add threshold-based detection if needed
    needs_threshold = (method == "threshold" or
                      (method == "combined" and len(key_frames) < min_peaks))
    
    if method in ["threshold", "combined"] and needs_threshold:
        high_threshold = actual_threshold * 1.1
        threshold_frames = np.where(smoothed_signal > high_threshold)[0].tolist()
        
        if method == "combined" and key_frames:
            additional = [
                tf for tf in threshold_frames
                if all(abs(tf - kf) >= min_dist for kf in key_frames)
            ]
            key_frames = sorted(set(key_frames + additional))
        else:
            key_frames = [
                tf for i, tf in enumerate(sorted(threshold_frames))
                if i == 0 or abs(tf - key_frames[-1]) >= min_dist
            ]

    # Ensure minimum/maximum number of frames
    if len(key_frames) < min_peaks:
        indices = np.argsort(smoothed_signal)[::-1]
        key_frames = []
        for idx in indices:
            is_far = all(abs(idx - kf) >= min_dist for kf in key_frames)
            if not key_frames or is_far:
                key_frames.append(idx)
            if len(key_frames) >= min_peaks:
                break

    if len(key_frames) > max_peaks:
        frame_values = [(i, smoothed_signal[i]) for i in key_frames]
        sorted_frames = sorted(frame_values, key=lambda x: x[1], reverse=True)
        key_frames = sorted(i for i, _ in sorted_frames[:max_peaks])

    # Return results
    if frame_col in df.columns and key_frames:
        valid_frames = [
            df.iloc[idx][frame_col] for idx in key_frames 
            if 0 <= idx < len(df)
        ]
        logger.info(f"Detected {len(valid_frames)} key frames using {method} method")
        return valid_frames
    
    logger.info(f"Detected {len(key_frames)} key frames using {method} method")
    return sorted(key_frames)


def create_key_frames_csv(
        input_csv_path,
        output_csv_path=None,
        key_frames=None,
        frame_col="frame",
        value_col="AU_sum",
        method="combined",
        min_distance=15,
        min_peaks=3,
        max_peaks=7,
        adaptive=True,
        smoothing=True,
        percentile=95
):
    """Create a CSV file containing only the data from key frames."""
    input_path = Path(input_csv_path)
    if output_csv_path is None:
        suffix = "_keyframes" + input_path.suffix
        output_csv_path = input_path.parent / f"{input_path.stem}{suffix}"
    
    logger.info(f"Processing {input_path}")
    df = read_csv_with_openface_handling(input_path)
    if value_col == "AU_sum" and value_col not in df.columns:
        df = compute_au_sum(df)
    
    if key_frames is None:
        key_frames = detect_key_frames(
            df, frame_col=frame_col, value_col=value_col,
            method=method, min_distance=min_distance,
            min_peaks=min_peaks, max_peaks=max_peaks,
            adaptive=adaptive, smoothing=smoothing,
            percentile=percentile
        )
    
    if frame_col in df.columns:
        key_frames_df = (df[df[frame_col].isin(key_frames)]
                        .sort_values(by=frame_col))
    else:
        key_frames_df = df.iloc[key_frames]
    
    try:
        key_frames_df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved {len(key_frames)} frames to {output_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save key frames CSV: {e}")
        raise
    
    return output_csv_path, key_frames


# Default columns to extract from OpenFace CSV output
OPENFACE_INTENSITY_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU45_r"
]

OPENFACE_PRESENCE_COLS = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c", "AU09_c",
    "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c",
    "AU25_c", "AU26_c", "AU28_c", "AU45_c"
]

COLUMNS_TO_EXTRACT = ["frame", "timestamp"] + OPENFACE_INTENSITY_COLS + OPENFACE_PRESENCE_COLS



""" COLUMNS_TO_EXTRACT = [
    "frame",
    "timestamp",
    "AU01_r",  # Inner brow raiser
    "AU02_r",  # Outer brow raiser
    "AU04_r",  # Brow lowerer
    "AU05_r",  # Upper lid raiser
    "AU06_r",  # Cheek raiser
    "AU07_r",  # Lid tightener
    "AU09_r",  # Nose wrinkler
    "AU10_r",  # Upper lip raiser
    "AU12_r",  # Lip corner puller (smile)
    "AU14_r",  # Dimpler
    "AU15_r",  # Lip corner depressor
    "AU17_r",  # Chin raiser
    "AU20_r",  # Lip stretcher
    "AU23_r",  # Lip tightener
    "AU25_r",  # Lips part
    "AU26_r",  # Jaw drop
    "AU45_r",  # Blink
    "AU01_c",  # Binary presence of AU01
    "AU02_c",  # Binary presence of AU02
    "AU04_c",  # Binary presence of AU04
    "AU05_c",  # Binary presence of AU05
    "AU06_c",  # Binary presence of AU06
    "AU07_c",  # Binary presence of AU07
    "AU09_c",  # Binary presence of AU09
    "AU10_c",  # Binary presence of AU10
    "AU12_c",  # Binary presence of AU12
    "AU14_c",  # Binary presence of AU14
    "AU15_c",  # Binary presence of AU15
    "AU17_c",  # Binary presence of AU17
    "AU20_c",  # Binary presence of AU20
    "AU23_c",  # Binary presence of AU23
    "AU25_c",  # Binary presence of AU25
    "AU26_c",  # Binary presence of AU26
    "AU28_c",  # Binary presence of AU28
    "AU45_c",  # Binary presence of AU45
]
 """
