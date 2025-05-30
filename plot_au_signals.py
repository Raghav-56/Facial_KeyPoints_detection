"""
Plot utility for visualizing Action Unit signals from OpenFace CSV data.
"""

import matplotlib.pyplot as plt
import argparse
import os
from config.logger_config import logger
from lib.csv_util import read_csv_with_openface_handling
from scipy.signal import find_peaks


def plot_au_sum(df, threshold=0, save_path=None):
    """Plot the sum of Action Unit intensities across frames."""
    
    # Calculate AU sum if not already in DataFrame
    if "AU_sum" not in df.columns:
        au_columns = [
            col
            for col in df.columns
            if col.startswith("AU") and col.endswith("_r")
        ]
        if not au_columns:
            logger.warning(
                "No AU intensity columns found, using all numeric columns"
            )
            numeric_cols = df.select_dtypes(include=["number"]).columns
            au_columns = [
                col for col in numeric_cols if "frame" not in col.lower()
            ]

        if not au_columns:
            raise ValueError(
                "No suitable columns found for AU sum calculation"
            )

        df["AU_sum"] = df[au_columns].sum(axis=1)

    # Create plot
    plt.figure(figsize=(14, 10))
    plt.plot(
        df.index,
        df["AU_sum"],
        label="AU Sum",
        color="magenta",
        linewidth=3,
        marker="o",
        markersize=8,
    )
    plt.xlabel("Frame Number", fontsize=14)
    plt.ylabel("Sum of AU Intensities", fontsize=14)
    plt.title("Action Unit Sum Over Time", fontsize=16, fontweight="bold")

    peaks = []
    if threshold > 0:
        # Find peaks above threshold
        peaks, _ = find_peaks(df["AU_sum"], height=threshold)
        if len(peaks) > 0:
            plt.plot(
                df.index[peaks],
                df["AU_sum"].iloc[peaks],
                "rx",
                markersize=12,
                label=f"Peaks (threshold={threshold})",
            )

    plt.legend(fontsize="large")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Plot saved to: %s", save_path)
        except (OSError, ValueError) as e:
            logger.error("Failed to save plot: %s", e)
    else:
        plt.show()

    return list(peaks) if threshold > 0 else []


def plot_individual_aus(df, top_n=None, save_path=None):
    """Plot individual Action Unit intensity values over time."""

    # Filter only AU intensity columns
    au_cols = [
        col
        for col in df.columns
        if col.startswith("AU") and col.endswith("_r")
    ]

    if not au_cols:
        logger.warning(
            "No AU intensity columns found, trying alternative patterns"
        )
        au_cols = [
            col
            for col in df.columns
            if "AU" in col.upper() and any(c.isdigit() for c in col)
        ]

    if not au_cols:
        raise ValueError(
            "No Action Unit intensity columns found in the CSV file"
        )

    # If top_n specified, select only the top N AUs with highest mean values
    if top_n and top_n < len(au_cols):
        au_means = df[au_cols].mean()
        top_aus = au_means.nlargest(top_n).index.tolist()
        au_cols = top_aus

    plt.figure(figsize=(12, 8))
    for col in au_cols:
        plt.plot(df.index, df[col], label=col, alpha=0.7)

    plt.xlabel("Frame Number", fontsize=12)
    plt.ylabel("AU Intensity Value", fontsize=12)
    plt.title("Action Unit Intensities Over Time", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Plot saved to: %s", save_path)
        except (OSError, ValueError) as e:
            logger.error("Failed to save plot: %s", e)
    else:
        plt.show()


def main():
    """Main entry point for the AU signal plotting tool."""
    parser = argparse.ArgumentParser(
        description="Plot Action Unit (AU) signals from OpenFace CSV data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file with OpenFace AU intensity values",
    )
    parser.add_argument(
        "--au_sum_threshold",
        type=float,
        default=0,
        help="Threshold for detecting peaks in the AU sum signal",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Only plot the top N AUs with highest mean intensity values",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save plot images instead of displaying them",
    )
    args = parser.parse_args()

    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        logger.error("CSV file not found: %s", args.csv_file)
        return 1

    # Create save directory if specified
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Generate save paths if needed
    au_signals_save_path = None
    au_sum_save_path = None
    if args.save_dir:
        base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
        au_signals_save_path = os.path.join(
            args.save_dir, f"{base_name}_au_signals.png"
        )
        au_sum_save_path = os.path.join(
            args.save_dir, f"{base_name}_au_sum.png"
        )

    try:
        df = read_csv_with_openface_handling(args.csv_file)
        
        if df.empty:
            logger.error("CSV file is empty: %s", args.csv_file)
            return 1
            
        logger.info(
            "Loaded CSV file with %d frames and %d columns",
            len(df),
            len(df.columns),
        )

        # Plot individual AU signals
        plot_individual_aus(
            df, top_n=args.top_n, save_path=au_signals_save_path
        )

        # Plot AU sum and detect peaks
        peaks = plot_au_sum(
            df, threshold=args.au_sum_threshold, save_path=au_sum_save_path
        )
        if peaks:
            logger.info(
                "Detected %d key frames at indices: %s", len(peaks), peaks
            )
        else:
            logger.info(
                "No peaks detected with threshold %s", args.au_sum_threshold
            )

        return 0

    except (OSError, ValueError, KeyError) as e:
        logger.error("Error processing CSV file: %s", e)
        return 1


if __name__ == "__main__":
    exit(main())
