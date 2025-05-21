from pathlib import Path
from typing import List, Dict
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataHandler:
    """Handles data operations for video frame extraction and processing."""
    
    log_file: Path | None = None
    metadata_csv: Path | None = None

    def get_output_dir(
        self,
        video_path: Path,
        input_path: Path,
        output_root: Path,
        maintain_structure: bool,
        use_parent_dir: bool
    ) -> Path:
        """Determine and create output directory for extracted frames."""
        if use_parent_dir:
            output_dir = video_path.parent / video_path.stem
        else:
            base_path = input_path if input_path.is_dir() else input_path.parent
            if maintain_structure and video_path.is_relative_to(base_path):
                rel_path = video_path.relative_to(base_path).with_suffix("")
                output_dir = output_root / rel_path
            else:
                output_dir = output_root / video_path.stem

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir.resolve()

    def find_video_files(
        self,
        input_path: Path,
        valid_extensions: List[str]
    ) -> List[Path]:
        """Find video files with specified extensions."""
        input_path = input_path.resolve()
        extensions = set(ext.lower() for ext in valid_extensions)
        
        if input_path.is_file():
            is_valid = input_path.suffix.lower() in extensions
            return [input_path] if is_valid else []
        
        return [
            p for p in input_path.rglob("*")
            if p.suffix.lower() in extensions
        ]

    def save_results(self, results: List[Dict]) -> None:
        """Save processing results and metadata."""
        if not results:
            return

        # Save both results and metadata if files are specified
        if self.log_file:
            pd.DataFrame(results).to_csv(self.log_file, index=False)
        
        metadata = [r["metadata"] for r in results if r.get("metadata")]
        if metadata and self.metadata_csv:
            pd.DataFrame(metadata).to_csv(self.metadata_csv, index=False)

    def count_extracted_frames(
        self,
        output_dir: Path,
        output_format: str
    ) -> int:
        """Count frames in output directory with specific format."""
        return len(list(output_dir.rglob(f"*.{output_format}")))
