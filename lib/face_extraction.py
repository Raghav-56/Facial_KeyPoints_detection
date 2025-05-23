"""
Face extraction module for extracting face regions using OpenFace keypoints.

This module processes OpenFace CSV output to extract standardized face crops
from original video frames, suitable for emotion recognition training.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import json

from config.logger_config import logger


class FaceExtractor:
    """Extract face regions from frames using OpenFace 68 facial landmarks."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize face extractor.

        Args:
            target_size: Target size for extracted face images (width, height)
        """
        self.target_size = target_size
        self.logger = logger

    def extract_face_bounding_box(
        self, landmarks: np.ndarray, padding_factor: float = 0.3
    ) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box from 68 facial landmarks.

        Args:
            landmarks: Array of shape (68, 2) containing x, y coordinates
            padding_factor: Factor to expand bounding box around face

        Returns:
            Tuple of (x, y, width, height) for bounding box
        """
        # Get face outline landmarks (indices 0-16 for jaw line)
        face_points = landmarks[:17]  # Jaw line points

        # Add eye, nose, and mouth landmarks for better bounding box
        eye_points = landmarks[36:48]  # Both eyes
        nose_points = landmarks[27:36]  # Nose
        mouth_points = landmarks[48:68]  # Mouth

        # Combine all relevant points
        all_points = np.vstack([face_points, eye_points, nose_points, mouth_points])

        # Calculate bounding box
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)

        # Add padding
        width = x_max - x_min
        height = y_max - y_min

        padding_x = int(width * padding_factor)
        padding_y = int(height * padding_factor)

        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        width = width + 2 * padding_x
        height = height + 2 * padding_y

        return int(x_min), int(y_min), int(width), int(height)

    def normalize_landmarks(
        self, landmarks: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Normalize landmarks to [0, 1] range relative to face bounding box.

        Args:
            landmarks: Array of shape (68, 2) containing x, y coordinates
            bbox: Bounding box (x, y, width, height)

        Returns:
            Normalized landmarks array of shape (68, 2)
        """
        x, y, width, height = bbox

        # Normalize to bounding box coordinates
        normalized = landmarks.copy()
        normalized[:, 0] = (landmarks[:, 0] - x) / width
        normalized[:, 1] = (landmarks[:, 1] - y) / height

        # Clamp to [0, 1] range
        normalized = np.clip(normalized, 0, 1)

        return normalized

    def extract_face_crop(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract and resize face crop from image.

        Args:
            image: Input image as numpy array
            bbox: Bounding box (x, y, width, height)

        Returns:
            Resized face crop of target_size
        """
        x, y, width, height = bbox
        h, w = image.shape[:2]

        # Ensure bounding box is within image boundaries
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        x2 = min(w, x + width)
        y2 = min(h, y + height)

        # Extract face region
        face_crop = image[y:y2, x:x2]

        # Resize to target size
        if face_crop.size > 0:
            face_crop = cv2.resize(face_crop, self.target_size)
        else:
            # Create empty image if crop failed
            face_crop = np.zeros(
                (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
            )

        return face_crop

    def parse_openface_landmarks(self, csv_row: pd.Series) -> Optional[np.ndarray]:
        """
        Extract 68 facial landmarks from OpenFace CSV row.

        Args:
            csv_row: Single row from OpenFace CSV output

        Returns:
            Array of shape (68, 2) with x, y coordinates, or None if parsing fails
        """
        try:
            # OpenFace landmarks are stored as x_0, y_0, x_1, y_1, ..., x_67, y_67
            landmarks = np.zeros((68, 2))

            for i in range(68):
                x_col = f"x_{i}"
                y_col = f"y_{i}"

                if x_col in csv_row and y_col in csv_row:
                    landmarks[i, 0] = csv_row[x_col]
                    landmarks[i, 1] = csv_row[y_col]
                else:
                    self.logger.warning(f"Missing landmark columns {x_col}, {y_col}")
                    return None

            return landmarks

        except Exception as e:
            self.logger.error(f"Error parsing landmarks: {e}")
            return None

    def process_frame(self, frame_path: Path, csv_row: pd.Series) -> Optional[Dict]:
        """
        Process a single frame to extract face crop and normalized landmarks.

        Args:
            frame_path: Path to the frame image
            csv_row: Corresponding row from OpenFace CSV

        Returns:
            Dictionary containing face crop, normalized landmarks, and metadata
        """
        try:
            # Load frame image
            image = cv2.imread(str(frame_path))
            if image is None:
                self.logger.error(f"Could not load image: {frame_path}")
                return None

            # Parse landmarks from CSV
            landmarks = self.parse_openface_landmarks(csv_row)
            if landmarks is None:
                return None

            # Calculate face bounding box
            bbox = self.extract_face_bounding_box(landmarks)

            # Extract face crop
            face_crop = self.extract_face_crop(image, bbox)

            # Normalize landmarks
            normalized_landmarks = self.normalize_landmarks(landmarks, bbox)

            return {
                "face_crop": face_crop,
                "normalized_landmarks": normalized_landmarks,
                "bbox": bbox,
                "original_landmarks": landmarks,
                "frame_path": str(frame_path),
                "frame_name": frame_path.name,
            }

        except Exception as e:
            self.logger.error(f"Error processing frame {frame_path}: {e}")
            return None

    def process_video_frames(
        self,
        frames_dir: Path,
        csv_file: Path,
        output_dir: Path,
        frame_format: str = "png",
    ) -> Dict:
        """
        Process all frames from a video using OpenFace landmarks.

        Args:
            frames_dir: Directory containing extracted frames
            csv_file: OpenFace CSV output file
            output_dir: Directory to save extracted faces
            frame_format: Format of frame files (png, jpg, etc.)

        Returns:
            Dictionary with processing results and statistics
        """
        try:
            # Load OpenFace CSV
            df = pd.read_csv(csv_file)
            self.logger.info(f"Loaded OpenFace CSV with {len(df)} rows")

            # Create output directories
            faces_dir = output_dir / "faces"
            landmarks_dir = output_dir / "landmarks"
            faces_dir.mkdir(parents=True, exist_ok=True)
            landmarks_dir.mkdir(parents=True, exist_ok=True)

            successful_extractions = 0
            failed_extractions = 0
            extracted_data = []

            for idx, row in df.iterrows():
                # Construct frame filename from frame number or timestamp
                frame_number = row.get("frame", idx)
                frame_filename = f"frame_{frame_number:04d}.{frame_format}"
                frame_path = frames_dir / frame_filename

                if not frame_path.exists():
                    # Try alternative naming patterns
                    alternative_patterns = [
                        f"frame_{frame_number:05d}.{frame_format}",
                        f"frame_{frame_number:06d}.{frame_format}",
                        f"{frame_number:04d}.{frame_format}",
                        f"{frame_number:05d}.{frame_format}",
                    ]

                    found = False
                    for pattern in alternative_patterns:
                        alt_path = frames_dir / pattern
                        if alt_path.exists():
                            frame_path = alt_path
                            found = True
                            break

                    if not found:
                        self.logger.warning(f"Frame not found: {frame_filename}")
                        failed_extractions += 1
                        continue

                # Process frame
                result = self.process_frame(frame_path, row)
                if result is None:
                    failed_extractions += 1
                    continue

                # Save face crop
                face_filename = f"face_{frame_number:04d}.{frame_format}"
                face_path = faces_dir / face_filename
                cv2.imwrite(str(face_path), result["face_crop"])

                # Save landmarks as JSON
                landmarks_filename = f"landmarks_{frame_number:04d}.json"
                landmarks_path = landmarks_dir / landmarks_filename

                landmarks_data = {
                    "frame_number": int(frame_number),
                    "normalized_landmarks": result["normalized_landmarks"].tolist(),
                    "bbox": result["bbox"],
                    "original_landmarks": result["original_landmarks"].tolist(),
                    "face_path": str(face_path),
                    "original_frame": str(result["frame_path"]),
                }

                with open(landmarks_path, "w") as f:
                    json.dump(landmarks_data, f, indent=2)

                # Add to extracted data list
                extracted_data.append(
                    {
                        "frame_number": int(frame_number),
                        "face_path": str(face_path),
                        "landmarks_path": str(landmarks_path),
                        "bbox": result["bbox"],
                    }
                )

                successful_extractions += 1

            # Save summary metadata
            summary = {
                "video_info": {
                    "frames_dir": str(frames_dir),
                    "csv_file": str(csv_file),
                    "output_dir": str(output_dir),
                },
                "extraction_stats": {
                    "total_frames": len(df),
                    "successful_extractions": successful_extractions,
                    "failed_extractions": failed_extractions,
                    "success_rate": (
                        successful_extractions / len(df) if len(df) > 0 else 0
                    ),
                },
                "extraction_config": {
                    "target_size": self.target_size,
                    "frame_format": frame_format,
                },
                "extracted_faces": extracted_data,
            }

            summary_path = output_dir / "extraction_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            self.logger.info(
                f"Face extraction completed. "
                f"Success: {successful_extractions}, Failed: {failed_extractions}"
            )

            return {
                "success": True,
                "summary": summary,
                "faces_dir": str(faces_dir),
                "landmarks_dir": str(landmarks_dir),
                "summary_file": str(summary_path),
            }

        except Exception as e:
            self.logger.error(f"Error processing video frames: {e}")
            return {"success": False, "error": str(e)}


def extract_faces_from_video_processing(
    frames_dir: Path,
    openface_csv: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (224, 224),
    frame_format: str = "png",
) -> Dict:
    """
    Convenience function to extract faces from a processed video.

    Args:
        frames_dir: Directory containing extracted frames
        openface_csv: OpenFace CSV output file
        output_dir: Directory to save extracted faces
        target_size: Target size for face crops (width, height)
        frame_format: Format of frame files

    Returns:
        Dictionary with processing results
    """
    extractor = FaceExtractor(target_size=target_size)
    return extractor.process_video_frames(
        frames_dir, openface_csv, output_dir, frame_format
    )
