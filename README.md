# Facial KeyPoints Detection & Face Extraction Pipeline

A comprehensive pipeline for extracting I-frames from videos, detecting facial keypoints with OpenFace, and creating standardized face crops for emotion recognition training.

## Features

- **I-Frame Extraction**: Uses FFmpeg to extract key frames from videos
- **Facial Landmark Detection**: Leverages OpenFace to detect 68 facial landmarks
- **Face Crop Extraction**: Automatically crops faces using landmark guidance
- **Training Data Generation**: Creates standardized face images with normalized keypoint annotations
- **Batch Processing**: Supports processing single videos or entire directories

## Pipeline Overview

1. **FFmpeg Frame Extraction** → Extract I-frames from input videos
2. **OpenFace Processing** → Detect 68 facial landmarks and Action Units
3. **Face Crop Extraction** → Generate 224x224 face crops with normalized landmarks
4. **Training Data Output** → Organized face images and landmark annotations

## Prerequisites

Before running the project, ensure you have FFmpeg and OpenFace installed and configured correctly.

### FFmpeg

- **Windows:** Download the latest static build from [FFmpeg Builds](https://ffmpeg.org/download.html#build-windows). Extract the archive and add the `bin` directory (containing `ffmpeg.exe`) to your system's PATH environment variable.
- **Linux:** You can usually install FFmpeg using your distribution's package manager. For example, on Debian/Ubuntu:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```
- **macOS:** You can install FFmpeg using Homebrew:
  ```bash
  brew install ffmpeg
  ```

Verify the installation by opening a terminal and typing `ffmpeg -version`.

### OpenFace

Follow the official installation instructions for your operating system from the [OpenFace Wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Installation).

Make sure the OpenFace executables (e.g., `FeatureExtraction`) are accessible in your system's PATH or provide the correct path in the project configuration.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Facial_KeyPoints_detection.git
   cd Facial_KeyPoints_detection
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure OpenFace is available in the `OpenFace_2.2.0_win_x64/` directory or update the path in `config/defaults.py`.

## Usage

### Complete Pipeline (Recommended)

Process videos through the entire pipeline (FFmpeg → OpenFace → Face Extraction):

```bash
# Process a single video
python main.py -i path/to/video.mp4 -o path/to/output

# Process all videos in a directory
python main.py -i path/to/videos -o path/to/output
```

### Face Extraction Only

If you already have processed videos with OpenFace output, extract faces directly:

```bash
# Extract faces from a single processed video
python face_extraction_demo.py --video path/to/processed/video --output path/to/face_output

# Extract faces from all processed videos in a directory
python face_extraction_demo.py --input path/to/processed/videos --output path/to/face_output

# Extract faces with custom size (default is 224x224)
python face_extraction_demo.py --video path/to/processed/video --output path/to/face_output --size 256 256
```

### Individual Components

Run pipeline components separately:

```bash
# Extract frames only
python ffmpeg_extr.py

# OpenFace processing only
python openFace_prs.py --input path/to/frames --output path/to/openface_output
```

## Output Structure

After running the complete pipeline, your output directory will contain:

```
output_directory/
├── video_name/
│   ├── frame_0001.png                    # Extracted I-frames
│   ├── frame_0002.png
│   ├── ...
│   ├── openface/
│   │   ├── video_name.csv                # OpenFace landmarks & AU data
│   │   └── video_name_keyframes.csv      # Key frames analysis
│   └── face_extraction/
│       ├── faces/
│       │   ├── face_0001.png             # 224x224 face crops
│       │   ├── face_0002.png
│       │   └── ...
│       ├── landmarks/
│       │   ├── landmarks_0001.json       # Normalized landmark data
│       │   ├── landmarks_0002.json
│       │   └── ...
│       └── extraction_summary.json       # Processing statistics
```

## Training Data Format

The pipeline generates training-ready data:

### Face Images
- **Size**: 224x224 pixels (configurable)
- **Format**: RGB images (PNG/JPG)
- **Content**: Cropped faces with consistent padding around facial landmarks

### Landmark Annotations
- **Format**: JSON files with normalized coordinates
- **Range**: [0, 1] normalized to face bounding box
- **Points**: 68 facial landmarks (OpenFace standard)
- **Structure**:
  ```json
  {
    "frame_number": 1,
    "normalized_landmarks": [[x1, y1], [x2, y2], ...],
    "bbox": [x, y, width, height],
    "face_path": "path/to/face_image.png"
  }
  ```

## Configuration

Key configuration files:

- `config/defaults.py`: Default paths and processing parameters
- `config/config.py`: Command-line argument parsing
- `requirements.txt`: Python dependencies

## Face Extraction Features

- **Automatic Bounding Box Calculation**: Uses facial landmarks to determine optimal face region
- **Consistent Sizing**: All face crops resized to standard dimensions
- **Landmark Normalization**: Coordinates normalized to [0, 1] range relative to face region
- **Metadata Preservation**: Frame numbers, original paths, and extraction statistics
- **Error Handling**: Robust processing with detailed logging and error recovery

## Training Integration

The extracted data is ready for emotion recognition training:

```python
import cv2
import json
import numpy as np

# Load face image
face_image = cv2.imread('path/to/faces/face_0001.png')

# Load corresponding landmarks
with open('path/to/landmarks/landmarks_0001.json', 'r') as f:
    landmark_data = json.load(f)
    landmarks = np.array(landmark_data['normalized_landmarks'])

# Use in your emotion recognition model
model.train(face_image, landmarks, emotion_label)
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is in your PATH or update `DEFAULT_FFMPEG_PATH` in `config/defaults.py`
2. **OpenFace not found**: Update `DEFAULT_OPENFACE_BINARY` path in `config/defaults.py`
3. **No faces extracted**: Check that OpenFace successfully detected landmarks in the CSV output
4. **Frame numbering mismatch**: The pipeline handles multiple frame naming patterns automatically

### Logging

Check the log files for detailed processing information:
- `logs/video_processor.log`: FFmpeg extraction logs
- `logs/face_extractor.log`: Face extraction logs

## Performance Notes

- **Processing Time**: Approximately 1-3 minutes per minute of video (depending on hardware)
- **Memory Usage**: Moderate (processes frames individually)
- **Storage**: Face crops are significantly smaller than original frames
- **Batch Processing**: Efficiently handles multiple videos in parallel

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
