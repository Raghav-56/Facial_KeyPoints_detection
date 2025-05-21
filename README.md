# Facial_KeyPoints_detection
Detect key frames from a video and extract key points from those key frames.

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

## Extract key frames from a video

use ffmpeg to extract key frames from a video

## Extract key points from the key frames

use mediapipe to extract key points from the key frames

# Process a single video
python main.py -i path/to/video.mp4 -o path/to/output

# Process all videos in a directory
python main.py -i path/to/videos -o path/to/output

## To-DO: Add Openface to the repo folder
