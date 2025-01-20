# Video Face Detection and Recognition

This project processes a collection of videos to detect and identify faces across multiple videos. It generates an HTML report showing each unique person detected, along with their appearances across different videos with timestamps.

## Features

- Detects faces in video files (supports .mp4, .avi, .mov)
- Identifies the same person across different videos
- Extracts high-quality face images with extended bounds
- Generates an HTML report with:
  - Clear face images for each person
  - List of appearances (video name and timestamp)
- Progress tracking with progress bars
- Detailed logging
- Debug mode for testing
- Results organized in timestamped folders

## Requirements

```bash
pip install face_recognition opencv-python jinja2 tqdm
```

## Usage

1. Update the `video_folder` path in `face_detection.py`
2. Run the script:
```bash
python face_detection.py
```

The script will create a new folder named `face_detection_results_YYYYMMDD_HHMMSS` containing:
- `detected_faces/` - Directory with extracted face images
- `face_detection.log` - Processing log file
- `face_detection_report.html` - Final report with all detections

## Development Process

This project was developed through an iterative chat process with Cursor. Below are the key interactions. Often, some errors occured, or it wasn't exctly what I wanted. In these cases, I asked for improvements or made some manual changes.

### Initial Request
> I want to do face recognition in a series of videos. Enumerate all videos in a specified folder. I want to detect and identify people, and give a list of all the people detected. The same people in different videos should be correctly identified as the same people. As a result, I want to create an HTML page with the list of people, and per person a clear image of their face. Per person also list the names of the videos and the timestamps where they are detected.

### Improvements Added
1. Added progress bars and logging:
> add debug output and progress bars

2. Enhanced face image quality:
> The faces that are extracted in the image are too small. can you extend the bounding box of the faces to make it 3 times bigger?

3. Organized output in timestamped folders:
> Create the report and the detected faces all in a separate folder, that is named with the current timestamp

4. Added parallelization:
> Add parallelization

5. Improved face quality:
> The quality of the face recognition is not well enough. People are seen as same people when they are not and the other way around


### Dependency problems
To install the dependencies, I tried to use:
```
pip install face-recognition opencv-python jinja2
```

However, this resulted in some errors compiling ```dlib```, which I fixed by installing the dependencies in the following way:

1. First re-install Xcode Command Line Tools:
```
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
```

2. Then install cmake, pkg-config and dlib:
```
brew install cmake pkg-config dlib
```

3. Then set the following environment variables:
```
export CFLAGS="-I$(xcrun --show-sdk-path)/usr/include"
export CPPFLAGS="-I$(xcrun --show-sdk-path)/usr/include"
export LDFLAGS="-L$(xcrun --show-sdk-path)/usr/lib"
```

4. Then install the dependencies:
```
pip install face-recognition opencv-python jinja2
```


## Implementation Details

- Uses `face_recognition` library for face detection and encoding
- Processes every 30th frame for performance
- Extracts faces with 3x larger bounding box for better quality
- Compares face encodings to identify the same person across videos
- Uses Jinja2 templating for HTML report generation
- Includes debug mode for testing with limited video processing
- It does parallel processing to use all CPU cores.

## Notes

- Face recognition accuracy depends on video quality, lighting, and face angles
- Processing speed depends on video resolution and frame sampling rate
- Debug mode can be enabled by setting `debug_mode = True` for testing, in this case it only processes the first minute of max 3 videos.

