# Tech Challenge 4

## Usage

```bash
pyenv install 3.12.12
pyenv local 3.12.12
python -m venv .venv
source .venv/bin/activate
```

## Images

Store images in folders named after the person's name (e.g., `murilo/`, `john/`). Each folder should contain images of that person.

## Video Processing

The system processes video files called `input.mp4` in the `videos/` directory and overlays face recognition and emotion detection results. The output video is saved in the `output/` directory with a timestamped filename.

## Description

This project is a face recognition system using OpenCV, face_recognition, and DeepFace libraries for both face detection and emotion analysis.
