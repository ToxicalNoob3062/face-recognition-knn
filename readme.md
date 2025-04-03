# Face Recognition Project

## Introduction

This project demonstrates a simple face recognition system using OpenCV for face detection and scikit-learn's K-Nearest Neighbors (KNN) classifier for face recognition. The system captures face data from a video stream, processes it, and then uses the trained classifier to recognize faces in real-time. This can be useful for applications such as attendance tracking in online video conferences or other related use cases.

## Files and Working

- `video-read.py`: Captures face data from a video stream and saves it as a numpy array file. Run this script, enter the name of the person when prompted, and it will capture and save face data every 10 frames.
- `face-rec.py`: Trains a KNN classifier on the saved face data and uses it to recognize faces in real-time from a video stream. Run this script to see the recognized names displayed on the video stream.

## Installation

1. Clone the repository and navigate to the directory:
    ```sh
    git clone https://github.com/ToxicalNoob3062/face-recognition-knn.git
    cd face-recognition-knn
    ```

2. Install the required packages using the provided `requirements.txt` file:
    ```sh
    pip install -r requirements.txt
    ```

## Changing Camera Setup

Ensure that the video stream URL is correctly set in both scripts. The current URL is set to `http://192.168.86.30:4747/video?1920x1080`. Adjust it according to your camera setup. The Haar Cascade file `haarcascade_frontalface_default.xml` should be in the same directory as the scripts. Make sure the `./data/` directory exists for saving face data.
