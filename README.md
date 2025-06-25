# Fall Detection App

A real-time fall detection system using pose estimation and deep learning, built with Streamlit, OpenPifPaf, and PyTorch. See online demo at [https://fall-detection.streamlit.app](https://fall-detection.streamlit.app).

## Overview

Every year, approximately 37 million falls occur worldwide, often necessitating immediate medical attention. For elderly individuals living alone, falls can be particularly dangerous, as they may be unable to call for help.

This project provides a real-time fall detection system that can be deployed with easily available hardware. The app uses OpenPifPaf for pose estimation and a pre-trained LSTM model (PyTorch) to classify falls in video streams.

## Features

- **Real-time fall detection** from device camera, video files, or image sequences
- **Pose estimation** using OpenPifPaf
- **LSTM-based classification** into five classes: `None`, `Normal`, `Falling`, `Warning`, `Fallen`
- **Streamlit web interface** for easy interaction
- **Demo videos** and dataset references

## How It Works

1. **Pose Estimation:** OpenPifPaf estimates human poses in each video frame.
2. **Feature Extraction:** Five features are extracted per frame:
    - Aspect ratio of the bounding box
    - Logarithm of the angle between vertical axis and torso vector
    - Rotation energy between frames
    - Derivative of the aspect ratio
    - Generalized force from joint angles and velocities
3. **Classification:** Features are fed into an LSTM model to classify the activity.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/supungamlath/fall-detection.git
    cd fall-detection
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app 
```sh
streamlit run main.py
```

### Modes

- **Demo Video:** Watch a pre-recorded fall detection demo.
- **Device Camera:** Run fall detection in real-time using your webcam.
- **Image Sequence:** Upload a zip file of images to process as a video.
- **Video File:** Upload a video file for fall detection.

## Datasets

- [UR Fall Detection Dataset](http://fenix.ur.edu.pl/mkepski/ds/uf.html)
- [FALL-UP Dataset](https://sites.google.com/up.edu.mx/har-up)

## References

1. Mohammad Taufeeque, Samad Koita, Nicolai Spicher, and Thomas M. Deserno "Multi-camera, multi-person, and real-time fall detection using long short term memory", Proc. SPIE 11601, Medical Imaging 2021: Imaging Informatics for Healthcare, Research, and Applications, 1160109 (15 February 2021); https://doi.org/10.1117/12.2580700
2. Bogdan Kwolek, Michal Kepski, Human fall detection on embedded platform using depth maps and wireless accelerometer, Computer Methods and Programs in Biomedicine, Volume 117, Issue 3, December 2014, Pages 489-501, ISSN 0169-2607; http://fenix.ur.edu.pl/mkepski/ds/uf.html
3. S. Kreiss, L. Bertoni, and A. Alahi, OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association. 2021; https://doi.org/10.48550/arXiv.2103.02440

## Author

Created by [Supun Gamlath](https://github.com/supungamlath)