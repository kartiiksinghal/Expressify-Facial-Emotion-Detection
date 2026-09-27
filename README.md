# Real-Time Facial Emotion Detection

A CNN-based system that detects human emotions from facial expressions in real time using a live webcam feed. Built with OpenCV for face detection and a custom Convolutional Neural Network for emotion classification.

## Overview

Emotion detection from facial expressions has wide-ranging applications in human-computer interaction, mental health monitoring, and customer behavior analysis. This project builds a real-time face emotion detection system that classifies expressions into six categories: **Angry, Disgust, Fear, Happy, Sad, and Surprise**.

The system:
- Uses OpenCV's Haar Cascade classifier to detect faces in each video frame
- Feeds each detected face through a trained CNN to classify the expressed emotion
- Displays the prediction live, overlaid on the video feed

## Results

- Achieved **97% accuracy** on the classification task
- Trained on the [JAFFE (Japanese Female Facial Expression) dataset](https://zenodo.org/record/3451524)

## Project Structure

```
├── emotion_detection_training.ipynb   # Data loading, CNN architecture, training & evaluation
├── reall_face_emotion.ipynb           # Real-time webcam inference
├── emotion_detection_cnn_model.h5     # Pre-trained model weights
├── haarcascade_frontalface_default.xml # OpenCV face detector
└── dataset/jaffe/                     # Training images, organized by emotion label
```

## Tech Stack

- **Python**
- **OpenCV** — face detection (Haar Cascade)
- **TensorFlow / Keras** — CNN model
- **scikit-learn** — label encoding, train/test split
- **NumPy, Matplotlib** — data handling & visualization

## Setup

```bash
pip install opencv-python tensorflow scikit-learn numpy matplotlib
```

## Usage

1. **Train the model** (optional — a pre-trained model is already included):
   Open `emotion_detection_training.ipynb` and run all cells. Update the dataset path variables to point to your local `dataset/jaffe` folder.

2. **Run real-time detection**:
   Open `reall_face_emotion.ipynb` and run all cells. This will launch your webcam and display live emotion predictions. Press `q` to quit.

## Model Architecture

A CNN trained on 48×48 grayscale face images, with convolutional + max-pooling layers feeding into a dense classification head over the six emotion classes.

## Notes

- Works best under standard lighting conditions with frontal facial views.
- Dataset paths in the notebooks are set for local use — update them to match your directory structure before running.
