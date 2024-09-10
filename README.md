# Advanced Face Detection System

## Overview
This project implements a real-time face detection system using OpenCV, age, and gender classification models. It also * includes additional functionality such as eye detection toggling, face image capture, and real-time FPS display. *

## The key features of this project include:

Face detection using Haar cascades.
Age and gender prediction using pre-trained deep learning models.
Toggle to enable or disable eye detection.
Capturing and saving face images locally.
User interface with buttons for various functionalities.
Real-time FPS display.

## Requirements
Before running the project, ensure you have the following prerequisites:

Python 3.x
OpenCV
NumPy

## Models:
Haarcascade Frontal Face and Eye detection models, which comes pre-installed with OpenCV.
Pre-trained Caffe models for age and gender classification.

## Project Functionality
### 1. Face Detection
The system detects faces using OpenCV's pre-trained haarcascade_frontalface_default.xml model. A green rectangle is drawn around the detected faces.

### 2. Eye Detection (Toggle)
Users can toggle the eye detection feature using the "Toggle Eyes" button. When enabled, blue rectangles will appear around the detected eyes within the face region.

### 3. Age and Gender Prediction
The system estimates the gender and age group of the detected faces using pre-trained Caffe models. The predictions are displayed on the frame above the detected face.

### 4. Capture Face
By clicking the "Capture Face" button, the first detected face is saved as an image in the captured_faces/ folder with a timestamp as the filename.

### 5. FPS Display
The frame per second (FPS) rate is calculated and displayed in the top-left corner of the window, providing information about the performance of the system.

## Known Issues
Camera Not Detected: Ensure your webcam is properly connected. If you're using an external webcam, try specifying the correct camera index in cap = cv2.VideoCapture(0) (use 1 for an external webcam).
Low FPS: On some machines, processing might be slower depending on the CPU/GPU. Consider optimizing by reducing image resolution or processing fewer frames per second.

## Future Enhancements
Add functionality for multi-face detection and capture.
Optimize model inference for better performance on low-end devices.
Add emotion detection and facial landmarks tracking.

## License
This project is open-source and free to use. No license is provided.
