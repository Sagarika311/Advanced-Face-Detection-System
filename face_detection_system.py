import cv2
import os
import time
import logging
import json
import numpy as np
import collections
import stat
import urllib.request

class FaceDetectionSystem:
    def __init__(self, config_path):
        # Load config
        with open(config_path) as f:
            self.config = json.load(f)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FaceDetectionSystem')

        # Ensure captured faces directory exists with secure permissions
        self.captured_faces_dir = self.config['captured_faces_dir']
        os.makedirs(self.captured_faces_dir, exist_ok=True)
        try:
            # Set directory permissions to user-only (Linux/macOS)
            os.chmod(self.captured_faces_dir, stat.S_IRWXU)
        except Exception as e:
            self.logger.warning(f"Could not set directory permissions: {e}")

        # Download models if not present
        self._download_models()

        # Load models with error handling
        try:
            # Use OpenCV DNN face detector for better accuracy and performance
            self.face_net = cv2.dnn.readNetFromCaffe(
                self.config['face_detector_proto'],
                self.config['face_detector_model']
            )
            # Load age and gender models
            self.age_net = cv2.dnn.readNetFromCaffe(
                self.config['age_proto'],
                self.config['age_model']
            )
            self.gender_net = cv2.dnn.readNetFromCaffe(
                self.config['gender_proto'],
                self.config['gender_model']
            )
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.config.get('camera_index', 0))
        if not self.cap.isOpened():
            self.logger.error("Cannot open camera")
            raise IOError("Cannot open camera")

        # Age and gender labels
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']

        # For stable FPS calculation
        self.fps_deque = collections.deque(maxlen=30)

    def _download_models(self):
        # Helper function to download models if missing
        model_urls = {
            'face_detector_proto': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
            'face_detector_model': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
            'age_proto': 'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/deploy_age.prototxt',
            'age_model': 'https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel',
            'gender_proto': 'https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/deploy_gender.prototxt',
            'gender_model': 'https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel'
        }
        for key, url in model_urls.items():
            path = self.config.get(key)
            if path and not os.path.exists(path):
                self.logger.info(f"Downloading {key} model...")
                try:
                    urllib.request.urlretrieve(url, path)
                    self.logger.info(f"Downloaded {key} to {path}")
                except Exception as e:
                    self.logger.error(f"Failed to download {key} from {url}: {e}")
                    raise

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                # Clamp coordinates to frame size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def estimate_age_gender(self, face_img):
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]

        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]

        return gender, age

    def capture_face(self, frame, face):
        x, y, w, h = face
        face_img = frame[y:y+h, x:x+w]
        timestamp = int(time.time())
        filename = os.path.join(self.captured_faces_dir, f'face_{timestamp}.jpg')
        try:
            cv2.imwrite(filename, face_img)
            self.logger.info(f"Face captured and saved as {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save face image: {e}")

    def run(self):
        self.logger.info("Starting Face Detection System. Press 'q' to quit, 'c' to capture face.")
        try:
            while True:
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to grab frame")
                    break

                faces = self.detect_faces(frame)
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w].copy()
                    gender, age = self.estimate_age_gender(face_img)
                    label = f"{gender}, {age}"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Calculate stable FPS
                end_time = time.time()
                self.fps_deque.append(end_time - start_time)
                avg_fps = 1 / (sum(self.fps_deque) / len(self.fps_deque)) if self.fps_deque else 0
                cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Face Detection System', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Quitting Face Detection System.")
                    break
                elif key == ord('c'):
                    if faces:
                        self.capture_face(frame, faces[0])
                    else:
                        self.logger.info("No face detected to capture.")

        except Exception as e:
            self.logger.error(f"Error during run: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if config.json exists in current directory
    config_path = 'config.json'
    if not os.path.exists(config_path):
        print("Config file 'config.json' not found. Please create it with required model paths.")
        exit(1)
    system = FaceDetectionSystem(config_path)
    system.run()
