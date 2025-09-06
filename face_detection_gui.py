import sys
import cv2
import numpy as np
import time
import os
import logging
import json
import collections
from PyQt5 import QtWidgets, QtGui, QtCore

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller bundle """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class FaceDetectionSystemGUI(QtWidgets.QWidget):
    def __init__(self, config_path):
        super().__init__()
        # Load config
        with open(resource_path(config_path)) as f:
            self.config = json.load(f)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('FaceDetectionSystemGUI')

        # Create captured faces directory (ensure it’s writable outside exe)
        self.captured_faces_dir = self.config.get('captured_faces_dir', 'captured_faces')
        os.makedirs(self.captured_faces_dir, exist_ok=True)

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.config.get('camera_index', 0))
        if not self.cap.isOpened():
            self.logger.error("Cannot open camera")
            raise IOError("Cannot open camera")

        # Initialize detection confidence threshold
        self.confidence_threshold = 0.5

        # Load models with progress bar
        self.progress_dialog = QtWidgets.QProgressDialog("Loading models...", None, 0, 0, self)
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()
        QtWidgets.QApplication.processEvents()
        self._load_models()
        self.progress_dialog.close()

        # Setup UI
        self.init_ui()

        # Timer for video update (~30 FPS)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # State variables
        self.face_detection_active = True
        self.fps_deque = collections.deque(maxlen=30)
        self.last_faces = []
        self.last_frame = None

        # Store captured face thumbnails (QPixmap)
        self.captured_thumbnails = []

    def _load_models(self):
        try:
            self.face_net = cv2.dnn.readNetFromCaffe(
                resource_path(self.config['face_detector_proto']),
                resource_path(self.config['face_detector_model'])
            )
            self.age_net = cv2.dnn.readNetFromCaffe(
                resource_path(self.config['age_proto']),
                resource_path(self.config['age_model'])
            )
            self.gender_net = cv2.dnn.readNetFromCaffe(
                resource_path(self.config['gender_proto']),
                resource_path(self.config['gender_model'])
            )
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
                         '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']

    def init_ui(self):
        self.setWindowTitle("Face Detection System")

        # Video display label
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")

        # Buttons
        self.btn_toggle_detection = QtWidgets.QPushButton("Toggle Detection", self)
        self.btn_toggle_detection.clicked.connect(self.toggle_detection)

        self.btn_capture_face = QtWidgets.QPushButton("Capture Faces", self)
        self.btn_capture_face.clicked.connect(self.capture_faces)

        self.btn_clear_faces = QtWidgets.QPushButton("Clear Captured Faces", self)
        self.btn_clear_faces.clicked.connect(self.clear_captured_faces)

        self.btn_exit = QtWidgets.QPushButton("Exit", self)
        self.btn_exit.clicked.connect(self.close)

        # Confidence threshold slider
        self.confidence_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(int(self.confidence_threshold * 100))
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)

        self.confidence_label = QtWidgets.QLabel(
            f"Confidence Threshold: {self.confidence_threshold:.2f}", self
        )

        # Info labels
        self.label_fps = QtWidgets.QLabel("FPS: 0", self)
        self.label_faces = QtWidgets.QLabel("Faces: 0", self)

        # Status bar
        self.status_bar = QtWidgets.QStatusBar(self)

        # Captured faces gallery (scroll area)
        self.gallery_widget = QtWidgets.QWidget()
        self.gallery_layout = QtWidgets.QHBoxLayout()
        self.gallery_widget.setLayout(self.gallery_layout)

        self.gallery_scroll = QtWidgets.QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setWidget(self.gallery_widget)
        self.gallery_scroll.setFixedHeight(120)

        # Layout
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.video_label)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(self.btn_toggle_detection)
        controls_layout.addWidget(self.btn_capture_face)
        controls_layout.addWidget(self.btn_clear_faces)
        controls_layout.addWidget(self.btn_exit)
        vbox.addLayout(controls_layout)

        confidence_layout = QtWidgets.QHBoxLayout()
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        vbox.addLayout(confidence_layout)

        info_layout = QtWidgets.QHBoxLayout()
        info_layout.addWidget(self.label_fps)
        info_layout.addWidget(self.label_faces)
        vbox.addLayout(info_layout)

        vbox.addWidget(self.gallery_scroll)
        vbox.addWidget(self.status_bar)

        self.setLayout(vbox)

    def update_confidence_threshold(self, value):
        self.confidence_threshold = value / 100.0
        self.confidence_label.setText(f"Confidence Threshold: {self.confidence_threshold:.2f}")
        self.status_bar.showMessage(
            f"Confidence threshold set to {self.confidence_threshold:.2f}", 3000
        )

    def toggle_detection(self):
        self.face_detection_active = not self.face_detection_active
        status = "enabled" if self.face_detection_active else "disabled"
        self.status_bar.showMessage(f"Face detection {status}", 3000)

    def update_frame(self):
        start_time = time.time()
        ret, frame = self.cap.read()
        if not ret:
            self.status_bar.showMessage("Failed to grab frame", 3000)
            return

        faces = []
        if self.face_detection_active:
            faces = self.detect_faces(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w].copy()
                gender, age = self.estimate_age_gender(face_img)
                label = f"{gender}, {age}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)

        # Calculate FPS
        end_time = time.time()
        self.fps_deque.append(end_time - start_time)
        avg_fps = 1 / (sum(self.fps_deque) / len(self.fps_deque)) if self.fps_deque else 0

        # Update info labels
        self.label_fps.setText(f"FPS: {avg_fps:.2f}")
        self.label_faces.setText(f"Faces: {len(faces)}")

        # Convert frame to QImage and display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line,
                                QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

        # Store last faces and frame for capture
        self.last_faces = faces
        self.last_frame = frame

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def estimate_age_gender(self, face_img):
        blob = cv2.dnn.blobFromImage(
            face_img, 1, (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746), swapRB=False
        )
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]

        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]

        return gender, age

    def capture_faces(self):
        if self.last_faces and self.last_frame is not None:
            count = 0
            for face in self.last_faces:
                x, y, w, h = face
                face_img = self.last_frame[y:y+h, x:x+w]
                timestamp = int(time.time() * 1000) + count
                filename = os.path.join(self.captured_faces_dir,
                                        f'face_{timestamp}.jpg')
                try:
                    cv2.imwrite(filename, face_img)
                    self.logger.info(f"Face captured and saved as {filename}")
                    self.add_thumbnail(filename)
                    count += 1
                except Exception as e:
                    self.logger.error(f"Failed to save face image: {e}")
            self.status_bar.showMessage(f"Captured {count} face(s)", 3000)
        else:
            self.status_bar.showMessage("No face detected to capture", 3000)

    def add_thumbnail(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        if pixmap.isNull():
            self.logger.warning(f"Failed to load thumbnail from {image_path}")
            return
        thumbnail = pixmap.scaled(100, 100, QtCore.Qt.KeepAspectRatio,
                                  QtCore.Qt.SmoothTransformation)
        label = QtWidgets.QLabel()
        label.setPixmap(thumbnail)
        label.setToolTip(image_path)
        self.gallery_layout.addWidget(label)
        self.captured_thumbnails.append(label)

    def clear_captured_faces(self):
        # Remove thumbnails from layout
        for label in self.captured_thumbnails:
            self.gallery_layout.removeWidget(label)
            label.deleteLater()
        self.captured_thumbnails.clear()

        # Delete files from directory
        try:
            for filename in os.listdir(self.captured_faces_dir):
                file_path = os.path.join(self.captured_faces_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            self.status_bar.showMessage("Cleared all captured faces", 3000)
        except Exception as e:
            self.logger.error(f"Failed to clear captured faces: {e}")
            self.status_bar.showMessage("Error clearing captured faces", 3000)

    def keyPressEvent(self, event):
        # Keyboard shortcuts
        if event.key() == QtCore.Qt.Key_Q:
            self.close()
        elif event.key() == QtCore.Qt.Key_C:
            self.capture_faces()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FaceDetectionSystemGUI('config.json')
    window.show()
    sys.exit(app.exec_())