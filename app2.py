import cv2
import numpy as np
import gradio as gr
import os
import sys
import json

# --------- Resource Path Helper ---------
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller bundle """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# --------- Load Config ---------
with open(resource_path("config.json")) as f:
    config = json.load(f)

# --------- Load Models ---------
face_net = cv2.dnn.readNetFromCaffe(
    resource_path(os.path.join("models", config["face_detector_proto"])),
    resource_path(os.path.join("models", config["face_detector_model"]))
)
age_net = cv2.dnn.readNetFromCaffe(
    resource_path(os.path.join("models", config["age_proto"])),
    resource_path(os.path.join("models", config["age_model"]))
)
gender_net = cv2.dnn.readNetFromCaffe(
    resource_path(os.path.join("models", config["gender_proto"])),
    resource_path(os.path.join("models", config["gender_model"]))
)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# --------- Inference Function ---------
def detect_faces(frame, confidence_threshold=0.5):
    """Take an image (numpy array), detect faces, return annotated image + labels"""
    if frame is None:
        return None, "No frame received"

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face_img = frame[y1:y2, x1:x2].copy()
            if face_img.size == 0:
                continue

            blob_face = cv2.dnn.blobFromImage(face_img, 1, (227, 227),
                                              (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            gender_net.setInput(blob_face)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            age_net.setInput(blob_face)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            results.append(label)

            # Draw box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_image, results if results else ["No faces detected"]

# --------- Gradio UI ---------
with gr.Blocks() as demo:
    gr.Markdown("# 👤 Face Detection + Age & Gender Estimation")
    gr.Markdown("Use your **webcam** or upload an image to test the model.")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["webcam", "upload"], type="numpy", label="Input Image")
            confidence_slider = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence Threshold")
            btn = gr.Button("Detect Faces")
        with gr.Column():
            output_img = gr.Image(type="numpy", label="Processed Image")
            output_labels = gr.Label(label="Predictions")

    btn.click(fn=detect_faces,
              inputs=[input_img, confidence_slider],
              outputs=[output_img, output_labels])

# --------- Launch ---------
if __name__ == "__main__":
    demo.launch()
