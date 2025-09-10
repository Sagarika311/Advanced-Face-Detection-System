import cv2
import gradio as gr
import numpy as np
import json
import os

def load_models(config_path="config.json"):
    with open(config_path) as f:
        config = json.load(f)

    face_net = cv2.dnn.readNetFromCaffe(config['face_detector_proto'], config['face_detector_model'])
    age_net = cv2.dnn.readNetFromCaffe(config['age_proto'], config['age_model'])
    gender_net = cv2.dnn.readNetFromCaffe(config['gender_proto'], config['gender_model'])

    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
                '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']

    return face_net, age_net, gender_net, age_list, gender_list


face_net, age_net, gender_net, age_list, gender_list = load_models("config.json")

def detect_and_predict(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            face_img = image[y1:y2, x1:x2]

            blob_face = cv2.dnn.blobFromImage(
                face_img, 1, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            gender_net.setInput(blob_face)
            gender = gender_list[gender_net.forward()[0].argmax()]

            age_net.setInput(blob_face)
            age = age_list[age_net.forward()[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            results.append(label)

    return image, results


demo = gr.Interface(
    fn=detect_and_predict,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.JSON()],
    live=False,
    title="Face Detection & Age/Gender Prediction"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
