import cv2
import numpy as np
import time
import os

# Load the pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load age and gender models
age_net = cv2.dnn.readNetFromCaffe('C:/Users/hp/Desktop/PROJECTS/SEM 7 FaceDetectionSystem/deploy_age.prototxt', 
                                   'C:/Users/hp/Desktop/PROJECTS/SEM 7 FaceDetectionSystem/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('C:/Users/hp/Desktop/PROJECTS/SEM 7 FaceDetectionSystem/deploy_gender.prototxt', 
                                      'C:/Users/hp/Desktop/PROJECTS/SEM 7 FaceDetectionSystem/gender_net.caffemodel')

# Define age and gender lists
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
fps_start_time = 0
fps = 0

# Create directory for saving captured faces
captured_faces_dir = r'C:\Users\hp\Desktop\PROJECTS\SEM 7 FaceDetectionSystem\captured_faces'
if not os.path.exists(captured_faces_dir):
    os.makedirs(captured_faces_dir)

def detect_and_draw_faces(frame, show_eyes):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Detect eyes only if the toggle is enabled
        if show_eyes:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        
        # Estimate age and gender
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        
        # Put age and gender info on the frame
        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    return faces

def draw_button(frame, text, position, size):
    cv2.rectangle(frame, position, (position[0] + size[0], position[1] + size[1]), (0, 255, 0), -1)
    cv2.putText(frame, text, (position[0] + 5, position[1] + size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def capture_face(frame, faces):
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w]
        timestamp = int(time.time())
        filename = os.path.join(captured_faces_dir, f'face_{timestamp}.jpg')
        cv2.imwrite(filename, face_img)
        print(f"Face captured and saved as {filename}")

def main():
    global fps_start_time, fps
    
    face_detection_active = True
    show_eyes = False
    capture_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and draw faces
        if face_detection_active:
            faces = detect_and_draw_faces(frame, show_eyes)
        else:
            faces = []

        # Calculate and display FPS
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        if time_diff > 0:
            fps = 1 / time_diff
        fps_start_time = fps_end_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display face count
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw buttons
        draw_button(frame, "Exit", (10, frame.shape[0] - 40), (60, 30))
        draw_button(frame, "Toggle Detection", (80, frame.shape[0] - 40), (140, 30))
        draw_button(frame, "Toggle Eyes", (230, frame.shape[0] - 40), (100, 30))
        draw_button(frame, "Capture Face", (340, frame.shape[0] - 40), (120, 30))

        # Display the resulting frame
        cv2.imshow('Advanced Face Detection System', frame)

        # Check for mouse clicks on the buttons
        def on_mouse(event, x, y, flags, param):
            nonlocal face_detection_active, show_eyes, capture_mode
            if event == cv2.EVENT_LBUTTONDOWN:
                if 10 <= x <= 70 and frame.shape[0] - 40 <= y <= frame.shape[0] - 10:
                    cv2.destroyAllWindows()
                    cap.release()
                    exit()
                elif 80 <= x <= 220 and frame.shape[0] - 40 <= y <= frame.shape[0] - 10:
                    face_detection_active = not face_detection_active
                elif 230 <= x <= 330 and frame.shape[0] - 40 <= y <= frame.shape[0] - 10:
                    show_eyes = not show_eyes
                elif 340 <= x <= 460 and frame.shape[0] - 40 <= y <= frame.shape[0] - 10:
                    capture_mode = True

        cv2.setMouseCallback('Advanced Face Detection System', on_mouse)

        # Capture face if in capture mode
        if capture_mode and len(faces) > 0:
            capture_face(frame, faces)
            capture_mode = False

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
