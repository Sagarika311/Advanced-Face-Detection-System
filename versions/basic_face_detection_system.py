import cv2
import numpy as np
import time

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
fps_start_time = 0
fps = 0

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def draw_button(frame, text, position, size):
    cv2.rectangle(frame, position, (position[0] + size[0], position[1] + size[1]), (0, 255, 0), -1)
    cv2.putText(frame, text, (position[0] + 5, position[1] + size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def main():
    global fps_start_time, fps
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detect_faces(frame)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calculate and display FPS
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        if time_diff > 0:
            fps = 1 / time_diff
        fps_start_time = fps_end_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display face count
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw exit button
        draw_button(frame, "Exit", (10, frame.shape[0] - 40), (60, 30))

        # Display the resulting frame
        cv2.imshow('Face Detection System', frame)

        # Check for mouse clicks on the exit button
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if 10 <= x <= 70 and frame.shape[0] - 40 <= y <= frame.shape[0] - 10:
                    cv2.destroyAllWindows()
                    cap.release()
                    exit()

        cv2.setMouseCallback('Face Detection System', on_mouse)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()