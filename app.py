import os, time, json, cv2, numpy as np, threading
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
def resource_path(relative_path):
    return os.path.join(ROOT, relative_path)

# Load config
with open(resource_path("config.json"), "r") as f:
    CONFIG = json.load(f)

CAPTURE_DIR = resource_path(CONFIG.get("captured_faces_dir", "captured_faces"))
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Load models
def load_models(cfg):
    face_net = cv2.dnn.readNetFromCaffe(resource_path(cfg["face_detector_proto"]),
                                        resource_path(cfg["face_detector_model"]))
    age_net = cv2.dnn.readNetFromCaffe(resource_path(cfg["age_proto"]),
                                       resource_path(cfg["age_model"]))
    gender_net = cv2.dnn.readNetFromCaffe(resource_path(cfg["gender_proto"]),
                                          resource_path(cfg["gender_model"]))
    return face_net, age_net, gender_net

face_net, age_net, gender_net = load_models(CONFIG)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)','(38-43)','(48-53)','(60-100)']
GENDER_LIST = ['Male', 'Female']

# Shared state
state = {
    "face_detection_active": True,
    "confidence_threshold": 0.5,
    "last_faces": [],
    "last_frame": None,
    "fps": 0.0,
    "faces_count": 0
}
state_lock = threading.Lock()

# Video capture with fallback
def make_capture():
    cap = cv2.VideoCapture(CONFIG.get("camera_index", 0))
    if not cap.isOpened():
        fallback = resource_path(CONFIG.get("video_fallback", "sample.mp4"))
        if os.path.exists(fallback):
            cap = cv2.VideoCapture(fallback)
        else:
            raise RuntimeError("No camera and no sample video found.")
    return cap

cap = make_capture()

# Detection function
def detect_faces(frame, conf_thresh):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0,0,i,2])
        if confidence > conf_thresh:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            faces.append((max(0,x1), max(0,y1), x2-x1, y2-y1))
    return faces

def estimate_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1, (227,227),
                                 (78.4263377603,87.7689143744,114.895847746), swapRB=False)
    gender_net.setInput(blob)
    gender = GENDER_LIST[int(np.argmax(gender_net.forward()[0]))]
    age_net.setInput(blob)
    age = AGE_LIST[int(np.argmax(age_net.forward()[0]))]
    return gender, age

# Background capture thread
def capture_worker():
    frame_times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video fallback
            time.sleep(0.05)
            continue

        start = time.time()
        faces = []
        with state_lock:
            active = state["face_detection_active"]
            conf = state["confidence_threshold"]

        if active:
            try: faces = detect_faces(frame, conf)
            except: faces = []
            for (x,y,w,h) in faces:
                face_img = frame[y:y+h, x:x+w]
                try:
                    gender, age = estimate_age_gender(face_img)
                    label = f"{gender}, {age}"
                except:
                    label = ""
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                if label: cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        end = time.time()
        frame_times.append(end-start)
        if len(frame_times)>30: frame_times.pop(0)
        fps_val = 1.0/(sum(frame_times)/len(frame_times)) if frame_times else 0

        with state_lock:
            state["last_faces"]=faces
            state["last_frame"]=frame.copy()
            state["fps"]=fps_val
            state["faces_count"]=len(faces)

threading.Thread(target=capture_worker, daemon=True).start()

# Flask app
app = Flask(__name__, static_folder="captured_faces", template_folder="templates")

@app.route("/")
def index(): return render_template("index.html")

def gen_mjpeg():
    boundary = b"--frame\r\n"
    while True:
        with state_lock: frame = state["last_frame"]
        if frame is None: time.sleep(0.05); continue
        ret,jpeg = cv2.imencode(".jpg", frame)
        if not ret: continue
        yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"

@app.route("/video_feed")
def video_feed(): return Response(gen_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    with state_lock: state["face_detection_active"] = not state["face_detection_active"]
    return jsonify({"face_detection_active": state["face_detection_active"]})

@app.route("/set_confidence", methods=["POST"])
def set_confidence():
    data = request.get_json() or {}
    val = float(data.get("value", state["confidence_threshold"]))
    val = max(0.01, min(0.99, val))
    with state_lock: state["confidence_threshold"]=val
    return jsonify({"confidence_threshold": state["confidence_threshold"]})

@app.route("/capture_faces", methods=["POST"])
def capture_faces():
    saved = 0
    ts_base = int(time.time()*1000)
    with state_lock:
        faces = list(state["last_faces"])
        frame = state["last_frame"].copy() if state["last_frame"] is not None else None
    if frame is None or not faces:
        return jsonify({"saved":0,"message":"No faces to capture"})
    for i,(x,y,w,h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]
        fname = f"face_{ts_base+i}.jpg"
        fpath = os.path.join(CAPTURE_DIR, fname)
        try: cv2.imwrite(fpath, face_img); saved+=1
        except: pass
    return jsonify({"saved":saved})

@app.route("/clear_captures", methods=["POST"])
def clear_captures():
    removed = 0
    for fn in os.listdir(CAPTURE_DIR):
        p = os.path.join(CAPTURE_DIR, fn)
        if os.path.isfile(p):
            try: os.remove(p); removed+=1
            except: pass
    return jsonify({"removed":removed})

@app.route("/stats")
def stats():
    with state_lock:
        return jsonify({"fps":round(state["fps"],2),
                        "faces":state["faces_count"],
                        "face_detection_active":state["face_detection_active"],
                        "confidence_threshold":state["confidence_threshold"]})

@app.route("/thumbnails")
def thumbnails():
    files=[]
    for fn in sorted(os.listdir(CAPTURE_DIR), reverse=True):
        if fn.lower().endswith((".png",".jpg",".jpeg")):
            files.append({"name":fn,"url":f"/captured/{fn}"})
    return jsonify(files)

@app.route("/captured/<path:filename>")
def captured_file(filename):
    return send_from_directory(CAPTURE_DIR, filename)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
