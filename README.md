# 🎭 Face Detection System (PyQt5 + OpenCV)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-green?logo=opencv)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-orange?logo=qt)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A desktop GUI application that detects faces in **real-time** using OpenCV’s DNN face detector.  
The system also performs **age and gender prediction**, allows you to **capture faces**, and manage them inside a **gallery view**.  

---

## ✨ Features
- 📸 Real-time face detection with bounding boxes  
- 🧑‍🤝‍🧑 Age & gender prediction using pre-trained models  
- ⚡ Adjustable confidence threshold  
- 💾 Capture faces into a local gallery  
- 🎮 Keyboard shortcuts:  
  - `C` → Capture face  
  - `Q` → Quit application  

---

## 🖼️ Demo

![Demo Screenshot](Demo.png)  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/FaceDetectionSystem.git
cd FaceDetectionSystem
pip install -r requirements.txt
````

Run the app:

```bash
python face_detection_gui.py
```

---

## 📦 Build as Executable (Windows)

To package into a `.exe` using **PyInstaller**:

```bash
pyinstaller --noconfirm --onefile --windowed ^
  --add-data "config.json;." ^
  --add-data "models;models" ^
  face_detection_gui.py
```

The built executable will be inside the `dist/` folder.

---

## 📂 Project Structure

```
FaceDetectionSystem/
│── face_detection_gui.py      # Main GUI application
│── config.json                # Config file (paths, settings)
│── models/                    # Pre-trained Caffe models
│── requirements.txt           # Dependencies
│── README.md                  # Documentation
│── .gitignore                 # Ignore build/venv/output
│── captured_faces/            # Saved face captures (auto-created)
│── dist/ / build/             # PyInstaller outputs (ignored)
```

---

## 👩‍💻 Tech Stack

* **Python 3.11**
* **OpenCV DNN**
* **PyQt5**
* **PyInstaller**

---

## 🛠️ Future Improvements

* Train on custom datasets
* Add face recognition (match against known people)
* Export captured face metadata

---

## 📜 License

This project is licensed under the **MIT License**.
Feel free to use and modify it for learning or personal projects.

---

## 🙌 Author

Made by **Sagarika**
