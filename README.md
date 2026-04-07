# Sign Language to Text and Speech Conversion

## ⚠️ Deployment Status (Important)
This project is **not deployed online** due to the following reasons:

- It requires **real-time webcam access using OpenCV**
- It depends on a **Python backend (TensorFlow, Keras, MediaPipe)**
- Static hosting platforms like **Netlify / GitHub Pages do not support Python execution or webcam-based processing**

👉 To deploy this project, it needs to be converted into:
- Flask / FastAPI (backend)
- Or Streamlit (simpler option)

Then it can be hosted on platforms like **Render, Railway, or AWS**

---

## 📌 Abstract
Sign language is one of the oldest and most natural forms of communication. This project presents a **real-time hand gesture recognition system** using computer vision and deep learning.

We use a **Convolutional Neural Network (CNN)** to recognize American Sign Language (ASL) finger-spelling gestures captured via webcam. The system processes hand images, classifies gestures, converts them into text, and finally into speech.

---

## 📖 Introduction
Communication is the exchange of thoughts and messages through speech, signals, and visuals. Deaf and mute individuals rely on **sign language**, a visual and gesture-based form of communication.

This project aims to bridge the communication gap by enabling computers to understand hand gestures and translate them into readable text and audible speech.

---

## 🎯 Objective
- Recognize ASL hand gestures using a webcam  
- Convert gestures into text  
- Convert text into speech using TTS (Text-to-Speech)  

---

## 🌍 Scope
This system is beneficial for:
- Deaf and mute individuals  
- People who do not understand sign language  

It enables real-time communication by translating gestures into both **text and audio output**.

---

## ⚙️ Technologies Used
- **Python**
- **OpenCV** – Image processing  
- **MediaPipe** – Hand tracking and landmark detection  
- **TensorFlow / Keras** – Deep learning (CNN model)  
- **NumPy** – Numerical operations  
- **pyttsx3** – Text-to-speech  

---

## 🧠 System Modules

### 1. Data Acquisition
- Uses webcam for real-time input  
- MediaPipe detects hand landmarks  

### 2. Preprocessing & Feature Extraction
- ROI extraction  
- Grayscale conversion  
- Gaussian blur  
- Thresholding  
- Landmark-based representation (background independent)  

### 3. Gesture Classification (CNN)
- Deep learning model trained on gesture data  
- Accuracy:
  - ~97% (normal conditions)  
  - ~99% (ideal conditions)  

### 4. Text-to-Speech
- Converts predicted text into speech using **pyttsx3**

---

## 📊 Model Details
- 26 alphabets grouped into 8 classes for better accuracy  
- Landmark-based approach reduces dependency on lighting and background  

---

## 💻 Requirements

### Hardware
- Webcam  

### Software
- OS: Windows 8+  
- Python: 3.9+  
- IDE: PyCharm / VS Code  

### Libraries
```
opencv-python
numpy
tensorflow
keras
mediapipe
pyttsx3
```

---

## ▶️ How to Run

1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main file:
   ```bash
   python main.py
   ```
4. Allow webcam access  
5. Start showing hand gestures  

---

## ⚠️ Limitations
- Requires proper hand visibility  
- Performance may drop in poor lighting  
- Cannot run directly in browser without backend  

---

## 🚀 Future Improvements
- Web deployment using Flask / Streamlit  
- Mobile application version  
- Support for full sentence recognition  

---

## 📌 Conclusion
This project demonstrates how **AI + Computer Vision** can solve real-world communication problems by translating sign language into text and speech in real time.
