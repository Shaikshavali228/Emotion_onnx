import torch
import cv2
import numpy as np
from flask import Flask, Response, render_template
from PIL import Image
from util import *

# Initialize Flask app
app = Flask(__name__)

# Check for CUDA availability
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

# Load models safely
try:
    detector = FaceDetector('weights\detection.onnx')
except Exception as e:
    print(f"Error loading FaceDetector model: {e}")
    exit()

try:
    fer = HSEmotionRecognizer('weights\emotion.onnx')
except Exception as e:
    print(f"Error loading Emotion Recognition model: {e}")
    exit()

def detect_face(frame):
    """Detect faces in the frame"""
    boxes = detector.detect(frame, (640, 640))
    return boxes if boxes is not None and len(boxes) else None

def generate_frames():
    """Capture video frames and process them for real-time emotion recognition"""
    stream = cv2.VideoCapture(0)
    if not stream.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = stream.read()
        if not ret:
            print("Camera index error")
            break

        # Convert frame to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes = detect_face(image)

        if boxes is not None:
            for box in boxes.astype('int32'):
                x1, y1, x2, y2 = box[:4]

                # Clamp bounding box values
                h, w, _ = image.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                # Extract and preprocess face
                face_image = image[y1:y2, x1:x2]
                if face_image.size == 0:
                    continue  # Skip empty face regions

                pil_image = Image.fromarray(face_image).convert('RGB')
                pil_image = pil_image.resize((224, 224))  # Resize for model compatibility

                # Predict emotion
                try:
                    emotion, scores = fer.predict_emotions(np.array(pil_image), logits=False)
                    draw_emotion_bars(fer, frame, scores, (x2 + 10, y1), bar_height=15, width=100)
                    cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Emotion prediction error: {e}")

        # Encode frame for HTTP streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Home page for displaying the video stream"""
    return """
    <html>
    <head>
        <title>Emotion Recognition Stream</title>
    </head>
    <body>
        <h1>Real-Time Emotion Recognition</h1>
        <img src="/video_feed" width="800">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Route to stream video frames"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.1.1', port=5000, debug=False)
