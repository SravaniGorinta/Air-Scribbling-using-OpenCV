from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import os

app = Flask(__name__)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize canvas
canvas = None
last_x, last_y = None, None

# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clear_canvas():
    global canvas
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

clear_canvas()

def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)  # Reduced threshold for better line detection
    angle = 0
    if lines is not None:
        angles = []
        for rho, theta in lines[:5]:  # Consider up to 5 lines for robustness
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        angle = np.mean(angles) if angles else 0
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
    return deskewed

def preprocess_image(image):
    image = deskew_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    thresh = cv2.bitwise_or(otsu, adaptive)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)  # Reduced iterations to avoid over-dilation
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_image = np.zeros_like(dilated)
    for contour in contours:
        if cv2.contourArea(contour) > 30:
            cv2.drawContours(filtered_image, [contour], -1, 255, thickness=cv2.FILLED)
    resized = cv2.resize(filtered_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  # Reduced resize factor
    return resized

def recognize_text(image):
    try:
        processed_image = preprocess_image(image)
        config_base = r'--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!? '
        psm_modes = [6]  # Simplified to single PSM mode for stability
        results = []
        for psm in psm_modes:
            config = f"{config_base} --psm {psm}"
            text = pytesseract.image_to_string(processed_image, config=config).strip()
            if text:
                results.append(text)
        if results:
            return max(results, key=len)
        return "No text recognized. Try writing more clearly or adjusting your hand position."
    except Exception as e:
        print(f"Recognition error: {str(e)}")  # Log the error for debugging
        return f"Recognition failed: {str(e)}"

def generate_frames():
    global canvas, last_x, last_y
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible.")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (canvas.shape[1], canvas.shape[0]))
            frame_resized = cv2.flip(frame_resized, 1)
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]
                    middle_knuckle = hand_landmarks.landmark[10]

                    h, w, _ = frame_resized.shape
                    cx, cy = int(index_tip.x * w), int(index_tip.y * h)

                    if middle_tip.y < middle_knuckle.y:
                        last_x, last_y = None, None
                    else:
                        if last_x is not None:
                            cv2.line(canvas, (last_x, last_y), (cx, cy), (0, 0, 0), 15)
                            cv2.line(frame_resized, (last_x, last_y), (cx, cy), (0, 0, 0), 15)
                        last_x, last_y = cx, cy
            else:
                last_x, last_y = None, None

            combined = np.hstack((frame_resized, canvas))
            _, buffer = cv2.imencode('.jpg', combined)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            continue

    cap.release()

@app.route('/')
def landing():
    print("Rendering landing.html")
    return render_template('landing.html')

@app.route('/ocr')
def index():
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_canvas', methods=['POST'])
def clear():
    clear_canvas()
    return jsonify({'status': 'Canvas cleared!'})

@app.route('/recognize_text', methods=['POST'])
def recognize():
    global canvas
    text = recognize_text(canvas)
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True)