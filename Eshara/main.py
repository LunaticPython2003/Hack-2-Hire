from flask import Flask, Response, render_template, jsonify
import cv2
import scripts.testing_sentences as testing
import sys
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import math
import time
import json

continue_camera_streaming = True

# Classifier class to load and compile the model
class Classifier:
    def __init__(self, modelPath, labelsPath):
        self.model = load_model(modelPath, compile=False)  # Load the model without compiling
        # Manually compile the model with an optimizer, loss, and metrics
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Load the labels
        with open(labelsPath, "r") as f:
            self.labels = f.read().splitlines()

    def getPrediction(self, img, draw=True):
        prediction = self.model.predict(np.expand_dims(img, axis=0))
        classIndex = np.argmax(prediction)
        confidence = prediction[0][classIndex]
        return confidence, classIndex

def get_output_string():
    output = testing.ret_text()
    return output

def gen_frames():  
    global continue_camera_streaming
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector(maxHands=2)
    classifier = Classifier("models/keras_model_digits.h5", "models/labels_digits.txt")

    offset = 20
    imgSize = 300

    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    predicted_characters = ""
    last_recognized_character = ""
    last_x1, last_y1, last_x2, last_y2 = 0, 0, 0, 0
    last_recognition_time = 0
    min_time_gap = 3
    display_duration = 3  # Duration to display recognized character and box (in seconds)
    no_hand_time = 2.5

    # Initialize variables for character recognition timing
    holding_position_start_time = None
    holding_position = False

    while continue_camera_streaming:
        success, img = cap.read()  # Read the camera frame
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            recognized_character = labels[index]
            current_time = time.time()

            if not holding_position:
                holding_position_start_time = current_time
                holding_position = True
            else:
                elapsed_holding_time = current_time - holding_position_start_time
                if elapsed_holding_time >= 3.0:
                    predicted_characters += ""
                    holding_position = False  # Reset the flag

            if current_time - last_recognition_time >= display_duration:
                cv2.rectangle(imgOutput, (last_x1, last_y1), (last_x2, last_y2), (0, 0, 0), 4)
            else:
                cv2.putText(imgOutput, f"Character Recognized: {last_recognized_character}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (176, 224, 230), 4)

            if current_time - last_recognition_time >= min_time_gap:
                if recognized_character != last_recognized_character:
                    predicted_characters += recognized_character
                    last_recognized_character = recognized_character
                    last_x1, last_y1, last_x2, last_y2 = x - offset, y - offset, x + w + offset, y + h + offset
                    last_recognition_time = current_time
        
        else:
            elapsed_no_hands_time = time.time() - last_recognition_time
            if elapsed_no_hands_time >= no_hand_time:
                predicted_characters += " "
                cv2.putText(imgOutput, "Space Added", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 4)
                last_recognized_character = " "
                last_recognition_time = time.time()
                holding_position = False

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        imgOutput = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + imgOutput + b'\r\n')

        d = {"key": predicted_characters}
        with open('predicted_characters.json', 'w') as json_file:
            json.dump(d, json_file)

    cap.release()
    cv2.destroyAllWindows()

app = Flask(__name__)

@app.route('/converter')
def test():
    return render_template('converter.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_characters')
def get_characters():
    print("Hello World")

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global continue_camera_streaming
    continue_camera_streaming = False
    print("Camera streaming stopped")
    output_string = get_output_string()
    print(output_string)
    return jsonify(output_string=output_string), 200

if __name__ == "__main__":
    app.run(debug=True)
    