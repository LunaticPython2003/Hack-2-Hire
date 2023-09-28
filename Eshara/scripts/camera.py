import cv2
import sys
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

def gen_frames():  
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        detector = HandDetector(maxHands=2)
        classifier = Classifier("models/keras_model_digits.h5", "models/labels_digits.txt")

        offset = 20
        imgSize = 300

        folder = "/Final_Model"
        counter = 0

        labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        predicted_characters = ""
        last_recognized_character = ""
        last_x1, last_y1, last_x2, last_y2 = 0, 0, 0, 0
        last_recognition_time = 0
        min_time_gap = 3
        display_duration = 3  # Duration to display recognized character and box (in seconds)
        no_hand_time=2.5

        text_box_position = (20, 480)  # Adjust the position as needed
        font_scale = 1
        font_thickness = 2
        font_color = (255, 255, 255)  # White color
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Initialize variables for character recognition timing
        start_recognition_time = None
        holding_position_start_time = None
        holding_position = False
        while True:
            success, img = cap.read()  # read the camera frame
            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Calculate hand position
                hand_position = (x, y)

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                recognized_character = labels[index]
                current_time = time.time()
                
                if not holding_position:
                    # Start tracking hand position
                    holding_position_start_time = current_time
                    holding_position = True
                else:
                        # Check if the hand position is held for 3 seconds
                    elapsed_holding_time = current_time - holding_position_start_time
                    if elapsed_holding_time >= 3.0:
                        predicted_characters += ""
                        holding_position = False  # Reset the flag
                        start_recognition_time = current_time
                current_time = time.time()
                if current_time - last_recognition_time >= display_duration:
                    display_text = ""
                    cv2.rectangle(imgOutput, (last_x1, last_y1),
                            (last_x2, last_y2), (0, 0, 0), 4)  # Remove the box after display_duration
                    cv2.putText(imgOutput, recognized_character, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0,0,0), 4)
                
                else:
                    cv2.putText(imgOutput, f"Character Recognized: {last_recognized_character} ", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                    cv2.rectangle(imgOutput, (x - offset, y - offset),
                                (x + w + offset, y + h + offset), (176, 224, 230), 4)
                    cv2.putText(imgOutput, recognized_character, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0,255,0), 2,cv2.LINE_AA)

                if current_time - last_recognition_time >= min_time_gap:
                    if recognized_character != last_recognized_character:
                        predicted_characters += recognized_character
                        last_recognized_character = recognized_character
                        last_x1, last_y1, last_x2, last_y2 = x - offset, y - offset, x + w + offset, y + h + offset
                        last_recognition_time = current_time
            
            else: 
                current_time = time.time()
                elapsed_no_hands_time = current_time - last_recognition_time
                if elapsed_no_hands_time >= no_hand_time:
                    # Recognize a space character
                    predicted_characters += " "
                    cv2.putText(imgOutput, "Space Added", (20,50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0,0,255), 4)
                    last_recognized_character = " "
                    last_x1, last_y1, last_x2, last_y2 = 0, 0, 0, 0
                    last_recognition_time = current_time
                    holding_position = False  # Reset the flag
            ret, buffer = cv2.imencode('.jpg', imgOutput)
            imgOutput = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + imgOutput + b'\r\n')  # concat frame one by one and show result