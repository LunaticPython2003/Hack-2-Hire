import cv2
import sys
def gen_frames():  
    if sys.platform() == "darwin":
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        success, frame = video.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result