from flask import Flask, Response, render_template
import cv2
from scripts.camera import gen_frames

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


if __name__=="__main__":
    app.run()