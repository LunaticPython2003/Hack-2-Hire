from flask import Flask, Response, render_template, jsonify
import cv2
from scripts.camera import gen_frames
from scripts.camera import predicted_characters

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
    return jsonify(predicted_characters=predicted_characters)

if __name__=="__main__":
    app.run(debug=True)