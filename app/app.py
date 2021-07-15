import json,time
from camera import VideoCamera
from flask import Flask, render_template, request, jsonify, Response
import requests
import base64,cv2

# Machine Learning Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

# Modules
from emotion import EmotionModel

# Create model object
model = EmotionModel('../emotion_model_5')
model.import_model()

app = Flask(__name__)
prediction = ""

def camera():
    cap=cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/cam.png",img)

        return json.dumps({'status': 'OK', 'result': "static/cam.png"})
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cap.release()
    return json.dumps({'status': 'OK', 'result': "static/cam.png"})

def gen(camera):
    while True:
        data = camera.get_frame(model)

        if data is not None:
            frame = data[0]
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            continue  
            
@app.route("/")
def home():
    return render_template('app.html')

@app.route("/webcam")
def webcam():
    return render_template('webcam.html')

@app.route("/video")
def video():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)