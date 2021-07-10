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

app = Flask(__name__)

def camera():
    cap=cv2.VideoCapture(0)
    while True:
        ret,img=cap.read()
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/cam.png",img)

        # return render_template("camera.html",result=)
        time.sleep(0.1)
        return json.dumps({'status': 'OK', 'result': "static/cam.png"})
        if cv2.waitKey(0) & 0xFF ==ord('q'):
            break
    cap.release()
    # file="/home/ashish/Downloads/THOUGHT.png"
    # with open(file,'rb') as file:
    #     image=base64.encodebytes(file.read())
    #     print(type(image))
    # return json.dumps({'status': 'OK', 'user': user, 'pass': password});
    return json.dumps({'status': 'OK', 'result': "static/cam.png"})

def gen(camera):
    model = keras.models.load_model('../emotion_model_5')
    
    while True:
        data = camera.get_frame()

        if data is not None:
            frame = data[0]

            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            continue
        
        final_image = data[1]

        if final_image is not None:
            Predictions = model.predict(final_image)
            label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'} 
            print(label_to_text[np.argmax(Predictions)])


@app.route("/")
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)