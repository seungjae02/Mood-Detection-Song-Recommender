import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

dataDirectory = "datasets/training/" # training dataset

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
classes = ["0", "1", "2", "3", "4", "5", "6"] # list of classes

IMG_SIZE = 224 # ImageNet => 224 x 224

# Read all images and convert them into an array
trainingData = []

def create_training_data():
    for category in classes:
        path = os.path.join(dataDirectory, category)
        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                trainingData.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(trainingData)

features = []
labels = []

for feature, label in trainingData:
    features.append(feature)
    labels.append(label)

features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 3) # Conversion to 4 Dimension

features = features/255.0 # Normalization before training

labels = np.array(labels)

# Deep learning model for training - Transfer Learning
model = tf.keras.applications.MobileNetV2() # Pre-trained model

# Transfer Learning - Tuning
base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output) # Adding new layer, after the output of global pooling layer
final_ouput = layers.Activation('relu')(final_output) # Activation function
final_output = layers.Dense(64)(final_ouput)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_ouput) # My classes are 07, classification layer

new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

new_model.fit(features, labels, epochs=25)
new_model = tf.keras.models.load_model('Final_model_95p07.h5')

new_model.evaluate # Test data, I will not use test, deploy live image demo

frame = cv2.imread("datasets/training/happyman.jpeg")

# Face detection algorithm

faceCascate = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.1, 4)
for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face not detected")
    else:
        for (ex,ey,ew,eh) in facess:
            face_roi = roi_color[ey:ey+eh, ex:ex+ew]


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))        