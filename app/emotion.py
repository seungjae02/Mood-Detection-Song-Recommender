# Machine Learning Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

class EmotionModel():
    def __init__(self, dir):
        self.model = None
        self.dir = dir

    def import_model(self):
        self.model = keras.models.load_model(self.dir)

model = EmotionModel('../emotion_model_5')
model.import_model()
