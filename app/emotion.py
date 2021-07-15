# Imports
from tensorflow import keras

# Emotion Model Class
class EmotionModel():
    def __init__(self, dir):
        self.model = None
        self.dir = dir

    def import_model(self):
        # Load emotion deep learning model
        self.model = keras.models.load_model(self.dir)
