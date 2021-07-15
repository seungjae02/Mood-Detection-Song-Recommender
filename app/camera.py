import cv2
from imutils.video import WebcamVideoStream
import time
import numpy as np
from settings import *

class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()

    def __del__(self):
        self.stream.stop()

    def get_frame(self, model):
        image = self.stream.read() # Read stream

        detector = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml') # Load face detection model from OpenCV
        face = detector.detectMultiScale(image, 1.1, 4)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_roi = None

        ### Detects all faces within the image frame
        for (x,y,w,h) in face:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]

            cv2.rectangle(image,(x,y), (x+w, y+h), SPOTIFY_GREEN, 2) # Draw rectangle arounf detected face

            faces = detector.detectMultiScale(roi_gray)
            if len(faces) == 0:
                pass
            else:
                for (ex,ey,ew,eh) in faces:
                    face_roi = roi_color[ey:ey+eh, ex:ex+ew]


        if face_roi is not None:            
            final_image = cv2.resize(face_roi, (224,224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image/255.0 # Data Normalization
            
        else:
            final_image = None


        if final_image is not None:
            Predictions = model.model.predict(final_image) # Returns predictions - emotions indexed by integer
            label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'} 
            prediction = label_to_text[np.argmax(Predictions)] # Max value is the closest estimate to the actual emotion
            label_pos = (x, y - 10) # Position of emotion text is right above the box
            cv2.putText(image, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, SPOTIFY_GREEN, 2)
        else:
            cv2.putText(image, "Face Not Detected", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, SPOTIFY_GREEN, 2)


        if image is not None:
            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())
            data.append(final_image)
        
            return data
        return None