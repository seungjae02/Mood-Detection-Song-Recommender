import cv2
from imutils.video import WebcamVideoStream
import time
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()

    def __del__(self):
        self.stream.stop()

    def get_frame(self):
        image = self.stream.read()
        #time.sleep(1)
        #image = cv2.imread('../happyasian2.jpeg')

        detector = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
        face = detector.detectMultiScale(image, 1.1, 4)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_roi = None

        for (x,y,w,h) in face:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]

            cv2.rectangle(image,(x,y), (x+w, y+h), (0, 255, 0), 2)

            faces = detector.detectMultiScale(roi_gray)
            if len(faces) == 0:
                print("Face not detected")
            else:
                for (ex,ey,ew,eh) in faces:
                    face_roi = roi_color[ey:ey+eh, ex:ex+ew]

        if face_roi is not None:            
            final_image = cv2.resize(face_roi, (224,224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image/255.0
        else:
            final_image = None

        if image is not None:
            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())
            data.append(final_image)

            return data
        
        return None