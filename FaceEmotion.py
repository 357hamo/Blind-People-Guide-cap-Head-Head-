from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from SR import say_speech, say_arabic

   # parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.106-0.65.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

camera = cv2.VideoCapture(0)

def detect_emotions():
        
    
        
    frame = camera.read()[0]
    #frame = cv2.flip(frame,0)
    #reading the frame
    frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    
    
    # frameClone = frame.copy()
    if len(faces) > 0:
        # global preds
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
    #Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
    #the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        say_speech(label)
    else:
        say_speech("no faces")


camera.release()    
cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_emotions()
    