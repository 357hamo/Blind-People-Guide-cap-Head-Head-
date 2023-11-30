import cv2
import numpy as np
import face_recognition as fr
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model
from SR import STT,say_arabic,say_speech,translate
import os
import time

path = r"faces"
global encodeListknown
global classNames
# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.106-0.65.hdf5'
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

def detect_emotions(img):
    frame = imutils.resize(img,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        global preds
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                # cv2.putText(frameClone, label, (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                # cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)

    # cv2.imshow('your_face', frameClone)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    return label

def img_and_name(dir):   # get img and names of persons
    images = []
    classNames = []
    mylist = os.listdir(dir)
    for cls in mylist:
        img = cv2.imread(os.path.join(dir, cls))
        images.append(img)
        classNames.append(os.path.splitext(cls)[0])
    return images, classNames


def findEncodings(images):   # Encode images
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def isEncoded(name, img, y1,x2,y2,x1):  # check if input image is one from the encoded faces
    try:
#         label_emotion = detect_emotions(img)
        if name == " شخص غير معروف":
            say_speech("Enter name ")
            imgName = STT()
            imgName = translate(imgName)
            say_arabic('الاسم هو '+ imgName)
#             say_speech(label_emotion)
            img = img[y1-50:y2+100, x1-50:x2+100]
            imgPath = os.path.join(path, imgName)
            cv2.imwrite(imgPath+'.jpg', img) 
    except:
        print("images not saved...\n\n\n")


def matching(img):
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)   
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = fr.face_locations(imgS)  
    encodeCurFrame = fr.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = fr.compare_faces(encodeListknown, encodeFace)
        faceDist = fr.face_distance(encodeListknown, encodeFace)   
        matchIndex = np.argmin(faceDist)
        label_emotion = detect_emotions(img)

        if matches[matchIndex] and faceDist[matchIndex] < .6:
            name = classNames[matchIndex].title()    
        else:
            name = " شخص غير معروف"
        say_arabic(name)
        say_speech(label_emotion)
        # print(name)
        y1,x2,y2,x1 = faceLoc
        isEncoded(name, img, y1,x2,y2,x1)

def recognize_face():
    global encodeListknown
    global classNames   
    while True:
        images, classNames = img_and_name(path)
        encodeListknown = findEncodings(images)
        cap = cv2.VideoCapture(0)
        qwe=0
        while True:
            qwe+=1
            success, img = cap.read()
            frame = cv2.flip(img,0)
            images, classNames = img_and_name(path)
            if len(images) > len(encodeListknown):
                encodeListknown = findEncodings(images)
            matching(frame)
            if qwe>1:
                cap.release()  
                break
            else:
                continue
        time.sleep(3)
        break

if __name__ == '__main__':
    recognize_face()
