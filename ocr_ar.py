import cv2
import os
import time
import uuid
import easyocr
from SR import say_arabic
# Load the model
reader = easyocr.Reader(['ar'],gpu=False)
global img_name

IMAGES_PATH = 'images'
num_img = 3

def start_video():
    global img_name
    cap = cv2.VideoCapture(0)
    print('collecting image for arabic ocr')
    time.sleep(5)
    for img_num in range(num_img):
        ret, frame = cap.read()
        frame = cv2.flip(frame,0)

        img_name = os.path.join(IMAGES_PATH, '1' + 'a' + '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(img_name, frame)
        #cv2.imshow('frame', frame)
        time.sleep(2)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    cap.release()


def read_text(image_name, model_name, in_line=False):
    # Read the data
    text = model_name.readtext(image_name, detail=0, paragraph=in_line)

    return '\n'.join(text)

def OCR_AR():
    global img_name
    start_video()
    image_path = img_name
    ar_text = read_text(image_path, reader)
    say_arabic(ar_text)

if __name__ == "__main__":
    OCR_AR()