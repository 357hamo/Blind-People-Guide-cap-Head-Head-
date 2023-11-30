import cv2
from pyzbar import pyzbar
from SR import say_speech,say_arabic
import os
global barcode_info

def read_barcodes(frame):
    global barcode_info
    barcodes = pyzbar.decode(frame)
    for barcode in barcodes:
        #1
        x, y , w, h = barcode.rect
        # x: data, y: type,h:rect,w:point
        barcode_info = barcode.data.decode('utf-8')
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0, 255, 0), 2)
        print(barcode_info)
        #2
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(frame, barcode_info,(50,50),font, 2.0, (255, 255, 255), 1)
        say_arabic(barcode_info)
    return frame,barcode_info

def recognize_bar_qr():
    #1
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    frame = cv2.flip(frame,0)
    #2
    while ret:
        ret, frame = camera.read()
        frame = read_barcodes(frame)
        cv2.imshow('Barcode/QR code reader', frame)
        cv2.waitKey(15)
        break
        
    #3
    camera.release()
    cv2.destroyAllWindows()
#4




if __name__ == '__main__':
    recognize_bar_qr()