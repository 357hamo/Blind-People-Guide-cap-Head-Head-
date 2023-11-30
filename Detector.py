import cv2
import numpy as np
from SR import say_speech
import time


np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classPath = classPath
        
        
        ###############################
        
        
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        
        self.readClasses()
        
    def readClasses(self):
        with open(self.classPath, 'r')as f:
            self.classesList = f.read().splitlines()
            
        self.classesList.insert(0, '__Background__')
        
        self.colorlist = np.random.uniform(low=0, high=255, size=(len(self.classesList),  3))
        
        # print(self.classesList)
        
    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        
        if (cap.isOpened()==False):
            print ("Error opening.......")
            return 
        
        (success, image) = cap.read()
        
        while success:

            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.4)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold= 0.2)

            if len(bboxIdx) != 0:

                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx [i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])] 
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorlist[classLabelID]] 
                    
                    displayText = "{}".format(classLabel)
                    print(displayText)
                    say_speech(displayText)
            break
            #         x,y,w,h = bbox
                    
            #         cv2.rectangle(image, (x,y), (x+w, y+h), color=classColor, thickness=1)
                  
            #         cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor,2)
                    
            #         lineWidth = min(int(w * 0.3), int(h * 0.3))

            #         cv2.line(image, (x,y), (x + lineWidth, y), classColor, thickness= 5) 
            #         cv2.line(image, (x,y), (x, y + lineWidth), classColor, thickness= 5)
             
            #     ##############################
                
            #         cv2.line(image, (x + w,y), (x + w - lineWidth, y), classColor, thickness= 5) 
            #         cv2.line(image, (x + w,y), (x + w, y + lineWidth), classColor, thickness= 5)
                    
            # cv2.imshow("Result", image)

            # key = cv2.waitKey(15) & 0xFF
            # if key == ord("q"):
            #     break

        (success, image) = cap.read()
        frame = cv2.flip(image,0)

        # cv2.destroyAllWindows()
            
        