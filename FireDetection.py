import torch
import cv2
# import beepy as beep
import ssl
from SR import say_speech

class FireDectetion: 
    
    def __init__(self,model_name):

        self.model = self.load_model(model_name)
        self.classes = self.model.names 
        self.device = 'cpu'
        # print("Using Device: ", self.device)

    def get_video_capture(self): 
        
        return cv2.VideoCapture(0)

    def load_model(self, model_name): 
        ssl._create_default_https_context = ssl._create_unverified_context

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name,force_reload=False)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:,-1],results.xyxyn[0][:, :-1]
        return labels ,cord
    
    def class_to_label(self, x):
        return self.classes [int(x)]

    def plot_boxes (self, results, frame):

        labels, cord = results

        n = len(labels)

        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):

            row = cord[i]

            if row[4] > 0.7:

                # beep.beep(3)
                say_speech(self.class_to_label(labels[i]))
                
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                bgr= (0, 255, 0)
                cv2. rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9 , (255,0,0),1)
                break
        return frame
    def __call__(self):
        cap = self.get_video_capture()
        
        while True:
            ret ,frame = cap.read()
            frame = cv2.flip(frame,0)
            results = self.score_frame(frame)
            frame2 = self.plot_boxes(results, frame)
            break
            # cv2.imshow('FIRE' , frame2)
            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     break
        cap.release() 
        return ret      

if __name__ == '__main__': 
    detector = FireDectetion(model_name = 'best.pt')        
    detector()
