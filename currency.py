import cv2 
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from SR import say_speech

#def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.5, num_test_images=10, savepath='/content/results', txt_only=False):

cap=cv2.VideoCapture(0)
# هتغير في 3 حجات دوولت
PATH_TO_MODEL="custom_model_lite/detect.tflite" #هتغير هنا المسار 
PATH_TO_LABELS='custom_model_lite/labelmap.txt' #هتغير هنا المسار 
min_conf=0.5 # هتغير هنا برحتك دي نسبه الدقه

# Load the label map into memory
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
        
# Load the Tensorflow Lite model into memory
interpreter = Interpreter(model_path=PATH_TO_MODEL)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

def recognize_coin():
  while True:
      # Grab filenames of all images in test folder
      _,img=cap.read()
      frame = cv2.flip(img,0)
      key=cv2.waitKey(15)
      # if key==ord('p'):
      # Load image and resize to expected shape [1xHxWx3]
      image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      imH, imW, _ = img.shape 
      image_resized = cv2.resize(image_rgb, (width, height))
      input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
      if float_input:
          input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
      interpreter.set_tensor(input_details[0]['index'],input_data)
      interpreter.invoke()

    # Retrieve detection results
      boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
      classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
      scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

      #detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
      for i in range(len(scores)):
          if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
              # Get bounding box coordinates and draw box
              # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
              ymin = int(max(1,(boxes[i][0] * imH)))
              xmin = int(max(1,(boxes[i][1] * imW)))
              ymax = int(min(imH,(boxes[i][2] * imH)))
              xmax = int(min(imW,(boxes[i][3] * imW)))
            
              cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 3)

            # Draw label
              object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
              label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
              labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
              label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
              cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
              cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3) # Draw label text
              print(object_name) #هيظهر ال class
              say_speech(object_name)
      break
  cap.release()
  cv2.destroyAllWindows()  
         #       detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
    # cv2.imshow('image',img)# ابقي غير دي بردو
    # if key==ord('q'):
    #     break


if __name__ == '__main__':
    recognize_coin()