from Detector import *

def ObjectDetectionCOCO():

    videoPath = 0 # For webcam, replace this with 0

    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

    modelPath = "frozen_inference_graph.pb"
    classesPath = "coco.names"

    detector = Detector(videoPath, configPath, modelPath, classesPath)

    detector.onVideo()
if __name__ == '__main__':
    ObjectDetectionCOCO()
