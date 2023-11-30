import json
import random 
import torch
import pickle
from FaceRecognition import recognize_face
from currency import recognize_coin
from FireDetection import FireDectetion
from ObjectDetectionCOCO import ObjectDetectionCOCO
from ImageCaption import get_caption
from QrBarCode import recognize_bar_qr
from ocr_ar import OCR_AR
from ocr_en import OCR_EN
from ColorRecognition import recognize_color
from model import NeuralNetwork 
from nlp import tokenize,bag_of_word
from SR import say_speech,speech_to_text
from timeit import default_timer as timer
    
start = timer()
device = torch.device('cpu') # device = cpu
detector_fire = FireDectetion(model_name = 'best.pt')   

modelCaption=pickle.load(open("image caption/model.bin",'rb'))
tokenizer=pickle.load(open("image caption/tokenizer.bin",'rb'))
image_processor=pickle.load(open("image caption/image_processor.bin",'rb'))

with open('intents.json') as f:
    intents = json.load(f)

file = 'data.pth'
data = torch.load(file)


input_size = data['input_size']
model_state = data['model_state']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']

model = NeuralNetwork(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()

def get_response(pattern):
    sentence = tokenize(pattern)
    BoW = bag_of_word(sentence,all_words)
    BoW = torch.from_numpy(BoW).to(device)
    output = model.forward_propagation(BoW)
    # print(output)
    _,predicted = torch.max(output,dim=-1)
    tag = tags[predicted.item()] # give prediction tag for input speech
    # print(tag)
    probs = torch.softmax(output,dim=-1)  # to make output probability between -1 and 1
    # print(props)
    prob = probs[predicted.item()] # to select the big probability
    # print(prob)
    return prob,tag

while True:
    try:
        pattern = speech_to_text()
        print(pattern)
        prob,tag = get_response(pattern)
        print(tag)
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == "greeting":
                    say_speech(random.choice(intent['responses']))
            if tag == 'Recognition':
                recognize_face()
            elif tag == 'currency':
                recognize_coin()
            elif tag == 'reading_en':
                OCR_EN()
            elif tag == 'reading_ar':
                OCR_AR()
            elif tag == 'code':
                recognize_bar_qr()
            elif tag == 'obstacles':
                ObjectDetectionCOCO()
            elif tag == 'fire':
                detector_fire()
            elif tag == 'color':
                recognize_color()
            elif tag == 'describe':
                # get the caption
                caption = get_caption(modelCaption, image_processor, tokenizer)
                say_speech(caption)
            elif tag == 'thank':
                say_speech(random.choice(intent['responses']))
                break
            else:
                say_speech("I don't understand")
    except:
        pass
end = timer()
print(end - start)
