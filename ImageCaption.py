import pickle
import cv2
from SR import say_speech

model=pickle.load(open("image caption/model.bin",'rb'))
tokenizer=pickle.load(open("image caption/tokenizer.bin",'rb'))
image_processor=pickle.load(open("image caption/image_processor.bin",'rb'))
# a function to perform inference
def get_caption(model, image_processor, tokenizer):
    cap = cv2.VideoCapture(0)
    ret,image = cap.read()
    image = cv2.flip(image,0)
    img = image_processor(image, return_tensors="pt")
    # generate the caption (using greedy decoding by default)
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    cap.release()
    return caption
# get the caption
caption = get_caption(model, image_processor, tokenizer)
print(caption)
say_speech(caption)
# if __name__ == "__main":
    
