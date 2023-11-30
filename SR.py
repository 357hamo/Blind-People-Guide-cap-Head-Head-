import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from translate import Translator
from playsound import playsound
import os
from datetime import datetime

def speech_to_text():
    try:
        r = sr.Recognizer()
        with sr.Microphone() as src:
            say_speech("Microphone is on")
            r.energy_threshold = 4000 # to prevent recognition from occurring when there is only background noise.
            audio = r.listen(src)
        say_speech("Microphone off")
        pattern = r.recognize_google(audio, language='en_US') # ar-EG
        return pattern
    except:
        pass

def say_speech(text):
    text_to_speech = pyttsx3.init()
    text_to_speech.setProperty('rate',125)
    text_to_speech.say(text)
    text_to_speech.runAndWait()

def STT():
    try:
        r = sr.Recognizer()
        with sr.Microphone() as src:
            say_speech("Microphone on")
            r.energy_threshold = 4000 # to prevent recognition from occurring when there is only background noise.
            audio = r.listen(src)
        say_speech("Microphone off")
        pattern = r.recognize_google(audio, language='ar-EG') 
        return pattern
    except:
        pass

def say_arabic(text):
    aud = gTTS(text,lang='ar')
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "voice"+date_string+".mp3"
    aud.save(filename)
    playsound(filename)
    os.remove(filename)

def translate(text):
    T=Translator(from_lang = "Arabic", to_lang = "English")
    translation = T.translate(text)
    return translation

if __name__ == "__main__":
    pattern = speech_to_text()
    print(pattern)
    say_speech(pattern)
    name = STT()
    print(name)
    say_arabic(name)
