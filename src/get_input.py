import speech_recognition as sr
from play_text import play_text

def get_input():

    input_received = False
    while not input_received:
        r = sr.Recognizer()

        mic = sr.Microphone()

        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            input_received = True
        except:
            play_text("I am having trouble hearing you. Please try again.")

    return text
    