import pyttsx3

def play_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    