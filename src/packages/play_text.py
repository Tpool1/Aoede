import pyttsx3
from packages.write_conversation_data import write_conversation_data

def play_text(text):

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
    write_conversation_data(text)
