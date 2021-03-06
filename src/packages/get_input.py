import speech_recognition as sr
from packages.play_text import play_text
from packages.write_conversation_data import write_conversation_data

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

    write_conversation_data(text)

    return text
    