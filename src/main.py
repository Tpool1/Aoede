import speech_recognition as sr
import os
from get_input import get_input
from play_text import play_text
from nlp_tools import *
from core import core

assistant = core()
assistant.run()

# delete temporary audio files after running
dir_list = os.listdir("../Asclepius")

for file in dir_list:
    if file[-4:] == ".wav" or file[-4:] == ".mp3":
        os.remove(file)
        