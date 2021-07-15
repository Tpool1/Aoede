from NLP import NLP
import os

nlp_engine = NLP()

nlp_engine.run()

# delete temporary audio files after running
dir_list = os.listdir("../Asclepius")

for file in dir_list:
    if file[-4:] == ".wav" or file[-4:] == ".mp3":
        os.remove(file)