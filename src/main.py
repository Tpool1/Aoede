from NLP import NLP
import os

nlp_engine = NLP()

nlp_engine.run()

# temporary files to be deleted after program runs
rm_files = ["bot_recording.mp3", "bot_recording0.mp3", "bot_recording01.mp3", 
            "bot_recording012.mp3", "bot_recording0123.mp3", "bot_recording01234.mp3", 
            "user_recording.wav"]

for file in rm_files:
    os.remove(file)