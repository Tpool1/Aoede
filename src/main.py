import speech_recognition as sr
import os
from get_input import get_input
from play_text import play_text
from nlp_tools import *
from core import core

assistant = core()
assistant.run()
