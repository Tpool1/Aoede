from plugins.get_cpu_load import get_cpu_load
from plugins.get_time import get_time
from nlp_tools import *
from nlp_tools.get_parsed_input import get_parsed_input
from models import *
from play_text import play_text
from get_input import get_input
import random

from models.cancer_ml import cancer_ml
from models.soteria import soteria

class core:

    def initialize(self):

        # determine dictionary of possible tasks with their corresponding functions
        self.possible_tasks = {'cpu': get_cpu_load, 'time': get_time, 'soteria': soteria, 'cancer': cancer_ml}

        greetings = ["Hello. What can I help you with today?", "How can I assist you?", "What's up?"]
        greet_choice = random.choice(greetings)

        play_text(greet_choice)

    def get_function(self):

        task_found = False
        while not task_found:
            input_received = False
            while not input_received:
                try:
                    user_input = get_parsed_input()
                    input_received = True
                except:
                    play_text("I am having trouble hearing you. Please try again.")

            for tag in list(user_input.values()):
                if tag in list(self.possible_tasks.keys()):
                    task_func = self.possible_tasks[tag]
                    task_found = True
                
            if not task_found:
                play_text("I cannot identify what you want to do. Try again.")
            else:
                task_func()

    def run(self):
        self.initialize()
        self.get_function()
        