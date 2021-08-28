from core import core
from packages.clear_user_data import clear_user_data
from packages.profile import profile
from packages.load_profiles import load_profiles

import os

class Api:

    def __init__(self):
        self._window = None
    
    def start(self):
        self.assistant = core()
        self.assistant.run()

    def set_window(self, window):
        self._window = window

    def pause(self):
        self.assistant.quit()

    def quit(self):
        self._window.destroy()

    def clear_user_data(self):
        clear_user_data()

    def add_profile(self, name):
        p = profile(name)
        
    def load_profiles(self):
        profiles = load_profiles()

        names = []
        for profile in profiles:
            names.append(profile.name)

        names = str(names)

        return names

    def load_conversation(self, name):
        root = "data\\profiles"
        profile_path = os.path.join(root, name)
        convo_path = os.path.join(profile_path, "conversations.txt")

        f = open(convo_path, "r")
        lines = f.readlines()
        f.close()

        lines = str(lines)
        
        return lines
        