from core import core
from packages.clear_user_data import clear_user_data
from packages.profile import profile

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
        