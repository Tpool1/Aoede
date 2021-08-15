from core import core

class Api:

    def __init__(self):
        self._window = None
        self.paused = False
        self.started = False
    
    def start(self):
        self.assistant = core()
        self.assistant.run()

    def set_window(self, window):
        self._window = window

    def pause(self):
        self.assistant.quit()

    def quit(self):
        self._window.destroy()
