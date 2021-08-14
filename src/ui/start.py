import webview
from core import core

def on_loaded():
    assistant = core()
    assistant.run()

def start():
    window = webview.create_window('Asclepius', url='../assets/index.html')
    window.loaded += on_loaded
    webview.start()
