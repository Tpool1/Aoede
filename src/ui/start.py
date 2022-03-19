import webview
from ui.js_api import Api

def start():
    api = Api()
    window = webview.create_window('Aoede', url='../assets/index.html', js_api=api)
    api.set_window(window)
    webview.start()
    