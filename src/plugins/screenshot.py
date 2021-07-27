import numpy as np
import cv2
import pyautogui
from play_text import play_text

def screenshot():

    ss = pyautogui.screenshot()

    ss = cv2.cvtColor(np.array(ss),
                        cv2.COLOR_RGB2BGR)

    cv2.imwrite('screenshot.png', ss)

    play_text('Screenshot saved successfully')

