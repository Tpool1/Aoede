from datetime import datetime
from play_text import play_text

def get_time():
    time = datetime.now()
    
    play_text('The time is ' + str(time))
    