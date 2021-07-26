import psutil
from play_text import play_text

def get_battery_percentage():
    battery = psutil.sensors_battery()
    percent = str(battery.percent)

    play_text('Your battery is at ' + percent + ' percent')
