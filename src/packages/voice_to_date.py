from urllib.parse import urldefrag
from packages.get_parsed_input import get_parsed_input
from packages.txt_num_dicts import month_dict
from play_text import play_text
import re

# function to convert a date that is said verbally to numerical format (year-month-day) (Ex. 2021-07-23)

def voice_to_date():
    
    date_received = False
    while not date_received:
        user_input = get_parsed_input()
        try:
            year = user_input['CD2']
            month = user_input['NNP']
            day = user_input['CD']
            date_received = True
        except KeyError:
            play_text('I could not identify this date. Please try again.')


    day = re.sub("[^0-9]", "", day)
    
    if len(day) != 2:
        day = '0' + day

    month = month_dict[month]

    full_date = year + '-' + month + '-' + day

    return full_date
