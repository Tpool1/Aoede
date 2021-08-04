import pandas as pd

from packages.play_text import play_text
from packages.voice_to_date import voice_to_date
from packages.get_parsed_input import get_parsed_input
from packages.txt_num_dicts import placings_dict

def get_workout_distance():
    df = pd.read_csv('data\\apple_health_export\\Workout.csv')
    
    play_text('What date was your workout?')

    workout_date = voice_to_date()

    # remove times from date column 
    date_column = df['startDate']

    date_column = list(date_column)

    # since date is only element in each cell before the space, split each string and collect first element
    i = 0
    for date in date_column:
        date = date.split()
        date = date[0]

        date_column[i] = date

        i = i + 1

    df['startDate'] = date_column

    df = df.set_index('startDate')

    # fiter data for given date
    df = df.loc[workout_date]

    if df.shape[0] > 1:
        play_text('There are ' + str(df.shape[0]) + ' on this date. Which would you like?')
        user_response = get_parsed_input()
        answer = user_response['RB']
        answer = placings_dict[answer]

        df = df.iloc[answer]

    workout_distance = str(df['totalDistance']) + ' ' + str(df['totalDistanceUnit'])

    play_text('Your workout was ' + workout_distance + ' long')
