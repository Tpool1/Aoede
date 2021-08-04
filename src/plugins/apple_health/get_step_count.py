import pandas as pd

from packages.play_text import play_text
from packages.voice_to_date import voice_to_date

def get_step_count():
    df = pd.read_csv('data\\apple_health_export\\StepCount.csv')

    play_text('What date?')

    date = voice_to_date()

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

    # filter data for given date
    df = df.loc[date]

    val_col = df['value']
    
    step_count = val_col.sum()

    play_text('You walked ' + str(step_count) + ' steps on this day')
