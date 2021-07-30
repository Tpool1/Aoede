import pandas as pd

from play_text import play_text
from packages.voice_to_date import voice_to_date

def plot_heart_rate():
    df = pd.read_csv('data\\apple_health_export\\HeartRate.csv')

    play_text('What day would you like to get heart rate data from?')

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

    df = df.loc[date]

    print(df)