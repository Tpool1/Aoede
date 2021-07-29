import pandas as pd
import matplotlib.pyplot as plt

def workout_info():
    df = pd.read_csv('data\apple_health_export\Workout.csv')
    print(df.head())
    