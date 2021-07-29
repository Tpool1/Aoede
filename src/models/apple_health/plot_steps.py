import pandas as pd
import matplotlib.pyplot as plt

def plot_steps(apple_export_path):
    df = pd.read_xml(apple_export_path)
    print(df['activeEnergyBurned'])

plot_steps('C:\\Users\\trist\\cs_projects\\Asclepius\\data\\apple_health_export\\export.xml')