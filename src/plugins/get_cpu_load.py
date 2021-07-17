import psutil

def get_cpu_load():
    # get load over the period of 1 second
    cpu_load = psutil.cpu_percent(1)

    return cpu_load
