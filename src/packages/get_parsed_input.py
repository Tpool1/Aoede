from packages.get_input import get_input
from packages.partition import partition
from packages.filterStops import filterStops
from packages.stem import stem
from packages.tag import tag

def get_parsed_input():
    user_input = get_input()
    user_input = partition(user_input)
    filtered_list = filterStops(user_input)
    filtered_list = stem(filtered_list)
    parsed_input = tag(filtered_list)

    return parsed_input
