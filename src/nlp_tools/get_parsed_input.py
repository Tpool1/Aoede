from get_input import get_input
from nlp_tools.partition import partition
from nlp_tools.filterStops import filterStops
from nlp_tools.stem import stem
from nlp_tools.tag import tag

def get_parsed_input():
    user_input = get_input()
    user_input = partition(user_input)
    filtered_list = filterStops(user_input)
    filtered_list = stem(filtered_list)
    parsed_input = tag(filtered_list)

    return parsed_input
