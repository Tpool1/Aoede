from googlesearch import search
from packages.get_parsed_input import get_parsed_input
from packages.play_text import play_text

def get_web_results():
    
    play_text('What would you like to search for today?')
    query = get_parsed_input()

    url_list = []
    for result in search(query, tld="co.in", num=5, stop=5, pause=2):
        url_list.append(result)

    print(url_list)
