from play_text import play_text
from packages.get_parsed_input import get_parsed_input

import json

# dictionary to put user data throughout setup process
user_data = {}

play_text('Hello! My name is Asclepius. Let me ask you some questions to get to know you then we will be set.')

name_found = False
while not name_found:
    play_text('What is your name?')
    tags = get_parsed_input()
    name = list(tags.values())[0]
    play_text('Your name is ' + name)
    play_text('Is this correct?')
    user_response = get_parsed_input()
    yn = list(user_response.values())[0]
    if yn == 'yes':
        name_found = True
    else:
        play_text("Okay, let's try again.")

user_data['name'] = name

with open('data\\user.json', 'w') as f:
    json.dump(user_data, f)