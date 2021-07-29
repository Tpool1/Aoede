from nlp_tools.get_parsed_input import get_parsed_input
import re

# function to convert a date that is said verbally to numerical format (year-month-day) (Ex. 2021-07-23)

def voice_to_date():
    user_input = get_parsed_input()

    year = user_input['CD2']
    month = user_input['NNP']
    day = user_input['CD']

    day = re.sub("[^0-9]", "", day)
    
    if len(day) != 2:
        day = '0' + day

    month_dict = {'January': '01', 'February': '02', 'March': '03', 'April': '04', 
                  'May': '05', 'June': '06', 'July': '07', 'August': '08', 
                  'September': '09', 'October': '10', 'November': '11', 'December': '12'}

    month = month_dict[month]

    full_date = year + '-' + month + '-' + day

    return full_date
