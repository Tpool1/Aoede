from nltk.corpus import stopwords

# function that filters out stopwords (an, in, is, etc.)
def filterStops(text):

    stops = set(stopwords.words("english"))
    filtered_list = []
    for sent in text: 
        for word in sent: 
            if word.casefold() not in stops: 
                filtered_list.append(word)

    return filtered_list