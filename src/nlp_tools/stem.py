from nltk.stem import WordNetLemmatizer

# function to extract primitive meaning of words
def stem(filtered_list):
    stemmer = WordNetLemmatizer()
    
    stemmed_words = []
    for word in filtered_list: 
        stem_word = stemmer.lemmatize(word)
        stemmed_words.append(stem_word)

    return stemmed_words