from PyDictionary import PyDictionary

def get_word_synonyms(word):
    dictionary = PyDictionary()

    synonyms = dictionary.synonym(word)

    return synonyms
