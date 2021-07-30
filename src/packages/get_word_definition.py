from PyDictionary import PyDictionary

def get_word_definition(word):
    dictionary = PyDictionary()

    definition = list(dictionary.meaning(word).values())

    return definition
