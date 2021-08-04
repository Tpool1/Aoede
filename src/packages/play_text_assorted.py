from packages.play_text import play_text
from packages.partition import partition
from packages.filterStops import filterStops
from packages.stem import stem
from packages.tag import tag
from packages.get_word_synonyms import get_word_synonyms

import random

# function to play a set of text with some words ambiguously replaced with synonyms as to make the speech more natural

def play_text_assorted(text):

    original_text_list = partition(text)
    word_list = filterStops(original_text_list)
    word_list = stem(word_list)
    tag_dict = tag(word_list)

    # dictionary to store original words with the synonyms chosen
    word_syn_dict = {}
    for word in word_list:

        # if block fails with type error, word is not valid and can be ignored
        try:
            synonyms = get_word_synonyms(word)
            syn_chosen = random.choice(synonyms)

            word_syn_dict[word] = syn_chosen
        except TypeError:
            pass

    i = 0
    text_list = []
    for sent in original_text_list:
        j = 0 
        for word in sent:
            if word in list(word_syn_dict.keys()):
                if word in list(tag_dict.values()):
                    replacement = word_syn_dict[word]
                    sent[j] = replacement

            j = j + 1

        text_list.append(sent)

        i = i + 1

    for sent in text_list:
        it = map(str, sent)
        generated_text = next(it, '')
        for s in it:
            generated_text += ' ' + s
    
    play_text(generated_text)
    