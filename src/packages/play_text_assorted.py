from play_text import play_text
from packages.partition import partition
from packages.filterStops import filterStops
from packages.stem import stem
from packages.get_word_synonyms import get_word_synonyms

import random

# function to play a set of text with some words ambiguously replaced with synonyms as to make the speech more natural

def play_text_assorted(text):
    original_text_list = partition(text)
    word_list = filterStops(original_text_list)
    word_list = stem(word_list)

    # dictionary to store original words with the synonyms chosen
    word_syn_dict = {}
    for word in word_list:

        # if block fails, word is not valid and can be ignored
        try:
            synonyms = get_word_synonyms(word)
            syn_chosen = random.choice(synonyms)

            word_syn_dict[word] = syn_chosen
        except:
            pass

    i = 0
    for sent in original_text_list:
        j = 0 
        for word in sent:
            if word in list(word_syn_dict.keys()):
                replacement = word_syn_dict[word]
                sent[j] = replacement

            j = j + 1

        original_text_list[i] = sent

        i = i + 1

    generated_text = ''
    print(original_text_list)
    for sent in original_text_list:
        generated_text = generated_text.join(sent)

    print(generated_text)

    play_text(generated_text)
    

