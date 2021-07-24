from nltk.tokenize import word_tokenize, sent_tokenize

# function to partition text into sentences, words, etc 
def partition(text): 

    # separate sentences 
    text = sent_tokenize(text)

    # get list of words for each sentence
    word_sent_list = []
    for sent in text: 

        # get list of words 
        text = word_tokenize(sent)

        word_sent_list.append(text)

    text = word_sent_list

    return text
    