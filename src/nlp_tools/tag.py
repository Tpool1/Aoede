import nltk

# function to classify different parts of the text
def tag(stemmed_words):
    tags = nltk.pos_tag(stemmed_words)

    # convert list of tuples to dict
    i = 0 
    tag_dict = {}
    for tup in tags: 
        word = tup[0]
        tag = tup[1]

        # check if tag already in dict
        if tag in tag_dict: 
            tag = tag + str(i)
        
        tag_dict[tag] = word

        i = i + 1 

    tags = tag_dict

    return tags
    