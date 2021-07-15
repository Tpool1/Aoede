import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import random
import os

from speech_txt_conv import recorder, sph_txt, txt_sph

class NLP: 

    def play_text(self, text):
        s = txt_sph(str(text), "bot_recording.mp3")
        s.translate()
        s.play()

    def get_message(self): 
        r = recorder(freq=44100, duration=7, file_name="user_recording.wav")
        r.record()

        conv = sph_txt(r.file_name)
        conv.translate()
        text = conv.text
        
        text = text.lower()

        return text

    # function to partition text into sentences, words, etc 
    def partition(self, text): 

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

    # function that filters out stopwords (an, in, is, etc.)
    def filterStops(self, text):

        stops = set(stopwords.words("english"))
        filtered_list = []
        for sent in text: 
            for word in sent: 
                if word.casefold() not in stops: 
                    filtered_list.append(word)

        return filtered_list

    # function to extract primitive meaning of words
    def stem(self, filtered_list):
        stemmer = WordNetLemmatizer()
        
        stemmed_words = []
        for word in filtered_list: 
            stem_word = stemmer.lemmatize(word)
            stemmed_words.append(stem_word)

        return stemmed_words

    # function to classify different parts of the text
    def tag(self, stemmed_words):
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

    def remove_breaks(self, val_list):
        i = 0
        for var in val_list: 
            new_var = var.replace('\n','')
            val_list[i] = new_var
            i = i + 1 
        
        return val_list

    def get_model(self):
        # identify model

        model_list = os.listdir("src/plugins")

        i = 0
        for model in model_list:
            model = model.lower()
            model_list[i] = model
            i = i + 1

        model_found = False
        while not model_found:
            audio_received = False
            self.play_text("Which model will we be using today?")

            while not audio_received:
                try:
                    response = self.get_message()
                    response = self.partition(response)
                    filtered_list = self.filterStops(response)
                    filtered_list = self.stem(filtered_list)
                    tags = self.tag(filtered_list)
                    audio_received = True
                except:
                    self.play_text("I can't hear you. Try again.")
            
            # check if a model name is inside text
            i = 0
            for model in model_list:

                # remove .py extension
                model = model[:-3]
                
                model_list[i] = model

                i = i + 1

            i = 0
            for tag in list(tags.values()):
                if tag in model_list:
                    model_found = True
                    model_use = model_list[i]

                i = i + 1
            
            if not model_found: 
                err = "Sorry, I can't find that. Maybe I misunderstood."
                self.play_text(err)

    def get_info(self): 
        # identify variable, value, and unit

        self.play_text("Tell me about the patient now")
        response = self.get_message()
        response = self.partition(response)
        filtered_list = self.filterStops(response)
        filtered_list = self.stem(filtered_list)
        tags = self.tag(filtered_list)

        NN_keys = []
        for key in tags.keys(): 
            if key[:2] == 'NN': 
                NN_keys.append(key)

        # read list of possible variables 
        f = open("data\\variables.txt")
        var_list = f.readlines()

        # remove line breakers throughout var list
        var_list = self.remove_breaks(var_list)
        
        var_found = False
        while not var_found: 
            # check if a variable name is inside of text
            for var in var_list: 
                if var in list(tags.values()): 
                    self.variable = var
                    var_found = True
                    break
                else: 
                    var_found = False
                    self.variable = "My apologies, what is the variable you speak of?"
                    
            self.play_text(self.variable)

        self.respond()

        val_found = False
        while not val_found:

            # identify numerical values in text for var value
            for var in tags.values(): 
                if var.isnumeric(): 
                    self.value = var
                    val_found = True
                    break
                else:
                    val_found = False
                    self.value = "Pardon me, what is the value for this variable?"
                
            self.play_text(self.value)
        
        self.respond()

        # identify unit by checking in units.txt
        f = open("data\\units.txt")
        unit_list = f.readlines()

        unit_list = self.remove_breaks(unit_list)

        unit_found = False
        while not unit_found:

            for unit in unit_list: 
                if unit in tags.values():
                    self.val_unit = unit
                    unit_found = True
                    break
                else: 
                    unit_found = False
                    self.val_unit = "I am sorry. I could not identify the unit. Please try again."
                
            self.play_text(self.val_unit)
        
        self.respond()

    def respond(self):

        affirmations = ['Okay, got it!', 'Alright', 'Okay']
        
        aff_choice = random.choice(affirmations)

        self.play_text(aff_choice)

    def run(self): 
        self.get_model()
        self.get_info()
    