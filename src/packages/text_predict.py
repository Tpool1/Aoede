import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split

class text_predict:
    
    def __init__(self, load_model=True):

        self.load_model = load_model

    def get_model(self, word_num):

        # number of words model with use for training data
        self.word_num = word_num

        data_dir = 'data\\nlp_datasets\\gutenberg'
        file_list = os.listdir(data_dir)

        # get full path for all data files
        i = 0
        for file in file_list:
            full_path = os.path.join(data_dir, file)
            file_list[i] = full_path
            i = i + 1

        files = []
        for path in file_list:
            file = open(path, 'r').read()
            files.append(file)

        self.tokenizer = Tokenizer()

        self.tokenizer.fit_on_texts(files)

        data = self.tokenizer.texts_to_sequences(files)

        data = np.asarray(data, dtype=object)

        x = np.empty(shape=(852250, 6), dtype=int)
        y = np.empty(shape=(852250,), dtype=int)

        i = 0
        j = self.word_num
        for text in data:
            while j < len(text):

                x_single = text[i:j]

                x[i:j] = x_single

                y_single = text[j]

                y[i] = y_single

                i = i + 1
                j = j + 1

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        if not self.load_model:

            input = keras.Input(shape=(6))

            x = layers.Dense(6, activation='relu')(input)
            x = layers.Dense(3, activation='relu')(x)
            x = layers.Dense(2, activation='relu')(x)

            output = layers.Dense(1, activation='linear')(x)

            model = keras.Model(inputs=input, outputs=output)

            model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=16, epochs=20)

            model.save('data\\saved_models\\text_prediction\\keras_word_prediction_model.h5')

        else:

            model = keras.models.load_model('data\\saved_models\\text_prediction\\keras_word_prediction_model.h5')

        return model

    def predict(self, text):

        # get num of words in text
        word_list = text.split()

        num_words = len(word_list)

        model = self.get_model(num_words)

        text = [text]
        text_sequence = list(self.tokenizer.texts_to_sequences(text))

        prediction = model.predict(text_sequence)[0][0]
        prediction = round(prediction, 0)

        text_prediction = self.tokenizer.sequences_to_texts([[prediction]])

        return text_prediction
