import os
from keras_preprocessing import text
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class sentiment_identifier:

    def __init__(self, load_model=True):

        self.load_model = load_model

    def get_model(self):
        df = pd.read_csv('data\\nlp_datasets\\trainingandtestdata\\training.1600000.processed.noemoticon.csv', low_memory=False, encoding='latin-1')

        text_col = list(df[list(df.columns)[-1]])

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(text_col)

        x = self.tokenizer.texts_to_sequences(text_col)
        
        # pad x data to retain standard dimension
        x = keras.preprocessing.sequence.pad_sequences(x, padding='post')

        # get first column as y data
        y = df[list(df.columns)[0]]

        x_train, self.x_test, y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        if not self.load_model:
            input = keras.Input(shape=(x_train.shape[1]))

            x = layers.Dense(6, activation='relu')(input)
            x = layers.Dense(3, activation='relu')(x)
            x = layers.Dense(2, activation='relu')(x)

            output = layers.Dense(1, activation='linear')(x)

            model = keras.Model(inputs=input, outputs=output)

            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=64, epochs=10)

            model.save('data\\saved_models\\sentiment_identification\\keras_sentiment_model.h5')

        else:

            model = keras.models.load_model('data\\saved_models\\sentiment_identification\\keras_sentiment_model.h5')

        return model

    def get_distribution(self, series):

        series_len = len(series)
        
        counts_dict = series.value_counts().to_dict()

        percent_list = []
        for count in list(counts_dict.values()):
            percent = (count/series_len)
            percent = round(percent, 0)
            percent_list.append(percent)

        count_keys = list(counts_dict.keys())

        percent_dict = dict(zip(count_keys, percent_list))

        return percent_dict
    
    def evaluate(self, model):
        model.evalutate(self.x_test)
        print(self.y_test)

    def get_sentiment(self, text):

        model = self.get_model()

        text = [text]

        text = list(self.tokenizer.texts_to_sequences(text))

        model_input_shape = model.layers[0].input_shape[0][1]
        
        zeros_required = model_input_shape - len(text[0])

        for sent in text:
            for i in range(zeros_required):
                sent.append(0)

        print(model.predict(text))
