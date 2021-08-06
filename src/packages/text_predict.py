import os
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# number of words model with use for training data
word_num = 6

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

tokenizer = Tokenizer()

tokenizer.fit_on_texts(files)

data = tokenizer.texts_to_sequences(files)

data = np.asarray(data, dtype=object)

x = np.array([])
y = np.array([])

i = 0
j = word_num
for text in data:
    while j < len(text):

        x_single = text[i:j]

        x = np.append(x, x_single)

        y_single = text[j]

        y = np.append(y, y_single)

        i = i + 1
        j = j + 1

    print(x.shape)
    print(y.shape)
        